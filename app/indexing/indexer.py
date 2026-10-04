"""Build the retrieval representation of a dataset: one search document per ICD entity.

Each document carries code, title, description, hierarchy path, inclusion terms, source
synonyms/index terms, notes, parent and child context and dataset metadata, and keeps the
authoritative record_id (node_id) and dataset_id.

Lexical index: weighted tsvector (A: code/title/synonyms/index terms, B: inclusions/description,
C: hierarchy/notes/children). Exclusion text is shown in `content` (clearly labelled) but kept
out of the tsvector and the embedding input, so a query never matches a code through what that
code excludes.

Embeddings are optional, computed only for new/changed content, and labelled with model and
dimension. Remote providers are refused unless the dataset's licence metadata allows it.
"""

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.constants import SEARCH_DOCUMENT_TYPE_RECORD, RuleType, TermType
from app.core.exceptions import LicenceRestrictionError
from app.core.text import ts_config_for
from app.indexing.embeddings import EmbeddingProvider
from app.models import IcdDataset, IcdIndexEntry, IcdNode, IcdRule, IcdSearchDocument, IcdTerm
from app.models.search_document import HNSW_INDEX_PREFIX

logger = logging.getLogger(__name__)

MAX_CHILDREN_IN_CONTEXT = 25
HNSW_MAX_DIMENSION = 2000  # pgvector limit for HNSW on `vector`
_EXCLUDES_PREFIX = "Excludes:"

_UPSERT_SQL = text(
    """
    INSERT INTO icd_search_documents
        (dataset_id, node_id, document_type, content, content_hash, metadata, search_vector)
    VALUES (
        :dataset_id, :node_id, :document_type, :content, :content_hash,
        CAST(:metadata AS jsonb),
        setweight(to_tsvector(CAST(:cfg AS regconfig), :weight_a), 'A')
        || setweight(to_tsvector(CAST(:cfg AS regconfig), :weight_b), 'B')
        || setweight(to_tsvector(CAST(:cfg AS regconfig), :weight_c), 'C')
    )
    ON CONFLICT ON CONSTRAINT uq_icd_search_documents_dataset_node_type DO UPDATE SET
        content = EXCLUDED.content,
        content_hash = EXCLUDED.content_hash,
        metadata = EXCLUDED.metadata,
        search_vector = EXCLUDED.search_vector,
        updated_at = now()
    """
)


def embedding_text(content: str) -> str:
    """The embedded text: the document content minus exclusion lines."""
    return "\n".join(line for line in content.split("\n") if not line.startswith(_EXCLUDES_PREFIX))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def allows_remote_processing(dataset: IcdDataset) -> bool:
    metadata = dataset.metadata_ or {}
    extra = metadata.get("extra", {}) if isinstance(metadata.get("extra"), dict) else {}
    licence = metadata.get("licence", {}) if isinstance(metadata.get("licence"), dict) else {}
    return bool(extra.get("allow_remote_processing") or licence.get("allow_remote_processing"))


@dataclass
class IndexStats:
    embedded: int = 0
    embedding_skipped_unchanged: int = 0
    embedding_model: str | None = None
    vector_index: str | None = None
    duration_ms: int = 0


class SearchIndexer:
    def __init__(self, session: AsyncSession, provider: EmbeddingProvider | None = None) -> None:
        self._session = session
        self._provider = provider

    # --- documents ------------------------------------------------------------------------

    async def _load(self, dataset_id: int) -> dict[str, Any]:
        nodes = (
            (
                await self._session.execute(
                    select(IcdNode)
                    .where(IcdNode.dataset_id == dataset_id)
                    .order_by(IcdNode.sort_order, IcdNode.id)
                )
            )
            .scalars()
            .all()
        )
        terms: dict[int, dict[TermType, list[str]]] = {}
        for node_id, term_type, term in await self._session.execute(
            select(IcdTerm.node_id, IcdTerm.term_type, IcdTerm.term)
            .where(IcdTerm.dataset_id == dataset_id)
            .order_by(IcdTerm.id)
        ):
            terms.setdefault(node_id, {}).setdefault(term_type, []).append(term)
        rules: dict[int, list[tuple[RuleType, str]]] = {}
        for node_id, rule_type, rule_text in await self._session.execute(
            select(IcdRule.node_id, IcdRule.rule_type, IcdRule.rule_text)
            .where(IcdRule.dataset_id == dataset_id, IcdRule.node_id.is_not(None))
            .order_by(IcdRule.id)
        ):
            rules.setdefault(node_id, []).append((rule_type, rule_text))
        index_terms: dict[int, list[str]] = {}
        for node_id, lead, modifier in await self._session.execute(
            select(IcdIndexEntry.target_node_id, IcdIndexEntry.lead_term, IcdIndexEntry.modifier)
            .where(
                IcdIndexEntry.dataset_id == dataset_id,
                IcdIndexEntry.target_node_id.is_not(None),
            )
            .order_by(IcdIndexEntry.id)
        ):
            index_terms.setdefault(node_id, []).append(f"{lead}, {modifier}" if modifier else lead)
        return {"nodes": nodes, "terms": terms, "rules": rules, "index_terms": index_terms}

    @staticmethod
    def _label(node: IcdNode) -> str:
        prefix = f"Chapter {node.code}" if node.node_type.value == "CHAPTER" else node.code
        return f"{prefix} {node.title}".strip() if prefix else node.title

    def _document(
        self,
        node: IcdNode,
        dataset: IcdDataset,
        by_id: dict[int, IcdNode],
        children: dict[int, list[IcdNode]],
        data: dict[str, Any],
    ) -> dict[str, Any]:
        path: list[IcdNode] = []
        current = by_id.get(node.parent_id) if node.parent_id else None
        while current is not None and len(path) < 20:
            path.append(current)
            current = by_id.get(current.parent_id) if current.parent_id else None
        path.reverse()
        node_terms = data["terms"].get(node.id, {})
        inclusions = node_terms.get(TermType.INCLUSION, [])
        synonyms = node_terms.get(TermType.SYNONYM, []) + node_terms.get(
            TermType.ALTERNATIVE_TERM, []
        )
        abbreviations = node_terms.get(TermType.ABBREVIATION, [])
        index_terms = data["index_terms"].get(node.id, [])
        node_rules = data["rules"].get(node.id, [])
        exclusions = [t for r, t in node_rules if r == RuleType.EXCLUDE]
        notes = [t for r, t in node_rules if r != RuleType.EXCLUDE]
        kids = children.get(node.id, [])
        hierarchy = " > ".join(self._label(p) for p in path)
        # Source synonyms/abbreviations/index terms of classification ancestors describe their
        # subdivisions too ("heart failure" on B01 applies to B01.0 "Acute ..."). They are
        # inherited as context (weight B), never copied as the child's own terms.
        inherited: list[str] = []
        for ancestor in path:
            if ancestor.node_type.value in {"CATEGORY", "SUBCATEGORY", "CODE"}:
                ancestor_terms = data["terms"].get(ancestor.id, {})
                inherited += ancestor_terms.get(TermType.SYNONYM, [])
                inherited += ancestor_terms.get(TermType.ABBREVIATION, [])
                inherited += data["index_terms"].get(ancestor.id, [])

        lines = [f"{node.code or ''} {node.title}".strip()]
        if node.description:
            lines.append(node.description)
        if hierarchy:
            lines.append(f"Hierarchy: {hierarchy}")
        if inclusions:
            lines.append("Includes: " + "; ".join(inclusions))
        if synonyms:
            lines.append("Synonyms: " + "; ".join(synonyms))
        if abbreviations:
            lines.append("Abbreviations: " + "; ".join(abbreviations))
        if index_terms:
            lines.append("Index terms: " + "; ".join(index_terms))
        if inherited:
            lines.append("Parent terms: " + "; ".join(dict.fromkeys(inherited)))
        if notes:
            lines.append("Instructions: " + " | ".join(notes))
        if kids:
            lines.append(
                "Subdivisions: " + "; ".join(self._label(k) for k in kids[:MAX_CHILDREN_IN_CONTEXT])
            )
        for exclusion in exclusions:
            lines.append(f"{_EXCLUDES_PREFIX} {exclusion}")
        lines.append(
            f"Dataset: {dataset.system} {dataset.version} ({dataset.country}, {dataset.language})"
        )
        content = "\n".join(lines)
        metadata = {
            "code": node.code,
            "normalized_code": node.normalized_code,
            "title": node.title,
            "node_type": node.node_type.value,
            "is_selectable": node.is_selectable,
            "hierarchy_path": [
                {"node_id": p.id, "code": p.code, "title": p.title, "node_type": p.node_type.value}
                for p in path
            ],
            "dataset": {
                "id": dataset.id,
                "system": dataset.system,
                "version": dataset.version,
                "country": dataset.country,
                "language": dataset.language,
            },
            "exclusions": exclusions,
            "inherited_terms": list(dict.fromkeys(inherited)),
        }
        return {
            "dataset_id": dataset.id,
            "node_id": node.id,
            "document_type": SEARCH_DOCUMENT_TYPE_RECORD,
            "content": content,
            "content_hash": sha256_text(content),
            "metadata": json.dumps(metadata),
            "cfg": ts_config_for(dataset.language),
            "weight_a": " ".join(
                filter(
                    None,
                    [
                        node.code,
                        node.normalized_code,
                        node.title,
                        *synonyms,
                        *abbreviations,
                        *index_terms,
                    ],
                )
            ),
            "weight_b": " ".join(filter(None, [*inclusions, node.description, *inherited])),
            "weight_c": " ".join([hierarchy, *notes, *(k.title for k in kids)]),
        }

    async def build_documents(self, dataset: IcdDataset) -> int:
        data = await self._load(dataset.id)
        nodes: list[IcdNode] = list(data["nodes"])
        by_id = {node.id: node for node in nodes}
        children: dict[int, list[IcdNode]] = {}
        for node in nodes:
            if node.parent_id:
                children.setdefault(node.parent_id, []).append(node)
        documents = [self._document(n, dataset, by_id, children, data) for n in nodes]
        for start in range(0, len(documents), 500):
            await self._session.execute(_UPSERT_SQL, documents[start : start + 500])
        return len(documents)

    # --- embeddings -----------------------------------------------------------------------

    async def embed_documents(self, dataset: IcdDataset, *, batch_size: int = 64) -> IndexStats:
        stats = IndexStats()
        provider = self._provider
        if provider is None:
            return stats
        if provider.remote and not allows_remote_processing(dataset):
            raise LicenceRestrictionError(
                f"Dataset {dataset.system} {dataset.version} does not allow remote processing; "
                "refusing to send its content to a remote embedding provider.",
                details={"dataset_id": dataset.id},
            )
        started = time.perf_counter()
        rows = (
            await self._session.execute(
                select(
                    IcdSearchDocument.id,
                    IcdSearchDocument.content,
                    IcdSearchDocument.embedding_model,
                    IcdSearchDocument.embedding_dimension,
                    IcdSearchDocument.embedding_content_hash,
                ).where(IcdSearchDocument.dataset_id == dataset.id)
            )
        ).all()
        pending: list[tuple[int, str, str]] = []
        for doc_id, content, model, dimension, content_hash in rows:
            source = embedding_text(content)
            # Identity is (model, dimension, text): the same model name at another dimension
            # (e.g. OpenAI `dimensions`) is a different embedding space and is re-embedded.
            source_hash = sha256_text(f"{provider.model}:{provider.dimension}:{source}")
            if (
                model == provider.model
                and dimension == provider.dimension
                and content_hash == source_hash
            ):
                stats.embedding_skipped_unchanged += 1
                continue
            pending.append((doc_id, source, source_hash))
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            vectors = await provider.embed([source for _, source, _ in batch])
            now = datetime.now(UTC)
            # ORM bulk UPDATE by primary key: one executemany per batch.
            await self._session.execute(
                update(IcdSearchDocument),
                [
                    {
                        "id": doc_id,
                        "embedding": vector,
                        "embedding_model": provider.model,
                        "embedding_dimension": provider.dimension,
                        "embedding_content_hash": source_hash,
                        "embedded_at": now,
                    }
                    for (doc_id, _, source_hash), vector in zip(batch, vectors, strict=True)
                ],
            )
            stats.embedded += len(batch)
        stats.embedding_model = provider.model
        stats.vector_index = await self.ensure_vector_index(provider.model, provider.dimension)
        stats.duration_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "embeddings generated",
            extra={
                "dataset_id": dataset.id,
                "embedding_model": provider.model,
                "embedded": stats.embedded,
                "skipped_unchanged": stats.embedding_skipped_unchanged,
                "duration_ms": stats.duration_ms,
            },
        )
        return stats

    async def ensure_vector_index(self, model: str, dimension: int) -> str | None:
        """Partial HNSW expression index for one (model, dimension). Idempotent.

        The predicate includes the dimension, so rows of another dimension can never reach the
        `::vector(dim)` cast (which would fail) and queries must filter on both columns.
        """
        if dimension > HNSW_MAX_DIMENSION:
            logger.warning(
                "embedding dimension exceeds HNSW limit; vector search will scan",
                extra={"embedding_model": model, "dimension": dimension},
            )
            return None
        slug = hashlib.sha1(f"{model}:{dimension}:dimension-scoped".encode()).hexdigest()[:12]
        name = f"{HNSW_INDEX_PREFIX}{slug}"
        literal = model.replace("'", "''")
        if not re.fullmatch(r"[a-z0-9_]+", name):  # defensive: identifiers are interpolated
            raise ValueError("invalid index name")
        await self._session.execute(
            text(
                f"CREATE INDEX IF NOT EXISTS {name} ON icd_search_documents "
                f"USING hnsw ((embedding::vector({int(dimension)})) vector_cosine_ops) "
                f"WHERE embedding_model = '{literal}' AND embedding_dimension = {int(dimension)}"
            )
        )
        return name
