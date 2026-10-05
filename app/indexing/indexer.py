"""Build the retrieval representation of a dataset: one search document per ICD entity.

Retrieval unit = one logical ICD record (chapter, block, category, code); never an arbitrary
chunk of source text. Each document keeps the authoritative record_id (node_id) and dataset_id.

Three views of each record are produced:
* `content`: display/context text (title, description, hierarchy, inclusions, terms, notes,
  instructions, subdivisions, exclusions clearly labelled, dataset).
* `search_vector` (lexical): A = code/title/synonyms/abbreviations/index terms,
  B = inclusions/description/inherited parent terms, C = hierarchy/notes/subdivision titles.
* `semantic_text` (embedded): positive evidence only: code + title, description, hierarchy
  path, inclusions, synonyms/abbreviations/index terms, inherited parent terms, notes.

Exclusions and instructions that name *other* conditions (code first / use additional code /
code also / see / see also) are kept out of both the lexical index and the semantic text, so a
query never matches a record through a condition it excludes or merely refers to. They remain
available to the rule engine.

Embeddings are computed per embedding space (provider, model, dimension, normalisation), only
for new or changed semantic text, in committed batches. Remote providers are refused unless the
dataset's licence metadata allows it.
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.constants import SEARCH_DOCUMENT_TYPE_RECORD, RuleType, TermType
from app.core.exceptions import EmbeddingProviderError, LicenceRestrictionError
from app.core.text import ts_config_for
from app.indexing.embeddings import EmbeddingProvider
from app.indexing.vector_space import Distance, VectorSpace
from app.models import IcdDataset, IcdIndexEntry, IcdNode, IcdRule, IcdSearchDocument, IcdTerm

logger = logging.getLogger(__name__)

MAX_CHILDREN_IN_CONTEXT = 25
_EXCLUDES_PREFIX = "Excludes:"
# Rule types whose text is positive evidence about the record itself. Other instruction
# types name *other* conditions and stay out of the lexical and semantic representations.
_SELF_DESCRIBING_RULES = {RuleType.NOTE, RuleType.INCLUDE}

_UPSERT_SQL = text(
    """
    INSERT INTO icd_search_documents
        (dataset_id, node_id, document_type, content, content_hash, semantic_text, metadata,
         search_vector)
    VALUES (
        :dataset_id, :node_id, :document_type, :content, :content_hash, :semantic_text,
        CAST(:metadata AS jsonb),
        setweight(to_tsvector(CAST(:cfg AS regconfig), :weight_a), 'A')
        || setweight(to_tsvector(CAST(:cfg AS regconfig), :weight_b), 'B')
        || setweight(to_tsvector(CAST(:cfg AS regconfig), :weight_c), 'C')
    )
    ON CONFLICT ON CONSTRAINT uq_icd_search_documents_dataset_node_type DO UPDATE SET
        content = EXCLUDED.content,
        content_hash = EXCLUDED.content_hash,
        semantic_text = EXCLUDED.semantic_text,
        metadata = EXCLUDED.metadata,
        search_vector = EXCLUDED.search_vector,
        updated_at = now()
    """
)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def embedding_identity(space: VectorSpace, semantic_text: str) -> str:
    """Deterministic identity of one stored vector: embedding space + embedded text."""
    return sha256_text(f"{space.key}\n{semantic_text}")


def allows_remote_processing(dataset: IcdDataset) -> bool:
    metadata = dataset.metadata_ or {}
    extra = metadata.get("extra", {}) if isinstance(metadata.get("extra"), dict) else {}
    licence = metadata.get("licence", {}) if isinstance(metadata.get("licence"), dict) else {}
    return bool(extra.get("allow_remote_processing") or licence.get("allow_remote_processing"))


@dataclass
class IndexStats:
    provider: str | None = None
    model: str | None = None
    dimension: int | None = None
    normalized: bool | None = None
    distance: str | None = None
    documents: int = 0
    embedded: int = 0
    skipped_unchanged: int = 0
    skipped_empty: int = 0
    failed: int = 0
    batches: int = 0
    vector_index: str | None = None
    embed_ms: int = 0
    write_ms: int = 0
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
        notes = [t for r, t in node_rules if r in _SELF_DESCRIBING_RULES]
        instructions = [
            t for r, t in node_rules if r != RuleType.EXCLUDE and r not in _SELF_DESCRIBING_RULES
        ]
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
            lines.append("Notes: " + " | ".join(notes))
        semantic_text = "\n".join(lines)  # positive evidence only (see module docstring)
        if instructions:
            lines.append("Instructions: " + " | ".join(instructions))
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
            "semantic_text": semantic_text,
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

    async def embed_documents(
        self,
        dataset: IcdDataset,
        *,
        batch_size: int = 32,
        force: bool = False,
        distance: Distance = "cosine",
        commit: bool = True,
    ) -> IndexStats:
        """Embed new/changed documents of a dataset in batches.

        Skips a document when its stored vector belongs to the same embedding space and was
        computed from the same semantic text (unless `force`). Each batch is written and (with
        `commit`) committed on its own, so an interrupted run resumes where it stopped: committed
        batches are skipped next time. A provider failure or a vector of the wrong length rolls
        back the current batch only, records the failed count and raises EmbeddingProviderError.
        """
        provider = self._provider
        if provider is None:
            return IndexStats()
        if provider.remote and not allows_remote_processing(dataset):
            raise LicenceRestrictionError(
                f"Dataset {dataset.system} {dataset.version} does not allow remote processing; "
                "refusing to send its content to a remote embedding provider.",
                details={"dataset_id": dataset.id},
            )
        space = VectorSpace.of(provider, distance)
        stats = IndexStats(
            provider=space.provider,
            model=space.model,
            dimension=space.dimension,
            normalized=space.normalized,
            distance=space.distance,
        )
        dataset_id = dataset.id
        started = time.perf_counter()
        rows = (
            await self._session.execute(
                select(
                    IcdSearchDocument.id,
                    IcdSearchDocument.semantic_text,
                    IcdSearchDocument.embedding_content_hash,
                    IcdSearchDocument.embedding_provider,
                    IcdSearchDocument.embedding_model,
                    IcdSearchDocument.embedding_dimension,
                    IcdSearchDocument.embedding_normalized,
                )
                .where(IcdSearchDocument.dataset_id == dataset_id)
                .order_by(IcdSearchDocument.id)
            )
        ).all()
        stats.documents = len(rows)
        pending: list[tuple[int, str, str]] = []
        for doc_id, semantic, stored_hash, prov, model, dim, normalized in rows:
            if not (semantic or "").strip():
                stats.skipped_empty += 1
                continue
            identity = embedding_identity(space, semantic)
            same_space = (prov, model, dim, normalized) == (
                space.provider,
                space.model,
                space.dimension,
                space.normalized,
            )
            if not force and same_space and stored_hash == identity:
                stats.skipped_unchanged += 1
                continue
            pending.append((doc_id, semantic, identity))

        batches = [pending[i : i + batch_size] for i in range(0, len(pending), batch_size)]
        for number, batch in enumerate(batches, start=1):
            try:
                t0 = time.perf_counter()
                vectors = await provider.embed([text_ for _, text_, _ in batch])
                stats.embed_ms += int((time.perf_counter() - t0) * 1000)
                for vector in vectors:
                    if len(vector) != space.dimension:  # providers check too; never store
                        raise EmbeddingProviderError(
                            f"Vector of length {len(vector)} for a {space.dimension}-dimensional "
                            "space"
                        )
                t1 = time.perf_counter()
                now = datetime.now(UTC)
                await self._session.execute(
                    update(IcdSearchDocument),
                    [
                        {
                            "id": doc_id,
                            "embedding": vector,
                            "embedding_provider": space.provider,
                            "embedding_model": space.model,
                            "embedding_dimension": space.dimension,
                            "embedding_normalized": space.normalized,
                            "embedding_content_hash": identity,
                            "embedded_at": now,
                        }
                        for (doc_id, _, identity), vector in zip(batch, vectors, strict=True)
                    ],
                )
                if commit:
                    await self._session.commit()
                stats.write_ms += int((time.perf_counter() - t1) * 1000)
            except Exception as exc:
                await self._session.rollback()
                stats.failed = len(pending) - stats.embedded
                logger.error(
                    "embedding batch failed; earlier batches are kept and will be skipped on retry",
                    extra={
                        "dataset_id": dataset_id,
                        "batch": number,
                        "batches": len(batches),
                        "failed": stats.failed,
                        "error_type": type(exc).__name__,
                    },
                )
                if isinstance(exc, EmbeddingProviderError):
                    raise
                raise EmbeddingProviderError(
                    f"Embedding failed in batch {number}/{len(batches)} ({type(exc).__name__})",
                    details={"embedded": stats.embedded, "failed": stats.failed},
                ) from exc
            stats.embedded += len(batch)
            stats.batches += 1
            logger.info(
                "embedding batch stored",
                extra={
                    "dataset_id": dataset_id,
                    "batch": number,
                    "batches": len(batches),
                    "embedded": stats.embedded,
                    "pending": len(pending) - stats.embedded,
                },
            )
        stats.vector_index = await self.ensure_vector_index(space)
        if commit:
            await self._session.commit()
        stats.duration_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "embeddings generated",
            extra={
                "dataset_id": dataset_id,
                "embedding_provider": space.provider,
                "embedding_model": space.model,
                "dimension": space.dimension,
                "embedded": stats.embedded,
                "skipped_unchanged": stats.skipped_unchanged,
                "skipped_empty": stats.skipped_empty,
                "duration_ms": stats.duration_ms,
            },
        )
        return stats

    async def ensure_vector_index(self, space: VectorSpace) -> str | None:
        """Partial HNSW index for one embedding space and distance. Idempotent.

        Its predicate is the space predicate the queries use, so rows of another space never
        reach its `::vector(dim)` cast, and its operator class matches the query operator.
        """
        if not space.indexable:
            logger.warning(
                "embedding dimension exceeds HNSW limit; vector search will scan",
                extra={"embedding_model": space.model, "dimension": space.dimension},
            )
            return None
        await self._session.execute(text(space.create_index_sql()))
        return space.index_name
