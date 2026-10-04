"""Hybrid retrieval: exact code + lexical (FTS) + fuzzy (trigram) + semantic (pgvector) +
hierarchy context + source index terms, with every component score exposed.

hybrid = sum(weight_i * score_i) / sum(weight_i over *available* components)

Semantic similarity is "available" only when the configured embedding provider's model has
vectors for this dataset; otherwise its weight is dropped rather than scored as zero.
Candidates always come from the database (record ids of the resolved dataset only).
"""

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.constants import NodeStatus, NodeType
from app.core.text import normalize_text, tokenize, ts_config_for
from app.indexing.embeddings import EmbeddingProvider
from app.ingestion.codes import looks_like_code
from app.models import IcdDataset, IcdNode
from app.repositories.icd_repository import IcdRepository
from app.repositories.search_repository import SearchRepository, TermMatch

logger = logging.getLogger(__name__)

STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "the",
        "to",
        "with",
        "without",
        "other",
        "unspecified",
        "disorder",
        "disorders",
        "condition",
        "conditions",
    ]
)
COMPONENTS = ("exact", "lexical", "fuzzy", "semantic", "hierarchy", "index_term")


@dataclass
class ComponentScores:
    exact: float = 0.0
    lexical: float = 0.0
    fuzzy: float = 0.0
    semantic: float | None = None  # None: semantic retrieval unavailable for this dataset
    hierarchy: float = 0.0
    index_term: float = 0.0

    def as_dict(self) -> dict[str, float | None]:
        return {name: getattr(self, name) for name in COMPONENTS}


@dataclass
class RetrievalCandidate:
    node: IcdNode
    ancestors: list[IcdNode]
    scores: ComponentScores
    hybrid_score: float = 0.0
    evidence: list[TermMatch] = field(default_factory=list)


@dataclass
class RetrievalFilters:
    node_types: Sequence[NodeType] | None = None
    selectable_only: bool = False
    include_disabled: bool = False


@dataclass
class RetrievalResult:
    candidates: list[RetrievalCandidate]
    semantic_available: bool
    pool_size: int
    duration_ms: int


def hybrid_score(scores: ComponentScores, weights: dict[str, float]) -> float:
    total, weight_sum = 0.0, 0.0
    for name in COMPONENTS:
        value = getattr(scores, name)
        if value is None:
            continue
        total += weights.get(name, 0.0) * value
        weight_sum += weights.get(name, 0.0)
    return round(total / weight_sum, 4) if weight_sum else 0.0


def _overlap(query_tokens: set[str], text: str) -> float:
    if not query_tokens:
        return 0.0
    return len(query_tokens & set(tokenize(text))) / len(query_tokens)


class HybridRetriever:
    def __init__(
        self,
        session: AsyncSession,
        settings: Settings,
        provider: EmbeddingProvider | None = None,
    ) -> None:
        self._search = SearchRepository(session)
        self._records = IcdRepository(session)
        self._settings = settings
        self._provider = provider

    async def retrieve(
        self,
        dataset: IcdDataset,
        queries: Sequence[str],
        *,
        top_k: int,
        filters: RetrievalFilters | None = None,
        include_node_ids: Sequence[int] = (),
    ) -> RetrievalResult:
        """`include_node_ids` forces known records (e.g. an exclusion's redirect target) into
        the pool so they are scored on the same terms as retrieved candidates."""
        started = time.perf_counter()
        filters = filters or RetrievalFilters()
        variants = [q.strip() for q in dict.fromkeys(queries) if q and q.strip()]
        settings = self._settings
        per_method = settings.retrieval_candidates_per_method
        threshold = settings.retrieval_fuzzy_threshold
        cfg = ts_config_for(dataset.language)

        exact: dict[int, float] = {}
        lexical: dict[int, float] = {}
        fuzzy: dict[int, float] = {}
        index_term: dict[int, float] = {}
        semantic: dict[int, float] = {}
        evidence: dict[int, list[TermMatch]] = {}

        def keep_best(target: dict[int, float], matches: list[TermMatch], *, track: bool) -> None:
            for match in matches:
                if match.score > target.get(match.node_id, 0.0):
                    target[match.node_id] = match.score
                if track:
                    evidence.setdefault(match.node_id, []).append(match)

        # 1. candidate generation, per query variant ------------------------------------------
        for variant in variants:
            if looks_like_code(variant):
                keep_best(exact, await self._search.exact_code(dataset.id, variant), track=True)
            for node_id, score in (
                await self._search.lexical(dataset.id, cfg, variant, limit=per_method)
            ).items():
                lexical[node_id] = max(lexical.get(node_id, 0.0), score)
            keep_best(
                fuzzy,
                await self._search.fuzzy_titles(
                    dataset.id, variant, limit=per_method, threshold=threshold
                ),
                track=True,
            )
            keep_best(
                index_term,
                await self._search.source_terms(
                    dataset.id, normalize_text(variant), limit=per_method, threshold=threshold
                ),
                track=True,
            )

        semantic_model = None
        query_vectors: list[list[float]] = []
        if self._provider is not None and variants:
            models = await self._search.embedded_models(dataset.id)
            if (self._provider.model, self._provider.dimension) in models:
                semantic_model = self._provider.model
                query_vectors = await self._provider.embed(variants)
                for vector in query_vectors:
                    for node_id, score in (
                        await self._search.semantic(
                            dataset.id,
                            semantic_model,
                            self._provider.dimension,
                            vector,
                            limit=per_method,
                        )
                    ).items():
                        semantic[node_id] = max(semantic.get(node_id, 0.0), score)

        pool = sorted(
            set(exact)
            | set(lexical)
            | set(fuzzy)
            | set(index_term)
            | set(semantic)
            | set(include_node_ids)
        )

        # 2. complete every component for the whole pool (bounded, no N+1) --------------------
        if pool:
            for variant in variants:
                for node_id, score in (
                    await self._search.lexical(
                        dataset.id, cfg, variant, limit=len(pool), node_ids=pool
                    )
                ).items():
                    lexical[node_id] = max(lexical.get(node_id, 0.0), score)
                keep_best(
                    fuzzy,
                    await self._search.fuzzy_titles(
                        dataset.id, variant, limit=len(pool), threshold=threshold, node_ids=pool
                    ),
                    track=False,
                )
                keep_best(
                    index_term,
                    await self._search.source_terms(
                        dataset.id,
                        normalize_text(variant),
                        limit=len(pool) * 10,
                        threshold=threshold,
                        node_ids=pool,
                    ),
                    track=True,
                )
            if semantic_model is not None:
                for vector in query_vectors:
                    for node_id, score in (
                        await self._search.semantic(
                            dataset.id,
                            semantic_model,
                            self._provider.dimension,  # type: ignore[union-attr]
                            vector,
                            limit=len(pool),
                            node_ids=pool,
                        )
                    ).items():
                        semantic[node_id] = max(semantic.get(node_id, 0.0), score)

        nodes = await self._records.get_many(dataset.id, pool)
        ancestors = await self._records.ancestors_of_many(dataset.id, list(nodes))
        query_tokens = {t for v in variants for t in tokenize(v)} - STOPWORDS
        weights = settings.retrieval_weights()

        candidates: list[RetrievalCandidate] = []
        for node_id, node in nodes.items():
            if filters.node_types and node.node_type not in filters.node_types:
                continue
            if filters.selectable_only and not node.is_selectable:
                continue
            if not filters.include_disabled and node.status != NodeStatus.ACTIVE:
                continue
            chain = ancestors.get(node_id, [])
            scores = ComponentScores(
                exact=exact.get(node_id, 0.0),
                lexical=lexical.get(node_id, 0.0),
                fuzzy=round(fuzzy.get(node_id, 0.0), 4),
                semantic=round(semantic.get(node_id, 0.0), 4) if semantic_model else None,
                hierarchy=round(_overlap(query_tokens, " ".join(a.title for a in chain)), 4),
                index_term=round(index_term.get(node_id, 0.0), 4),
            )
            matches = sorted(evidence.get(node_id, []), key=lambda m: -m.score)
            unique: dict[tuple[str, str], TermMatch] = {}
            for match in matches:
                unique.setdefault((match.match_type, match.matched_text), match)
            candidates.append(
                RetrievalCandidate(
                    node=node,
                    ancestors=chain,
                    scores=scores,
                    hybrid_score=hybrid_score(scores, weights),
                    evidence=list(unique.values())[:5],
                )
            )
        # An exact code match always ranks first, whatever the weights; then hybrid score.
        candidates.sort(key=lambda c: (-c.scores.exact, -c.hybrid_score, c.node.code or ""))
        duration = int((time.perf_counter() - started) * 1000)
        logger.info(
            "retrieval completed",
            extra={
                "dataset_id": dataset.id,
                "variants": len(variants),
                "pool_size": len(pool),
                "returned": min(top_k, len(candidates)),
                "semantic_available": semantic_model is not None,
                "duration_ms": duration,
            },
        )
        return RetrievalResult(
            candidates=candidates[:top_k],
            semantic_available=semantic_model is not None,
            pool_size=len(pool),
            duration_ms=duration,
        )
