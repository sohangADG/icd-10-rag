"""Hybrid retrieval: exact code + lexical (FTS) + fuzzy (trigram) + semantic (pgvector) +
hierarchy context + source index terms, with every component score exposed.

hybrid = Σ wᵢ·scoreᵢ / Σ wᵢ over *available* components, on the absolute 0..1 scales of
app.retrieval.scoring. An exact code match always ranks first.

Semantic retrieval runs only when the configured provider's embedding space covers *every*
record document of the dataset (a partially embedded dataset would bias ranking towards the
embedded part). Otherwise, or when the provider fails, behaviour follows
SEMANTIC_RETRIEVAL_MODE:
* optional (default): semantic weight is dropped, the other signals still rank, and the result
  reports why (`semantic_status`);
* required: the request fails with RetrievalError (503). Nothing is ever invented.

Candidates always come from the database (record ids of the resolved dataset only).
"""

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.constants import NodeStatus, NodeType
from app.core.exceptions import AppError, RetrievalError
from app.core.text import normalize_text, tokenize, ts_config_for
from app.indexing.embeddings import EmbeddingProvider, similarity_floor
from app.indexing.vector_space import VectorSpace
from app.ingestion.codes import looks_like_code
from app.models import IcdDataset, IcdNode
from app.repositories.icd_repository import IcdRepository
from app.repositories.search_repository import SearchRepository, TermMatch
from app.retrieval.scoring import COMPONENTS, calibrate_similarity, calibrate_trigram
from app.retrieval.scoring import hybrid_score as _weighted

logger = logging.getLogger(__name__)

STOPWORDS = frozenset(
    {
        *("a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have"),
        *("in", "is", "it", "of", "on", "or", "the", "to", "with", "without", "other"),
        *("unspecified", "disorder", "disorders", "condition", "conditions"),
    }
)


class SemanticStatus(StrEnum):
    OK = "ok"
    DISABLED = "disabled"  # EMBEDDING_PROVIDER=none
    PROVIDER_UNAVAILABLE = "provider_unavailable"  # provider failed to initialise
    NOT_INDEXED = "not_indexed"  # dataset has no vectors in the provider's space
    INCOMPLETE_INDEX = "incomplete_index"  # only part of the dataset is embedded
    ERROR = "error"  # query embedding / vector search failed for this request


@dataclass
class ComponentScores:
    exact: float = 0.0
    lexical: float = 0.0
    fuzzy: float = 0.0
    semantic: float | None = None  # None: semantic retrieval unavailable (see status)
    hierarchy: float = 0.0
    index_term: float = 0.0
    semantic_raw: float | None = None  # uncalibrated vector similarity, for transparency

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
    semantic_status: SemanticStatus
    pool_size: int
    duration_ms: int
    embedding_space: dict[str, Any] | None = None
    timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def semantic_available(self) -> bool:
        return self.semantic_status is SemanticStatus.OK


def hybrid_score(scores: ComponentScores, weights: dict[str, float]) -> float:
    return _weighted(scores.as_dict(), weights)


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
        *,
        provider_status: SemanticStatus | None = None,
    ) -> None:
        """`provider_status` explains a missing provider (DISABLED or PROVIDER_UNAVAILABLE)."""
        self._search = SearchRepository(session)
        self._records = IcdRepository(session)
        self._settings = settings
        self._provider = provider
        self._provider_status = provider_status or (
            SemanticStatus.OK if provider is not None else SemanticStatus.DISABLED
        )

    def _degrade(self, status: SemanticStatus, detail: str) -> SemanticStatus:
        if self._settings.semantic_retrieval_mode == "required":
            raise RetrievalError(
                f"Semantic retrieval is required but unavailable ({status.value}): {detail}",
                details={"semantic_status": status.value},
            )
        logger.warning(
            "semantic retrieval unavailable; continuing with exact/lexical/fuzzy signals",
            extra={"semantic_status": status.value},
        )
        return status

    async def _semantic_space(
        self, dataset: IcdDataset
    ) -> tuple[SemanticStatus, VectorSpace | None]:
        if self._provider is None:
            if self._provider_status is SemanticStatus.DISABLED:
                return SemanticStatus.DISABLED, None  # explicit configuration, not a fault
            return self._degrade(self._provider_status, "embedding provider not loaded"), None
        space = VectorSpace.of(self._provider, self._settings.embedding_distance)
        embedded, total = await self._search.space_coverage(dataset.id, space)
        if embedded == 0:
            return self._degrade(SemanticStatus.NOT_INDEXED, f"no vectors for {space.key}"), None
        if embedded < total:
            return (
                self._degrade(
                    SemanticStatus.INCOMPLETE_INDEX, f"{embedded}/{total} documents embedded"
                ),
                None,
            )
        return SemanticStatus.OK, space

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
        timings: dict[str, float] = {}

        exact: dict[int, float] = {}
        lexical: dict[int, float] = {}
        fuzzy: dict[int, float] = {}
        index_term: dict[int, float] = {}
        semantic_raw: dict[int, float] = {}
        evidence: dict[int, list[TermMatch]] = {}

        def keep_best(target: dict[int, float], matches: list[TermMatch], *, track: bool) -> None:
            for match in matches:
                if match.score > target.get(match.node_id, 0.0):
                    target[match.node_id] = match.score
                if track:
                    evidence.setdefault(match.node_id, []).append(match)

        # 1. candidate generation, per query variant ------------------------------------------
        t0 = time.perf_counter()
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
        timings["lexical_fuzzy_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        semantic_status = SemanticStatus.DISABLED
        space: VectorSpace | None = None
        query_vectors: list[list[float]] = []
        if variants:
            semantic_status, space = await self._semantic_space(dataset)
        if space is not None:
            t1 = time.perf_counter()
            try:
                # Any provider failure (our errors or a raw library exception from the model)
                # degrades or fails according to SEMANTIC_RETRIEVAL_MODE, never a bare 500.
                query_vectors = await self._provider.embed_queries(variants)  # type: ignore[union-attr]
            except RetrievalError:
                raise
            except Exception as exc:  # noqa: BLE001
                detail = exc.message if isinstance(exc, AppError) else type(exc).__name__
                semantic_status = self._degrade(SemanticStatus.ERROR, detail)
                space, query_vectors = None, []
            timings["query_embedding_ms"] = round((time.perf_counter() - t1) * 1000, 2)
        if space is not None:
            # Database errors are not masked: the transaction would be unusable anyway.
            t2 = time.perf_counter()
            for vector in query_vectors:
                for node_id, similarity in (
                    await self._search.semantic(dataset.id, space, vector, limit=per_method)
                ).items():
                    semantic_raw[node_id] = max(semantic_raw.get(node_id, -1.0), similarity)
            timings["vector_query_ms"] = round((time.perf_counter() - t2) * 1000, 2)

        pool = sorted(
            set(exact)
            | set(lexical)
            | set(fuzzy)
            | set(index_term)
            | set(semantic_raw)
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
            if space is not None:
                for vector in query_vectors:
                    for node_id, similarity in (
                        await self._search.semantic(
                            dataset.id, space, vector, limit=len(pool), node_ids=pool
                        )
                    ).items():
                        semantic_raw[node_id] = max(semantic_raw.get(node_id, -1.0), similarity)

        nodes = await self._records.get_many(dataset.id, pool)
        ancestors = await self._records.ancestors_of_many(dataset.id, list(nodes))
        query_tokens = {t for v in variants for t in tokenize(v)} - STOPWORDS
        weights = settings.retrieval_weights()
        floor = similarity_floor(settings, self._provider) if space is not None else 0.0

        candidates: list[RetrievalCandidate] = []
        for node_id, node in nodes.items():
            if filters.node_types and node.node_type not in filters.node_types:
                continue
            if filters.selectable_only and not node.is_selectable:
                continue
            if not filters.include_disabled and node.status != NodeStatus.ACTIVE:
                continue
            chain = ancestors.get(node_id, [])
            raw = semantic_raw.get(node_id)
            scores = ComponentScores(
                exact=exact.get(node_id, 0.0),
                lexical=lexical.get(node_id, 0.0),
                fuzzy=calibrate_trigram(fuzzy.get(node_id, 0.0), threshold),
                semantic=(calibrate_similarity(raw, floor) if raw is not None else 0.0)
                if space is not None
                else None,
                hierarchy=round(_overlap(query_tokens, " ".join(a.title for a in chain)), 4),
                index_term=calibrate_trigram(index_term.get(node_id, 0.0), threshold),
                semantic_raw=round(raw, 4) if raw is not None else None,
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
        timings["total_ms"] = round((time.perf_counter() - started) * 1000, 2)
        logger.info(
            "retrieval completed",
            extra={
                "dataset_id": dataset.id,
                "variants": len(variants),
                "pool_size": len(pool),
                "returned": min(top_k, len(candidates)),
                "semantic_status": semantic_status.value,
                "duration_ms": duration,
            },
        )
        return RetrievalResult(
            candidates=candidates[:top_k],
            semantic_status=semantic_status,
            pool_size=len(pool),
            duration_ms=duration,
            embedding_space={
                "provider": space.provider,
                "model": space.model,
                "dimension": space.dimension,
                "normalized": space.normalized,
                "distance": space.distance,
                "similarity_floor": floor,
            }
            if space is not None
            else None,
            timings_ms=timings,
        )
