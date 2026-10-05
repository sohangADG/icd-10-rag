"""Compare retrieval modes and embedding providers on the same cases.

Modes (all restricted to the selected dataset and to classification records):
* lexical_only: hybrid retrieval with the semantic component disabled
  (exact + full-text + trigram + source terms + hierarchy);
* vector_only:  pgvector similarity alone, in the provider's embedding space;
* hybrid:       every component, including the provider's semantic score.

For each provider the dataset is (re-)embedded in that provider's space first; switching
providers therefore rebuilds embeddings (the last provider compared stays active). Results
measure SYSTEM BEHAVIOUR on synthetic data only, never real coding accuracy.
"""

import statistics
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.constants import CLASSIFICATION_NODE_TYPES
from app.evaluation import metrics
from app.evaluation.models import EvalCase
from app.indexing.embeddings import EmbeddingProvider
from app.indexing.indexer import SearchIndexer
from app.indexing.vector_space import VectorSpace
from app.models import IcdDataset
from app.repositories.icd_repository import IcdRepository
from app.repositories.search_repository import SearchRepository
from app.retrieval.hybrid import HybridRetriever, RetrievalFilters

POOL = 15


@dataclass
class ModeResult:
    mode: str
    provider: str | None
    model: str | None
    cases: int
    recall_at_k: float
    mrr: float
    top1: float
    latency_ms_mean: float
    latency_ms_p95: float
    k: int
    details: list[dict[str, Any]] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("details")
        return data


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], 2)


def _result(
    mode: str,
    provider: EmbeddingProvider | None,
    rows: list[tuple[EvalCase, list[str | None], float]],
    k: int,
) -> ModeResult:
    ranks = [metrics.rank_of(case.expected_code or "", codes) for case, codes, _ in rows]
    latencies = [latency for *_, latency in rows]
    return ModeResult(
        mode=mode,
        provider=provider.provider_name if provider else None,
        model=provider.model if provider else None,
        cases=len(rows),
        recall_at_k=round(metrics.recall_at_k(ranks, k), 4),
        mrr=round(metrics.mean_reciprocal_rank(ranks), 4),
        top1=round(metrics.recall_at_k(ranks, 1), 4),
        latency_ms_mean=round(statistics.fmean(latencies), 2) if latencies else 0.0,
        latency_ms_p95=_p95(latencies),
        k=k,
        details=[
            {
                "case_id": case.case_id,
                "expected": case.expected_code,
                "rank": rank,
                "top3": codes[:3],
            }
            for (case, codes, _), rank in zip(rows, ranks, strict=True)
        ],
    )


async def _hybrid_mode(
    session: AsyncSession,
    settings: Settings,
    dataset: IcdDataset,
    cases: list[EvalCase],
    provider: EmbeddingProvider | None,
) -> list[tuple[EvalCase, list[str | None], float]]:
    retriever = HybridRetriever(session, settings, provider)
    rows = []
    for case in cases:
        started = time.perf_counter()
        result = await retriever.retrieve(
            dataset,
            [case.clinical_note],
            top_k=POOL,
            filters=RetrievalFilters(node_types=CLASSIFICATION_NODE_TYPES),
        )
        latency = (time.perf_counter() - started) * 1000
        rows.append((case, [c.node.code for c in result.candidates], latency))
    return rows


async def _vector_mode(
    session: AsyncSession,
    settings: Settings,
    dataset: IcdDataset,
    cases: list[EvalCase],
    provider: EmbeddingProvider,
) -> list[tuple[EvalCase, list[str | None], float]]:
    search, records = SearchRepository(session), IcdRepository(session)
    space = VectorSpace.of(provider, settings.embedding_distance)
    rows = []
    for case in cases:
        started = time.perf_counter()
        vector = await provider.embed_query(case.clinical_note)
        similarities = await search.semantic(dataset.id, space, vector, limit=POOL * 3)
        latency = (time.perf_counter() - started) * 1000
        nodes = await records.get_many(dataset.id, list(similarities))
        ranked = sorted(similarities, key=lambda node_id: -similarities[node_id])
        codes = [
            nodes[n].code
            for n in ranked
            if n in nodes and nodes[n].node_type in CLASSIFICATION_NODE_TYPES
        ]
        rows.append((case, codes[:POOL], latency))
    return rows


async def compare_providers(
    session: AsyncSession,
    settings: Settings,
    dataset: IcdDataset,
    cases: list[EvalCase],
    providers: list[EmbeddingProvider],
    *,
    k: int = 3,
) -> dict[str, Any]:
    results: list[ModeResult] = [
        _result(
            "lexical_only", None, await _hybrid_mode(session, settings, dataset, cases, None), k
        )
    ]
    indexing: list[dict[str, Any]] = []
    for provider in providers:
        stats = await SearchIndexer(session, provider).embed_documents(
            dataset, batch_size=settings.embedding_batch_size, distance=settings.embedding_distance
        )
        indexing.append(
            {
                key: getattr(stats, key)
                for key in (
                    "provider",
                    "model",
                    "dimension",
                    "documents",
                    "embedded",
                    "skipped_unchanged",
                    "skipped_empty",
                    "failed",
                    "embed_ms",
                    "write_ms",
                    "duration_ms",
                    "vector_index",
                )
            }
        )
        results.append(
            _result(
                "vector_only",
                provider,
                await _vector_mode(session, settings, dataset, cases, provider),
                k,
            )
        )
        results.append(
            _result(
                "hybrid",
                provider,
                await _hybrid_mode(session, settings, dataset, cases, provider),
                k,
            )
        )
    return {
        "dataset": {"id": dataset.id, "coding_system": dataset.system, "version": dataset.version},
        "cases": len(cases),
        "k": k,
        "indexing": indexing,
        "results": [r.summary() for r in results],
        "details": {f"{r.mode}:{r.provider or '-'}": r.details for r in results},
        "disclaimer": "Synthetic cases measure system behaviour only; they are not evidence of "
        "real-world ICD coding accuracy, and timings on a tiny dataset are not production "
        "performance figures.",
    }
