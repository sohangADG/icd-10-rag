"""Run a clinical scenario suite against READY datasets and build the report."""

from datetime import UTC, datetime
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.evaluation.clinical.evaluator import ClinicalEvaluator
from app.evaluation.clinical.report import build_report
from app.evaluation.clinical.scenarios import ClinicalScenario
from app.indexing.embeddings import EmbeddingProvider
from app.retrieval.hybrid import SemanticStatus


async def run_clinical_evaluation(
    session: AsyncSession,
    settings: Settings,
    provider: EmbeddingProvider | None,
    scenarios: list[ClinicalScenario],
    *,
    top_k: int = 5,
    provider_status: SemanticStatus | None = None,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    evaluator = ClinicalEvaluator(
        session, settings, provider, provider_status=provider_status, top_k=top_k
    )
    results = [await evaluator.evaluate(scenario) for scenario in scenarios]
    return build_report(
        results,
        {
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "scenarios": len(scenarios),
            "top_k": top_k,
            "embedding_provider": provider.provider_name if provider else None,
            "embedding_model": provider.model if provider else None,
            "embedding_dimension": provider.dimension if provider else None,
            "semantic_retrieval_mode": settings.semantic_retrieval_mode,
            "evidence_thresholds": {
                "suggestion_min_evidence": settings.suggestion_min_evidence,
                "suggestion_min_lexical_evidence": settings.suggestion_min_lexical_evidence,
            },
            **(meta or {}),
        },
    )
