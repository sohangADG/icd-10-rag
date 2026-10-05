"""Stage-separated evaluation against a READY dataset.

Stages measured independently:
1. Concept extraction — expected concept found (word overlap) and its assertion status.
2. Retrieval          — rank of the expected code among hybrid-retrieval candidates.
3. Reranking          — rank of the expected code after rule/specificity filtering + rerank.
4. Final selection    — top-1 / top-3 of the returned suggestion; negative cases must return
                        nothing.
Plus: unsupported-code rate (every returned code re-checked in the database), hierarchy accuracy
(prediction on the expected branch), rule-violation rate, and human-coder agreement.
"""

from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.clinical.extractor import RuleBasedConceptExtractor
from app.clinical.models import ClinicalConcept
from app.coding.specificity import stems
from app.coding.suggestion_service import SuggestionService
from app.core.config import Settings
from app.evaluation import metrics
from app.evaluation.models import EvalCase
from app.indexing.embeddings import EmbeddingProvider
from app.repositories.icd_repository import IcdRepository
from app.services.dataset_resolver import DatasetResolver

CONCEPT_MATCH = 0.6


def _overlap(a: str, b: str) -> float:
    left, right = stems(a), stems(b)
    return len(left & right) / len(left | right) if left | right else 0.0


class EvaluationRunner:
    def __init__(
        self,
        session: AsyncSession,
        settings: Settings,
        provider: EmbeddingProvider | None = None,
        *,
        k: int = 3,
    ) -> None:
        self._session = session
        self._service = SuggestionService(session, settings, provider)
        self._extractor = RuleBasedConceptExtractor()
        self._records = IcdRepository(session)
        self._k = k

    async def run(self, cases: list[EvalCase]) -> dict[str, Any]:
        results = [await self._case(case) for case in cases]
        positives = [r for r in results if r["expected_code"]]
        negatives = [r for r in results if not r["expected_code"]]
        with_concept = [r for r in results if r["concept_found"] is not None]
        with_status = [r for r in results if r["status_correct"] is not None]
        returned = sum(r["returned_codes"] for r in results)
        unsupported = sum(r["unsupported_codes"] for r in results)
        k = self._k
        return {
            "cases": len(results),
            "positive_cases": len(positives),
            "negative_cases": len(negatives),
            "concept_extraction": {
                "concept_recall": metrics.rate(
                    sum(1 for r in with_concept if r["concept_found"]), len(with_concept)
                ),
                "status_accuracy": metrics.rate(
                    sum(1 for r in with_status if r["status_correct"]), len(with_status)
                ),
                "evaluated_cases": len(with_concept),
            },
            "retrieval": {
                f"recall@{k}": metrics.recall_at_k([r["retrieval_rank"] for r in positives], k),
                "recall@10": metrics.recall_at_k([r["retrieval_rank"] for r in positives], 10),
                "mrr": metrics.mean_reciprocal_rank([r["retrieval_rank"] for r in positives]),
            },
            "reranking": {
                "top1_accuracy": metrics.recall_at_k([r["rerank_rank"] for r in positives], 1),
                f"recall@{k}": metrics.recall_at_k([r["rerank_rank"] for r in positives], k),
                "mrr": metrics.mean_reciprocal_rank([r["rerank_rank"] for r in positives]),
            },
            "final_selection": {
                "top1_accuracy": metrics.recall_at_k([r["final_rank"] for r in positives], 1),
                "top3_recall": metrics.recall_at_k([r["final_rank"] for r in positives], 3),
                "mrr": metrics.mean_reciprocal_rank([r["final_rank"] for r in positives]),
                "negative_case_accuracy": metrics.rate(
                    sum(1 for r in negatives if r["primary_suggestions"] == 0), len(negatives)
                ),
                "hierarchy_accuracy": metrics.rate(
                    sum(1 for r in positives if r["hierarchy_related"]), len(positives)
                ),
            },
            "safety": {
                "returned_codes": returned,
                "unsupported_code_rate": metrics.rate(unsupported, returned),
                "rule_violation_rate": metrics.rate(
                    sum(r["rule_violations"] for r in results),
                    sum(r["primary_suggestions"] for r in results),
                ),
            },
            "human_coder_agreement": metrics.agreement(
                [(r["top1"], r["coder_code"]) for r in results]
            ),
            "details": results,
            "disclaimer": "Synthetic or local test cases measure system behaviour only; they "
            "are not evidence of real-world ICD coding accuracy.",
        }

    async def _case(self, case: EvalCase) -> dict[str, Any]:
        dataset = await DatasetResolver(self._session).resolve(
            coding_system=case.expected_dataset, version=case.expected_version
        )
        concepts = self._extractor.extract(case.clinical_note)
        concept_found: bool | None = None
        status_correct: bool | None = None
        target: ClinicalConcept | None = None
        if case.expected_concept:
            scored = sorted(
                ((_overlap(c.text, case.expected_concept), c) for c in concepts),
                key=lambda pair: -pair[0],
            )
            concept_found = bool(scored and scored[0][0] >= CONCEPT_MATCH)
            target = scored[0][1] if concept_found else None
        if case.expected_status:
            pool = [target] if target else concepts
            status_correct = any(c.status.value == case.expected_status for c in pool)

        best: dict[str, Any] = {"retrieval_rank": None, "rerank_rank": None, "final_rank": None}
        top1 = None
        returned = unsupported = violations = primaries = 0
        all_codes: list[str] = []
        for concept in concepts:
            others = [c for c in concepts if c is not concept]
            suggestion, _, trace = await self._service.trace_concept(
                dataset, concept, others, top_k=3
            )
            if case.expected_code:
                for key, ranked in (
                    ("retrieval_rank", trace["retrieval_codes"]),
                    ("rerank_rank", trace["reranked_codes"]),
                ):
                    rank = metrics.rank_of(case.expected_code, ranked)
                    if rank is not None and (best[key] is None or rank < best[key]):
                        best[key] = rank
            if suggestion is None:
                continue
            primaries += 1
            violations += 1 if suggestion.validation.get("rule_status") == "rejected" else 0
            codes = [suggestion.code] + [a.code for a in suggestion.alternatives if a.code]
            all_codes += codes
            if top1 is None or suggestion.code == case.expected_code:
                top1 = suggestion.code
            if case.expected_code:
                rank = metrics.rank_of(case.expected_code, codes)
                if rank is not None and (best["final_rank"] is None or rank < best["final_rank"]):
                    best["final_rank"] = rank
        returned = len(all_codes)
        if all_codes:
            found = await self._records.get_by_codes(dataset.id, all_codes)
            unsupported = sum(1 for code in all_codes if code not in found)

        related = False
        if case.expected_code and top1:
            nodes = await self._records.get_by_codes(dataset.id, [top1, case.expected_code])
            ancestors = await self._records.ancestors_of_many(
                dataset.id, [n.id for n in nodes.values()]
            )
            ancestry = {
                code: {a.code for a in ancestors.get(node.id, []) if a.code}
                for code, node in nodes.items()
            }
            related = metrics.hierarchy_related(top1, case.expected_code, ancestry)
        return {
            "case_id": case.case_id,
            "expected_code": case.expected_code,
            "top1": top1,
            "coder_code": case.coder_code,
            "concepts": [{"text": c.text, "status": c.status.value} for c in concepts],
            "concept_found": concept_found,
            "status_correct": status_correct,
            **best,
            "hierarchy_related": related,
            "returned_codes": returned,
            "unsupported_codes": unsupported,
            "rule_violations": violations,
            "primary_suggestions": primaries,
        }
