"""Per-scenario evaluation: runs the real suggestion pipeline (with a trace), each retrieval
method on its own, and independent database / specificity / evidence checks.

Nothing here changes what the pipeline returns; it only observes and re-verifies it.
"""

import time
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.clinical.extractor import RuleBasedConceptExtractor
from app.clinical.models import AssertionStatus, ClinicalConcept
from app.coding.suggestion_service import SuggestionService
from app.core.config import Settings
from app.core.constants import CLASSIFICATION_NODE_TYPES, DatasetStatus, NodeStatus, TermType
from app.core.exceptions import AppError
from app.core.text import normalize_text, ts_config_for
from app.evaluation.clinical import oracles
from app.evaluation.clinical.scenarios import CONFIDENCE_ORDER, ClinicalScenario
from app.indexing.embeddings import EmbeddingProvider
from app.indexing.vector_space import VectorSpace
from app.ingestion.codes import looks_like_code
from app.models import IcdDataset, IcdNode
from app.repositories.icd_repository import IcdRepository
from app.repositories.search_repository import SearchRepository
from app.retrieval.hybrid import SemanticStatus
from app.schemas.icd import SuggestRequest, SuggestResponse
from app.services.dataset_resolver import DatasetResolver
from app.services.presenters import provenance

MODES = ("exact", "lexical", "fuzzy", "index_term", "vector", "hybrid")
MODE_LIMIT = 30
STAGE_ORDER = (
    "DATASET_ISOLATION",
    "DB_VALIDATION",
    "CONCEPT_EXTRACTION",
    "ASSERTION",
    "QUERY_GENERATION",
    "EXACT_RETRIEVAL",
    "LEXICAL_RETRIEVAL",
    "FUZZY_RETRIEVAL",
    "VECTOR_RETRIEVAL",
    "HYBRID_SCORING",
    "RERANKING",
    "RULE_VALIDATION",
    "SPECIFICITY_GUARD",
    "ABSTENTION",
    "OTHER",
)
_ATTRIBUTE_FIELDS = {
    "laterality": "laterality",
    "severity": "severity",
    "acuity": "acuity",
    "subtype": "subtype",
    "stage": "stage",
    "encounter": "encounter",
    "anatomy": "anatomy",
    "complication": "complications",
    "cause": "causes",
    "absence": "explicit_absence",
}
_HISTORY_TITLE_WORDS = ("history", "former", "previous")
NEAR_TIE = 0.05


@dataclass
class _Dataset:
    node: IcdDataset
    space: VectorSpace | None


def _attribute_matches(name: str, expected: Any, concept: ClinicalConcept) -> bool:
    value = getattr(concept.attributes, _ATTRIBUTE_FIELDS[name])
    if isinstance(value, list):
        if expected is None:
            return not value
        if expected is True:
            return bool(value)
        return any(str(expected).lower() in str(v).lower() for v in value)
    if expected is None:
        return value is None
    return str(value).lower() == str(expected).lower()


def _expected_status_of(scenario: ClinicalScenario, concept_text: str) -> AssertionStatus | None:
    best, best_score = None, 0.0
    for text, status in zip(
        scenario.expected_concepts, scenario.expected_assertion_states, strict=True
    ):
        score = oracles.concept_similarity(text, concept_text)
        if score > best_score:
            best, best_score = status, score
    return best if best_score >= oracles.CONCEPT_MATCH else None


class ClinicalEvaluator:
    def __init__(
        self,
        session: AsyncSession,
        settings: Settings,
        provider: EmbeddingProvider | None,
        *,
        provider_status: SemanticStatus | None = None,
        top_k: int = 5,
    ) -> None:
        self._session = session
        self._settings = settings
        self._provider = provider
        self._service = SuggestionService(
            session, settings, provider, provider_status=provider_status
        )
        self._search = SearchRepository(session)
        self._records = IcdRepository(session)
        self._extractor = RuleBasedConceptExtractor()
        self._top_k = top_k
        self._datasets: dict[tuple[str, str], _Dataset] = {}

    # --- datasets -----------------------------------------------------------------------------

    async def _dataset(self, scenario: ClinicalScenario) -> _Dataset:
        key = (scenario.coding_system.upper(), scenario.version)
        if key not in self._datasets:
            node = await DatasetResolver(self._session).resolve(
                coding_system=scenario.coding_system, version=scenario.version
            )
            space = None
            if self._provider is not None:
                candidate = VectorSpace.of(self._provider, self._settings.embedding_distance)
                embedded, total = await self._search.space_coverage(node.id, candidate)
                space = candidate if embedded and embedded == total else None
            self._datasets[key] = _Dataset(node, space)
        return self._datasets[key]

    # --- one scenario -----------------------------------------------------------------------

    async def evaluate(self, scenario: ClinicalScenario) -> dict[str, Any]:
        result: dict[str, Any] = {
            "id": scenario.id,
            "tags": scenario.tags,
            "coding_system": scenario.coding_system,
            "version": scenario.version,
            "should_abstain": scenario.should_abstain,
            "expected_codes": scenario.expected_codes,
            "failures": [],
        }
        try:
            ds = await self._dataset(scenario)
        except AppError as exc:
            result["error"] = exc.code
            result["failures"].append(("OTHER", f"dataset resolution failed: {exc.code}"))
            result["passed"] = False
            result["failure_stage"] = "OTHER"
            return result

        started = time.perf_counter()
        concepts = self._extractor.extract(scenario.clinical_note)
        extraction_ms = (time.perf_counter() - started) * 1000
        trace: dict[str, Any] = {}
        try:
            response = await self._service.suggest(
                SuggestRequest(
                    clinical_note=scenario.clinical_note,
                    coding_system=scenario.coding_system,
                    version=scenario.version,
                    top_k=self._top_k,
                    include_uncertain=scenario.include_uncertain,
                ),
                trace=trace,
            )
        except AppError as exc:
            result["error"] = exc.code
            result["failures"].append(("OTHER", f"suggestion failed: {exc.code}"))
            result["passed"] = False
            result["failure_stage"] = "OTHER"
            return result
        result["semantic_status"] = next(
            (c.get("semantic_status") for c in trace.get("concepts", []) if "semantic_status" in c),
            None,
        )
        self._concepts(scenario, concepts, result)
        await self._retrieval(scenario, ds, response, trace, result)
        await self._database(scenario, ds, response, result)
        await self._final(scenario, ds, response, trace, result)
        self._latency(trace, extraction_ms, result)
        self._classify(scenario, result)
        return result

    # --- concept extraction ------------------------------------------------------------------

    def _concepts(
        self, scenario: ClinicalScenario, concepts: list[ClinicalConcept], result: dict[str, Any]
    ) -> None:
        texts = [c.text for c in concepts]
        matched = oracles.match_concepts(
            scenario.expected_concepts, [[c.text, c.evidence] for c in concepts]
        )
        statuses, attributes = [], []
        for i, j in matched.items():
            expected_status = scenario.expected_assertion_states[i]
            statuses.append((expected_status.value, concepts[j].status.value))
            expected_attrs = (
                scenario.expected_attributes[i] if scenario.expected_attributes else None
            ) or {}
            for name, value in expected_attrs.items():
                attributes.append(
                    {
                        "concept": scenario.expected_concepts[i],
                        "attribute": name,
                        "expected": value,
                        "actual": getattr(concepts[j].attributes, _ATTRIBUTE_FIELDS[name]),
                        "ok": _attribute_matches(name, value, concepts[j]),
                    }
                )
        result["concepts"] = {
            "extracted": [
                {
                    "text": c.text,
                    "status": c.status.value,
                    "type": c.concept_type.value,
                    "section": c.section,
                }
                for c in concepts
            ],
            "expected": scenario.expected_concepts,
            "true_positives": len(matched),
            "false_negatives": [
                scenario.expected_concepts[i]
                for i in range(len(scenario.expected_concepts))
                if i not in matched
            ],
            "false_positives": [
                texts[j] for j in range(len(texts)) if j not in set(matched.values())
            ],
            "statuses": statuses,
            "attributes": attributes,
            # extracted concept start offset -> index of the expected concept it matches
            "expected_by_start": {concepts[j].start: i for i, j in matched.items()},
            "matched_similarity": {
                scenario.expected_concepts[i]: round(
                    max(
                        oracles.concept_similarity(scenario.expected_concepts[i], reading)
                        for reading in (texts[j], concepts[j].evidence)
                    ),
                    3,
                )
                for i, j in matched.items()
            },
        }

    # --- retrieval (each method on its own) ---------------------------------------------------

    async def _mode_codes(self, ds: _Dataset, queries: list[str]) -> dict[str, list[str]]:
        dataset = ds.node
        threshold = self._settings.retrieval_fuzzy_threshold
        cfg = ts_config_for(dataset.language)
        scores: dict[str, dict[int, float]] = {m: {} for m in MODES if m != "hybrid"}

        def keep(mode: str, node_id: int, score: float) -> None:
            scores[mode][node_id] = max(scores[mode].get(node_id, float("-inf")), score)

        for query in queries:
            if looks_like_code(query):
                for match in await self._search.exact_code(dataset.id, query):
                    keep("exact", match.node_id, match.score)
            for node_id, score in (
                await self._search.lexical(dataset.id, cfg, query, limit=MODE_LIMIT)
            ).items():
                keep("lexical", node_id, score)
            for match in await self._search.fuzzy_titles(
                dataset.id, query, limit=MODE_LIMIT, threshold=threshold
            ):
                keep("fuzzy", match.node_id, match.score)
            for match in await self._search.source_terms(
                dataset.id, normalize_text(query), limit=MODE_LIMIT, threshold=threshold
            ):
                keep("index_term", match.node_id, match.score)
        if ds.space is not None and self._provider is not None and queries:
            try:
                vectors = await self._provider.embed_queries(queries)
            except Exception:  # noqa: BLE001 - reported as "no vector results"
                vectors = []
            for vector in vectors:
                for node_id, similarity in (
                    await self._search.semantic(dataset.id, ds.space, vector, limit=MODE_LIMIT)
                ).items():
                    keep("vector", node_id, similarity)
        ids = sorted({i for per_mode in scores.values() for i in per_mode})
        nodes = await self._records.get_many(dataset.id, ids)
        ranked: dict[str, list[str]] = {}
        for mode, per_mode in scores.items():
            ordered = sorted(
                (
                    (-score, nodes[i].code or "")
                    for i, score in per_mode.items()
                    if i in nodes
                    and nodes[i].node_type in CLASSIFICATION_NODE_TYPES
                    and nodes[i].status == NodeStatus.ACTIVE
                ),
            )
            ranked[mode] = [code for _, code in ordered]
        return ranked

    async def _retrieval(
        self,
        scenario: ClinicalScenario,
        ds: _Dataset,
        response: SuggestResponse,
        trace: dict[str, Any],
        result: dict[str, Any],
    ) -> None:
        concept_traces = [c for c in trace.get("concepts", []) if "queries" in c]
        per_concept = []
        for concept_trace in concept_traces:
            modes = await self._mode_codes(ds, concept_trace["queries"])
            modes["hybrid"] = concept_trace.get("retrieval_codes", [])
            concept_text = response.clinical_concepts[concept_trace["concept_index"]].text
            per_concept.append((concept_text, concept_trace, modes))

        targets = scenario.expected_retrieval_codes or list(scenario.expected_codes)
        result["retrieval_targets"] = targets
        ranks: dict[str, dict[str, int | None]] = {}
        rerank: dict[str, dict[str, Any]] = {}
        for code in targets:
            wanted_concept = scenario.code_concepts.get(code)
            ranks[code] = {}
            for mode in MODES:
                best = None
                for concept_text, _, modes in per_concept:
                    if wanted_concept and (
                        oracles.concept_similarity(wanted_concept, concept_text)
                        < oracles.CONCEPT_MATCH
                    ):
                        continue
                    if code in modes[mode]:
                        rank = modes[mode].index(code) + 1
                        best = rank if best is None else min(best, rank)
                ranks[code][mode] = best
            before = after = None
            rejected_reasons: list[str] = []
            for _, concept_trace, _ in per_concept:
                retrieved = concept_trace.get("retrieval_codes", [])
                reranked = concept_trace.get("reranked_codes", [])
                if code in retrieved:
                    rank = retrieved.index(code) + 1
                    before = rank if before is None else min(before, rank)
                if code in reranked:
                    rank = reranked.index(code) + 1
                    after = rank if after is None else min(after, rank)
                for candidate in concept_trace.get("candidates", []):
                    if candidate["code"] == code and candidate["reject_reasons"]:
                        rejected_reasons += candidate["reject_reasons"]
            rerank[code] = {
                "retrieval_rank": before,
                "rerank_rank": after,
                "rejected_reasons": rejected_reasons,
            }

        concept_rerank = []
        for concept_text, concept_trace, _ in per_concept:
            retrieved = concept_trace.get("retrieval_codes", [])
            reranked = concept_trace.get("reranked_codes", [])
            concept_rerank.append(
                {
                    "concept": concept_text,
                    "retrieval_top1": retrieved[0] if retrieved else None,
                    "rerank_top1": reranked[0] if reranked else None,
                    "rejected": concept_trace.get("rejected_codes", []),
                    # Individual reranking feature scores of the best candidates (logged for
                    # failure analysis).
                    "top_candidates": sorted(
                        concept_trace.get("candidates", []), key=lambda c: -c["rerank"]
                    )[:5],
                }
            )
        result["retrieval"] = {"ranks": ranks}
        result["reranking"] = {"per_code": rerank, "per_concept": concept_rerank}

    # --- database safety --------------------------------------------------------------------

    async def _database(
        self,
        scenario: ClinicalScenario,
        ds: _Dataset,
        response: SuggestResponse,
        result: dict[str, Any],
    ) -> None:
        """Every returned record id (primary, alternatives, rule targets) -> database row ->
        same dataset, coding system and version; dataset READY; record ACTIVE."""
        returned: list[tuple[str, int, str | None]] = []
        rule_targets: list[tuple[int | None, str | None, bool | None]] = []
        for suggestion in response.suggestions:
            returned.append(("primary", suggestion.record_id, suggestion.code))
            returned += [("alternative", a.record_id, a.code) for a in suggestion.alternatives]
            for check in suggestion.validation.get("rule_checks", []):
                if check.get("target_code"):
                    rule_targets.append(
                        (
                            check.get("target_record_id"),
                            check.get("target_code"),
                            check.get("target_exists"),
                        )
                    )
        ids = {record_id for _, record_id, _ in returned} | {
            t[0] for t in rule_targets if t[0] is not None
        }
        rows = (
            await self._session.execute(
                select(IcdNode, IcdDataset)
                .join(IcdDataset, IcdDataset.id == IcdNode.dataset_id)
                .where(IcdNode.id.in_(sorted(ids)))
            )
        ).all()
        found = {node.id: (node, dataset) for node, dataset in rows}
        issues: list[dict[str, Any]] = []
        counts = {
            "returned": len(returned),
            "primary": sum(1 for kind, *_ in returned if kind == "primary"),
            "unsupported": 0,
            "cross_version": 0,
            "cross_system": 0,
            "non_ready": 0,
            "inactive": 0,
            "title_mismatch": 0,
        }
        for kind, record_id, code in returned:
            row = found.get(record_id)
            problem = None
            if row is None or row[0].code != code:
                problem = "unsupported"
            else:
                node, dataset = row
                if dataset.id != ds.node.id:
                    problem = (
                        "cross_system"
                        if dataset.system.upper() != scenario.coding_system.upper()
                        else "cross_version"
                    )
                elif dataset.status != DatasetStatus.READY:
                    problem = "non_ready"
                elif node.status != NodeStatus.ACTIVE:
                    problem = "inactive"
            if problem:
                counts[problem] += 1
                issues.append(
                    {"kind": kind, "record_id": record_id, "code": code, "issue": problem}
                )
        for suggestion in response.suggestions:
            row = found.get(suggestion.record_id)
            if row is not None and row[0].title != suggestion.title:
                counts["title_mismatch"] += 1
                issues.append({"code": suggestion.code, "issue": "title_mismatch"})
            reference = suggestion.icd_reference
            if (
                reference.get("dataset_id") != ds.node.id
                or str(reference.get("version")) != scenario.version
                or str(reference.get("coding_system")).upper() != scenario.coding_system.upper()
            ):
                counts["cross_version"] += 1
                issues.append({"code": suggestion.code, "issue": "icd_reference_dataset"})
        invalid_targets = 0
        for record_id, code, exists in rule_targets:
            row = found.get(record_id) if record_id is not None else None
            truly_exists = (
                await self._records.get_by_code(ds.node.id, code or "", classification_only=False)
                is not None
            )
            if row is not None and row[1].id != ds.node.id:
                invalid_targets += 1
                issues.append({"code": code, "issue": "rule_target_other_dataset"})
            elif exists is True and not truly_exists:
                invalid_targets += 1
                issues.append({"code": code, "issue": "rule_target_reported_but_missing"})
        counts["rule_targets"] = len(rule_targets)
        counts["invalid_rule_targets"] = invalid_targets
        result["database"] = {"counts": counts, "issues": issues}

    # --- final selection, rules, specificity, abstention, confidence, evidence ----------------

    async def _final(
        self,
        scenario: ClinicalScenario,
        ds: _Dataset,
        response: SuggestResponse,
        trace: dict[str, Any],
        result: dict[str, Any],
    ) -> None:
        primaries = [s.code for s in response.suggestions]
        allowed = scenario.allowed_codes
        result["final"] = {
            "primary_codes": primaries,
            "suggestions": [
                {
                    "code": s.code,
                    "concept": s.clinical_concept,
                    "status": s.concept_status,
                    "confidence": s.confidence,
                    "missing_information": s.missing_information,
                    "resolution": s.validation.get("resolution"),
                    "rule_status": s.validation.get("rule_status"),
                }
                for s in response.suggestions
            ],
            "unmatched": [
                {"concept": u.clinical_concept, "status": u.concept_status, "reason": u.reason}
                for u in response.unmatched_concepts
            ],
            "hits": [c for c in scenario.expected_codes if c in primaries],
            "missed": [c for c in scenario.expected_codes if c not in primaries],
            "acceptable": [c for c in primaries if c in scenario.acceptable_alternatives],
            "false_positives": [c for c in primaries if c not in allowed],
            "must_not_return_violations": [c for c in primaries if c in scenario.must_not_return],
            "duplicates": sorted({c for c in primaries if primaries.count(c) > 1}),
            "abstained": not primaries,
        }
        # Concept association: an expected code must be attached to its concept.
        association_errors = []
        for code, concept_text in scenario.code_concepts.items():
            attached = [s for s in response.suggestions if s.code == code]
            if attached and all(
                max(
                    oracles.concept_similarity(concept_text, s.clinical_concept),
                    oracles.concept_similarity(concept_text, str(s.evidence[0].get("text", ""))),
                )
                < oracles.CONCEPT_MATCH
                for s in attached
            ):
                association_errors.append(code)
        result["final"]["association_errors"] = association_errors

        # Missing information: required items reported; reported items genuinely missing.
        reported = [m for s in response.suggestions for m in s.missing_information]
        result["missing_information"] = {
            "required": scenario.required_missing_information,
            "found": [
                r
                for r in scenario.required_missing_information
                if any(r.lower() in m.lower() for m in reported)
            ],
            "reported": reported,
            "not_genuine": [
                m
                for s in response.suggestions
                for m in s.missing_information
                if not oracles.missing_item_is_genuine(m, str(s.evidence[0].get("text", "")))
            ],
        }

        # Rules: surfaced instructions, expected rejections, fatal violations.
        surfaced = sorted(
            {
                check["rule_type"]
                for s in response.suggestions
                for check in s.validation.get("rule_checks", [])
            }
        )
        rejected_codes = {
            candidate["code"]
            for concept_trace in trace.get("concepts", [])
            for candidate in concept_trace.get("candidates", [])
            if candidate["reject_reasons"]
        } | {r.get("code") for u in response.unmatched_concepts for r in u.rejected_candidates}
        fatal = [
            {"code": s.code, "issue": "rule_status_rejected"}
            for s in response.suggestions
            if s.validation.get("rule_status") == "rejected"
        ] + [
            {"code": code, "issue": "expected_rejection_returned"}
            for code in scenario.expected_rejected
            if code in primaries
        ]
        result["rules"] = {
            "expected_instructions": scenario.expected_instructions,
            "surfaced": surfaced,
            "instructions_found": [
                r
                for r in scenario.expected_instructions
                if r in surfaced
                or (r == "EXCLUDES" and bool(rejected_codes & set(scenario.expected_rejected)))
            ],
            "expected_rejected": scenario.expected_rejected,
            "rejected_ok": [
                c for c in scenario.expected_rejected if c in rejected_codes or c not in primaries
            ],
            "rejected_observed": [c for c in scenario.expected_rejected if c in rejected_codes],
            "fatal_violations": fatal,
        }

        # Specificity (independent oracle) + status safety, per primary suggestion.
        specificity, status_violations = [], []
        for suggestion in response.suggestions:
            node = await self._records.get_by_code(ds.node.id, suggestion.code)
            if node is None:
                continue
            ancestors = await self._records.ancestors(ds.node.id, node.id)
            # Baseline = the CATEGORY: everything a subdivision adds beyond the category (at any
            # level, e.g. a subtype added by an intermediate subdivision) must be documented.
            parent = next((a for a in ancestors if a.node_type.value == "CATEGORY"), None)
            details = (await self._records.details_of_many(ds.node.id, [node.id]))[node.id]
            terms = [
                t.term
                for t in details["terms"]
                if t.term_type in (TermType.INCLUSION, TermType.SYNONYM, TermType.ABBREVIATION)
            ]
            clause = str(suggestion.evidence[0].get("text", ""))
            for finding in oracles.unsupported_specificity(
                node.title, parent.title if parent else None, clause, terms
            ):
                specificity.append(
                    {
                        "code": suggestion.code,
                        "attribute": finding.attribute,
                        "detail": finding.detail,
                    }
                )
            expected_status = self._expected_status(scenario, suggestion, result)
            title = node.title.lower() + " " + " ".join(t.lower() for t in terms)
            history_code = any(w in title for w in _HISTORY_TITLE_WORDS)
            family_code = "family history" in title
            if suggestion.concept_status in ("negated", "ruled_out") or expected_status in (
                AssertionStatus.NEGATED,
                AssertionStatus.RULED_OUT,
            ):
                status_violations.append({"code": suggestion.code, "issue": "negated_as_active"})
            if expected_status == AssertionStatus.FAMILY_HISTORY and not family_code:
                status_violations.append(
                    {"code": suggestion.code, "issue": "family_history_as_active"}
                )
            if expected_status == AssertionStatus.HISTORY and not history_code:
                status_violations.append({"code": suggestion.code, "issue": "history_as_active"})
        result["specificity"] = {"unsupported": specificity}
        result["status_safety"] = status_violations

        # Confidence audit: evidence signals behind each level.
        confidence = []
        for index, suggestion in enumerate(response.suggestions):
            scores = suggestion.retrieval_scores
            concept_trace = self._trace_for(trace, response, suggestion.clinical_concept)
            accepted = sorted(
                (c for c in (concept_trace or {}).get("candidates", []) if not c["reject_reasons"]),
                key=lambda c: -c["rerank"],
            )
            near_tie = (
                len(accepted) > 1 and accepted[0]["rerank"] - accepted[1]["rerank"] < NEAR_TIE
            )
            lexical_strong = (
                (scores.get("exact") or 0) >= 1
                or (scores.get("lexical") or 0) >= 0.6
                or (scores.get("index_term") or 0) >= 0.6
                or (scores.get("fuzzy") or 0) >= 0.6
            )
            vector_strong = (scores.get("semantic") or 0) >= 0.5
            features = scores.get("rerank_features", {})
            signals = {
                "lexical_strong": lexical_strong,
                "vector_strong": vector_strong,
                "agree": lexical_strong and vector_strong,
                "near_tie": near_tie,
                "specificity_incomplete": bool(suggestion.missing_information),
                "rule_conflict": suggestion.validation.get("rule_status") != "passed",
                "terminology_support": bool(
                    features.get("exact_terminology") or features.get("inclusion_match")
                ),
            }
            weak = not lexical_strong and not signals["terminology_support"]
            violation = suggestion.confidence == "HIGH" and (
                weak
                or signals["near_tie"]
                or signals["specificity_incomplete"]
                or signals["rule_conflict"]
            )
            if scenario.max_confidence and (
                CONFIDENCE_ORDER[suggestion.confidence] > CONFIDENCE_ORDER[scenario.max_confidence]
            ):
                violation = True
            confidence.append(
                {
                    "index": index,
                    "code": suggestion.code,
                    "level": suggestion.confidence,
                    "correct": suggestion.code in allowed,
                    "signals": signals,
                    "violation": violation,
                }
            )
        result["confidence"] = confidence

        result["evidence"] = await self._evidence(scenario, ds, response)

    @staticmethod
    def _expected_status(
        scenario: ClinicalScenario, suggestion: Any, result: dict[str, Any]
    ) -> AssertionStatus | None:
        """Expected status of the concept a suggestion codes: located by the evidence span
        (two concepts may share a text, e.g. a family history and the patient's own)."""
        start = suggestion.evidence[0].get("start") if suggestion.evidence else None
        index = result["concepts"]["expected_by_start"].get(start)
        if index is not None:
            return scenario.expected_assertion_states[index]
        return _expected_status_of(scenario, suggestion.clinical_concept)

    @staticmethod
    def _trace_for(
        trace: dict[str, Any], response: SuggestResponse, concept_text: str
    ) -> dict[str, Any] | None:
        for concept_trace in trace.get("concepts", []):
            index = concept_trace["concept_index"]
            if response.clinical_concepts[index].text == concept_text and "candidates" in (
                concept_trace
            ):
                return concept_trace
        return None

    async def _evidence(
        self, scenario: ClinicalScenario, ds: _Dataset, response: SuggestResponse
    ) -> dict[str, Any]:
        """Every evidence item must trace to the note (clinical text) or to the database record
        (matched terms, inclusion terms, provenance). Exclusions are never support."""
        note = scenario.clinical_note
        checked = traceable = incorrect = fabricated = 0
        provenance_ok = provenance_total = 0
        problems: list[dict[str, Any]] = []
        for suggestion in response.suggestions:
            retrieved_id = suggestion.retrieval_scores.get("retrieved_record_id")
            ids = [suggestion.record_id] + ([retrieved_id] if retrieved_id else [])
            nodes = await self._records.get_many(ds.node.id, ids)
            details = await self._records.details_of_many(ds.node.id, list(nodes))
            truth: set[str] = set()
            inclusions: set[str] = set()
            exclusions: set[str] = set()
            for node_id, node in nodes.items():
                truth |= {normalize_text(node.title), normalize_text(node.code or "")}
                for term in details[node_id]["terms"]:
                    truth.add(normalize_text(term.term))
                    if term.term_type == TermType.INCLUSION:
                        inclusions.add(normalize_text(term.term))
                for entry in details[node_id]["index_terms"]:
                    truth.add(normalize_text(entry.lead_term))
                for rule in details[node_id]["rules"]:
                    if rule.rule_type.value == "EXCLUDE":
                        exclusions.add(normalize_text(rule.rule_text))
            for item in suggestion.evidence:
                checked += 1
                kind = item.get("type")
                text = str(item.get("text", ""))
                ok = True
                if kind == "clinical_text":
                    start, end = item.get("start"), item.get("end")
                    ok = isinstance(start, int) and isinstance(end, int) and note[start:end] == text
                elif kind == "matched_term":
                    ok = normalize_text(text) in truth
                elif kind == "inclusion_term":
                    ok = normalize_text(text) in inclusions
                elif kind == "documented_specificity":
                    ok = isinstance(item.get("details"), list)
                else:
                    fabricated += 1
                    ok = False
                if (
                    kind in ("matched_term", "inclusion_term")
                    and normalize_text(text) in exclusions
                ):
                    incorrect += 1
                    ok = False
                    problems.append({"code": suggestion.code, "issue": "exclusion_as_support"})
                if ok:
                    traceable += 1
                else:
                    problems.append({"code": suggestion.code, "type": kind, "text": text[:80]})
            final_node = nodes.get(suggestion.record_id)
            if final_node is not None:
                provenance_total += 1
                expected = provenance(final_node, details.get(final_node.id))
                if suggestion.source_provenance == expected:
                    provenance_ok += 1
                else:
                    problems.append({"code": suggestion.code, "issue": "provenance_mismatch"})
                ancestors = await self._records.ancestors(ds.node.id, final_node.id)
                hierarchy = [h.get("code") for h in suggestion.icd_reference.get("hierarchy", [])]
                if hierarchy != [a.code for a in ancestors]:
                    incorrect += 1
                    problems.append({"code": suggestion.code, "issue": "hierarchy_mismatch"})
        return {
            "items": checked,
            "traceable": traceable,
            "incorrect": incorrect,
            "fabricated": fabricated,
            "provenance_ok": provenance_ok,
            "provenance_total": provenance_total,
            "problems": problems,
        }

    # --- latency -----------------------------------------------------------------------------

    @staticmethod
    def _latency(trace: dict[str, Any], extraction_ms: float, result: dict[str, Any]) -> None:
        def total(stage: str) -> float:
            return round(
                sum(c.get("timings_ms", {}).get(stage, 0.0) for c in trace.get("concepts", [])), 3
            )

        result["latency_ms"] = {
            "concept_extraction": round(trace.get("extraction_ms", extraction_ms), 3),
            "retrieval": total("retrieval_ms"),
            "reranking": total("rerank_ms"),
            "rule_validation": round(
                total("rules_ms") + total("rule_data_ms") + total("specificity_ms"), 3
            ),
            "resolution_and_db_validation": total("resolution_ms"),
            "total": round(trace.get("total_ms", 0.0), 3),
        }

    # --- failure taxonomy ------------------------------------------------------------------

    def _classify(self, scenario: ClinicalScenario, result: dict[str, Any]) -> None:
        failures: list[tuple[str, str]] = result["failures"]
        db = result["database"]["counts"]
        if db["cross_version"] or db["cross_system"] or db["non_ready"]:
            failures.append(("DATASET_ISOLATION", str(result["database"]["issues"])))
        if db["unsupported"] or db["inactive"] or db["title_mismatch"]:
            failures.append(("DB_VALIDATION", str(result["database"]["issues"])))
        concepts = result["concepts"]
        if concepts["false_negatives"]:
            failures.append(("CONCEPT_EXTRACTION", f"missed {concepts['false_negatives']}"))
        if concepts["false_positives"]:
            failures.append(("CONCEPT_EXTRACTION", f"spurious {concepts['false_positives']}"))
        bad_attributes = [a for a in concepts["attributes"] if not a["ok"]]
        if bad_attributes:
            failures.append(
                (
                    "CONCEPT_EXTRACTION",
                    "attributes "
                    + "; ".join(
                        f"{a['attribute']}: expected {a['expected']!r} got {a['actual']!r}"
                        for a in bad_attributes
                    ),
                )
            )
        wrong_status = [(e, a) for e, a in concepts["statuses"] if e != a]
        if wrong_status:
            failures.append(("ASSERTION", f"status expected/actual {wrong_status}"))
        for violation in result["status_safety"]:
            failures.append(("ASSERTION", f"{violation['issue']}: {violation['code']}"))

        final = result["final"]
        for code in final["missed"]:
            failures.append(self._missed_stage(scenario, result, code))
        for code in final["false_positives"]:
            failures.append(self._false_positive_stage(scenario, result, code))
        for code in final["duplicates"]:
            failures.append(("OTHER", f"{code} suggested more than once"))
        for code in final["association_errors"]:
            failures.append(("RERANKING", f"{code} attached to the wrong concept"))
        for item in result["specificity"]["unsupported"]:
            failures.append(
                ("SPECIFICITY_GUARD", f"{item['code']} adds undocumented {item['attribute']}")
            )
        missing = result["missing_information"]
        not_found = [r for r in missing["required"] if r not in missing["found"]]
        if not_found:
            failures.append(("SPECIFICITY_GUARD", f"missing information not reported {not_found}"))
        if missing["not_genuine"]:
            failures.append(
                (
                    "SPECIFICITY_GUARD",
                    f"reported as missing but documented {missing['not_genuine']}",
                )
            )
        rules = result["rules"]
        not_surfaced = [
            r for r in rules["expected_instructions"] if r not in rules["instructions_found"]
        ]
        if not_surfaced:
            failures.append(("RULE_VALIDATION", f"instructions not surfaced {not_surfaced}"))
        for violation in rules["fatal_violations"]:
            failures.append(("RULE_VALIDATION", f"{violation['issue']}: {violation['code']}"))
        for item in result["confidence"]:
            if item["violation"]:
                failures.append(
                    ("OTHER", f"confidence {item['level']} not justified for {item['code']}")
                )
        evidence = result["evidence"]
        if evidence["problems"]:
            failures.append(("OTHER", f"evidence problems {evidence['problems'][:3]}"))

        result["passed"] = not failures
        stages = sorted({stage for stage, _ in failures}, key=STAGE_ORDER.index)
        result["failure_stage"] = stages[0] if stages else None
        result["failure_stages"] = stages

    @staticmethod
    def _retrieval_stage(scenario: ClinicalScenario, ranks: dict[str, int | None]) -> str:
        if any(ranks.get(m) is not None and ranks[m] <= 10 for m in MODES if m != "hybrid"):
            return "HYBRID_SCORING"
        tags = set(scenario.tags)
        if "code_query" in tags:
            return "EXACT_RETRIEVAL"
        if tags & {"typo", "fuzzy"}:
            return "FUZZY_RETRIEVAL"
        if "paraphrase" in tags:
            return "VECTOR_RETRIEVAL"
        return "LEXICAL_RETRIEVAL"

    def _missed_stage(
        self, scenario: ClinicalScenario, result: dict[str, Any], code: str
    ) -> tuple[str, str]:
        rerank = result["reranking"]["per_code"].get(code, {})
        ranks = result["retrieval"]["ranks"].get(code, {})
        concept = scenario.code_concepts.get(code)
        if concept and concept in result["concepts"]["false_negatives"]:
            return ("CONCEPT_EXTRACTION", f"{code}: concept '{concept}' not extracted")
        reasons = " ".join(rerank.get("rejected_reasons", []))
        if rerank.get("rerank_rank") is None and reasons:
            if "exclusion" in reasons:
                return ("RULE_VALIDATION", f"{code} rejected: {reasons[:160]}")
            if "Insufficient retrieval evidence" in reasons:
                return ("ABSTENTION", f"{code} below the evidence gate")
            if "code specifies" in reasons:
                return ("SPECIFICITY_GUARD", f"{code} rejected: {reasons[:160]}")
            if "History" in reasons or "history" in reasons:
                return ("ASSERTION", f"{code} rejected by the status gate: {reasons[:160]}")
            if reasons.startswith("Symptom"):
                return (
                    "CONCEPT_EXTRACTION",
                    f"{code} rejected because the concept was typed as a symptom: {reasons[:120]}",
                )
            if "condition" in reasons or "meaning-only" in reasons:
                return ("ABSTENTION", f"{code} rejected by the condition gate: {reasons[:160]}")
            return ("OTHER", f"{code} rejected: {reasons[:160]}")
        if rerank.get("rerank_rank") is not None and rerank["rerank_rank"] > 1:
            return ("RERANKING", f"{code} reranked to #{rerank['rerank_rank']}")
        if rerank.get("retrieval_rank") is None and ranks.get("hybrid") is None:
            similarities = result["concepts"]["matched_similarity"].values()
            if similarities and min(similarities) < 0.8:
                return ("QUERY_GENERATION", f"{code} not retrieved; noisy concept text")
            return (self._retrieval_stage(scenario, ranks), f"{code} not retrieved")
        if not result["final"]["primary_codes"]:
            return ("ABSTENTION", f"{code} retrieved but nothing returned")
        return ("SPECIFICITY_GUARD", f"{code} retrieved; resolution chose another record")

    @staticmethod
    def _false_positive_stage(
        scenario: ClinicalScenario, result: dict[str, Any], code: str
    ) -> tuple[str, str]:
        suggestion = next(s for s in result["final"]["suggestions"] if s["code"] == code)
        if any(v["code"] == code for v in result["status_safety"]):
            return ("ASSERTION", f"{code} returned for a non-active concept")
        if any(s["code"] == code for s in result["specificity"]["unsupported"]):
            return ("SPECIFICITY_GUARD", f"{code} over-specific")
        if code in scenario.expected_rejected:
            return ("RULE_VALIDATION", f"{code} should have been rejected")
        if scenario.should_abstain:
            return ("ABSTENTION", f"{code} returned where abstention was expected")
        expected = _expected_status_of(scenario, suggestion["concept"])
        if expected is None:
            return ("CONCEPT_EXTRACTION", f"{code} attached to unexpected concept")
        return ("RERANKING", f"{code} chosen instead of {scenario.expected_codes or 'nothing'}")
