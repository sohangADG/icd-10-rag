"""Evidence-backed ICD code suggestion.

clinical note
  -> concept extraction (status-aware; negated / ruled-out concepts are never coded)
  -> per concept: hybrid retrieval in the resolved dataset/version only
  -> rule evaluation (excludes / includes / instructions, incl. ancestor rules)
  -> specificity protection (no unsupported laterality/severity/acuity/subtype/...)
  -> reranking with explicit features
  -> hierarchy-aware resolution (fall back to an "unspecified" sibling or the parent;
     descend from a category only to a child the documentation supports)
  -> MANDATORY database re-verification of the final code (exists, active, same dataset)
  -> response with evidence, alternatives, missing information, scores and provenance.

No code is ever produced that is not a row of the selected dataset.
"""

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.clinical.extractor import ConceptExtractor, RuleBasedConceptExtractor
from app.clinical.models import AssertionStatus, ClinicalConcept, ConceptType
from app.coding.condition import condition_support
from app.coding.reranker import Reranker, RerankFeatures
from app.coding.rules import RuleEngine, RuleEvaluation
from app.coding.specificity import SpecificityCheck, check_specificity
from app.core.config import Settings
from app.core.constants import CLASSIFICATION_NODE_TYPES, NodeStatus, NodeType, TermType
from app.core.exceptions import InvalidClinicalNoteError
from app.core.text import normalize_text
from app.indexing.embeddings import EmbeddingProvider
from app.models import IcdDataset, IcdNode
from app.repositories.icd_repository import IcdRepository
from app.retrieval.hybrid import (
    HybridRetriever,
    RetrievalCandidate,
    RetrievalFilters,
    SemanticStatus,
)
from app.schemas.icd import (
    Alternative,
    Suggestion,
    SuggestRequest,
    SuggestResponse,
    UnmatchedConcept,
)
from app.services.dataset_resolver import DatasetResolver
from app.services.presenters import dataset_summary, hierarchy_item, provenance

logger = logging.getLogger(__name__)

CANDIDATE_POOL = 15
HIGH_CONFIDENCE = 0.55
MEDIUM_CONFIDENCE = 0.35
MIN_ALTERNATIVE_SCORE = 0.2
_HISTORY_WORDS = ("history", "former", "previous")
_UNCERTAIN = {AssertionStatus.UNCERTAIN, AssertionStatus.SUSPECTED}
_STATUS_MENTIONS = {AssertionStatus.HISTORY, AssertionStatus.FAMILY_HISTORY}


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 3)


class _StageClock:
    """Accumulates elapsed time per stage into `timings` (no-op when it is None)."""

    def __init__(self, timings: dict[str, float] | None) -> None:
        self._timings = timings

    @contextmanager
    def __call__(self, stage: str) -> Iterator[None]:
        if self._timings is None:
            yield
            return
        started = time.perf_counter()
        try:
            yield
        finally:
            self._timings[stage] = round(
                self._timings.get(stage, 0.0) + (time.perf_counter() - started) * 1000, 3
            )


@dataclass
class Evaluated:
    candidate: RetrievalCandidate
    parent: IcdNode | None
    specificity: SpecificityCheck
    rules: RuleEvaluation
    features: RerankFeatures
    rerank: float
    reject_reasons: list[str] = field(default_factory=list)
    condition: str = "supported"  # app.coding.condition verdict (or "exempt")

    @property
    def node(self) -> IcdNode:
        return self.candidate.node


def _classification_parent(ancestors: list[IcdNode]) -> IcdNode | None:
    if ancestors and ancestors[-1].node_type in CLASSIFICATION_NODE_TYPES:
        return ancestors[-1]
    return None


def _flag_conflicting_documentation(suggestions: list[Suggestion]) -> None:
    """Two suggestions in the same category whose documented specificity contradicts (left vs
    right lung, chronic vs acute on chronic) come from contradicting statements in the note
    (copy-forward text, an imaging line...). Both are kept for the reviewer, never decided
    here, and neither may be confident."""
    by_category: dict[int, list[Suggestion]] = {}
    for suggestion in suggestions:
        suggestion.validation.setdefault("conflicting_documentation", [])
        reference = suggestion.icd_reference
        category = next(
            (h["record_id"] for h in reference["hierarchy"] if h["level"] == NodeType.CATEGORY),
            reference["record_id"],
        )
        by_category.setdefault(category, []).append(suggestion)
    for group in by_category.values():
        values: dict[str, dict[str, list[str]]] = {}
        for suggestion in group:
            for item in suggestion.validation["specificity"]["matched"]:
                attribute, _, value = item.partition("=")
                if value:
                    values.setdefault(attribute, {}).setdefault(value, []).append(suggestion.code)
        for attribute, by_value in values.items():
            if len(by_value) < 2:
                continue
            codes = sorted({c for found in by_value.values() for c in found})
            for suggestion in group:
                if suggestion.code in codes:
                    suggestion.confidence = "LOW"
                    suggestion.validation["conflicting_documentation"].append(
                        {
                            "attribute": attribute,
                            "values": sorted(by_value),
                            "codes": codes,
                            "message": "The note documents contradicting values for the same "
                            "condition; review the documentation before coding.",
                        }
                    )


def _category(node: IcdNode, ancestors: list[IcdNode]) -> IcdNode:
    """The category a record belongs to (itself for a category). `ancestors` is root-first."""
    return next((a for a in ancestors if a.node_type == NodeType.CATEGORY), node)


def _own_terms(details: dict[str, Any]) -> tuple[list[str], list[str]]:
    """(all source terms of a record, its inclusion terms)."""
    terms = details.get("terms", [])
    own = [t.term for t in terms] + [e.lead_term for e in details.get("index_terms", [])]
    inclusions = [t.term for t in terms if t.term_type == TermType.INCLUSION]
    return own, inclusions


def queries_for(concept: ClinicalConcept) -> list[str]:
    """Retrieval phrasings: the concept as written plus explicit expansions. History and
    family-history mentions search for history codes, never for the active condition."""
    base = [concept.text, *concept.expansions]
    if concept.status == AssertionStatus.FAMILY_HISTORY:
        return [f"family history of {q}" for q in base] + base
    if concept.status == AssertionStatus.HISTORY:
        prefixed = [f"personal history of {q}" for q in base if "history" not in q.lower()]
        return prefixed + base
    return base


class SuggestionService:
    def __init__(
        self,
        session: AsyncSession,
        settings: Settings,
        provider: EmbeddingProvider | None = None,
        *,
        provider_status: SemanticStatus | None = None,
        extractor: ConceptExtractor | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self._session = session
        self._settings = settings
        self._records = IcdRepository(session)
        self._retriever = HybridRetriever(
            session, settings, provider, provider_status=provider_status
        )
        self._semantic: dict[str, Any] = {}
        # Only a meaning-based provider can support a code that shares no word with the note.
        self._meaning_based = provider is not None and provider.meaning_based
        self._extractor = extractor or RuleBasedConceptExtractor()
        self._reranker = reranker or Reranker()
        self._rules = RuleEngine()

    # --- public -------------------------------------------------------------------------------

    async def suggest(
        self, request: SuggestRequest, *, trace: dict[str, Any] | None = None
    ) -> SuggestResponse:
        """`trace` (evaluation only): filled with per-stage outputs, rerank features and
        timings for every concept. It never changes the result."""
        started = time.perf_counter()
        if len(request.clinical_note) > self._settings.clinical_note_max_chars:
            raise InvalidClinicalNoteError(
                f"clinical_note exceeds {self._settings.clinical_note_max_chars} characters"
            )
        top_k = min(request.top_k, self._settings.max_top_k)
        dataset = await DatasetResolver(self._session).resolve(
            dataset_id=request.dataset_id,
            coding_system=request.coding_system,
            version=request.version,
            country=request.country,
            language=request.language,
        )
        include_uncertain = (
            request.include_uncertain
            if request.include_uncertain is not None
            else self._settings.suggest_uncertain_concepts
        )
        extraction_started = time.perf_counter()
        concepts = self._extractor.extract(request.clinical_note)
        if trace is not None:
            trace["extraction_ms"] = _elapsed_ms(extraction_started)
            trace["concepts"] = []
        if self._settings.log_clinical_text:  # opt-in debugging only; refused in production
            logger.debug(
                "extracted concepts",
                extra={"concepts": [(c.text, c.status.value) for c in concepts]},
            )
        suggestions: list[Suggestion] = []
        unmatched: list[UnmatchedConcept] = []
        for index, concept in enumerate(concepts):
            concept_trace: dict[str, Any] | None = None
            if trace is not None:
                concept_trace = {"concept_index": index, "timings_ms": {}}
                trace["concepts"].append(concept_trace)
            reason = self._not_codable_reason(concept, include_uncertain)
            if reason:
                if concept_trace is not None:
                    concept_trace["not_codable"] = reason
                unmatched.append(
                    UnmatchedConcept(
                        clinical_concept=concept.text,
                        concept_status=concept.status.value,
                        reason=reason,
                    )
                )
                continue
            others = [c for c in concepts if c is not concept]
            suggestion, miss = await self._suggest_for_concept(
                dataset, concept, others, top_k, concept_trace
            )
            if suggestion is not None and all(
                s.record_id != suggestion.record_id for s in suggestions
            ):  # a repeated statement ("Severe asthma. Severe asthma.") is one suggestion
                suggestions.append(suggestion)
            if miss is not None:
                unmatched.append(miss)

        _flag_conflicting_documentation(suggestions)
        duration = int((time.perf_counter() - started) * 1000)
        if trace is not None:
            trace["total_ms"] = _elapsed_ms(started)
        logger.info(
            "suggestion completed",
            extra={
                "dataset_id": dataset.id,
                "concepts": len(concepts),
                "suggestions": len(suggestions),
                "unmatched": len(unmatched),
                "note_chars": len(request.clinical_note),
                "duration_ms": duration,
            },
        )
        return SuggestResponse(
            dataset=dataset_summary(dataset),
            clinical_concepts=concepts,
            suggestions=suggestions,
            unmatched_concepts=unmatched,
            duration_ms=duration,
        )

    async def trace_concept(
        self,
        dataset: IcdDataset,
        concept: ClinicalConcept,
        others: list[ClinicalConcept],
        *,
        top_k: int = 5,
        include_uncertain: bool | None = None,
    ) -> tuple[Suggestion | None, UnmatchedConcept | None, dict[str, Any]]:
        """Run one concept through the pipeline and expose stage outputs (for evaluation)."""
        trace: dict[str, Any] = {"retrieval_codes": [], "reranked_codes": [], "rejected_codes": []}
        include = (
            include_uncertain
            if include_uncertain is not None
            else self._settings.suggest_uncertain_concepts
        )
        reason = self._not_codable_reason(concept, include)
        if reason:
            trace["not_codable"] = reason
            return None, None, trace
        suggestion, unmatched = await self._suggest_for_concept(
            dataset, concept, others, top_k, trace
        )
        return suggestion, unmatched, trace

    @staticmethod
    def _not_codable_reason(concept: ClinicalConcept, include_uncertain: bool) -> str | None:
        if concept.status == AssertionStatus.NEGATED:
            return "Concept is negated in the documentation; negated findings are not coded."
        if concept.status == AssertionStatus.RULED_OUT:
            return "Concept was ruled out; ruled-out conditions are not coded."
        if concept.concept_type == ConceptType.PROCEDURE:
            return (
                "Procedure/intervention: coded with an intervention classification, not with "
                "this diagnosis classification."
            )
        if concept.status in _UNCERTAIN and not include_uncertain:
            return "Uncertain/suspected diagnosis; suggestions for uncertain concepts are disabled."
        return None

    # --- per concept --------------------------------------------------------------------------

    async def _suggest_for_concept(
        self,
        dataset: IcdDataset,
        concept: ClinicalConcept,
        others: list[ClinicalConcept],
        top_k: int,
        trace: dict[str, Any] | None = None,
    ) -> tuple[Suggestion | None, UnmatchedConcept | None]:
        timings: dict[str, float] | None = (
            trace.setdefault("timings_ms", {}) if trace is not None else None
        )
        filters = RetrievalFilters(node_types=CLASSIFICATION_NODE_TYPES)
        retrieval_started = time.perf_counter()
        retrieval = await self._retriever.retrieve(
            dataset, queries_for(concept), top_k=CANDIDATE_POOL, filters=filters
        )
        self._semantic = {
            "semantic_status": retrieval.semantic_status.value,
            "embedding_space": retrieval.embedding_space,
        }
        if trace is not None and timings is not None:
            timings["retrieval_ms"] = _elapsed_ms(retrieval_started)
            trace["queries"] = queries_for(concept)
            trace["semantic_status"] = retrieval.semantic_status.value
            trace["retrieval_codes"] = [c.node.code for c in retrieval.candidates]
            trace["retrieval_record_ids"] = [c.node.id for c in retrieval.candidates]
        evaluated = await self._evaluate(dataset, concept, others, retrieval.candidates, timings)

        # Exclusion redirects ("Excludes: ... (B15)"): score the referenced code too.
        redirect_codes = {c for e in evaluated for c in e.rules.redirect_codes}
        known = {e.node.code for e in evaluated}
        missing_redirects = [c for c in redirect_codes if c not in known]
        if missing_redirects:
            targets = await self._records.get_by_codes(dataset.id, missing_redirects)
            if targets:
                extra = await self._retriever.retrieve(
                    dataset,
                    queries_for(concept),
                    top_k=CANDIDATE_POOL + len(targets),
                    filters=filters,
                    include_node_ids=[n.id for n in targets.values()],
                )
                target_ids = {n.id for n in targets.values()}
                evaluated += await self._evaluate(
                    dataset,
                    concept,
                    others,
                    [c for c in extra.candidates if c.node.id in target_ids],
                    timings,
                )

        accepted = sorted(
            (e for e in evaluated if not e.reject_reasons),
            key=lambda e: (-e.rerank, not e.node.is_selectable, e.node.code or ""),
        )
        rejected = [e for e in evaluated if e.reject_reasons]
        if trace is not None:
            trace["reranked_codes"] = [e.node.code for e in accepted]
            trace["rejected_codes"] = [e.node.code for e in rejected]
            trace["candidates"] = [
                {
                    "code": e.node.code,
                    "record_id": e.node.id,
                    "hybrid": e.candidate.hybrid_score,
                    "scores": e.candidate.scores.as_dict(),
                    "rerank": e.rerank,
                    "features": e.features.as_dict(),
                    "specificity": e.specificity.status,
                    "missing": e.specificity.missing,
                    "rule_status": e.rules.status,
                    "reject_reasons": e.reject_reasons,
                }
                for e in evaluated
            ]
        resolution_started = time.perf_counter()
        try:
            return await self._select(dataset, concept, others, top_k, accepted, rejected, timings)
        finally:
            if timings is not None:
                timings["resolution_ms"] = _elapsed_ms(resolution_started)

    async def _select(
        self,
        dataset: IcdDataset,
        concept: ClinicalConcept,
        others: list[ClinicalConcept],
        top_k: int,
        accepted: list[Evaluated],
        rejected: list[Evaluated],
        timings: dict[str, float] | None,
    ) -> tuple[Suggestion | None, UnmatchedConcept | None]:
        """Hierarchy-aware resolution of the best accepted candidate + database validation."""
        for item in rejected:
            logger.info(
                "candidate rejected by rules",
                # Reasons can quote note-derived attributes: log codes and counts only.
                extra={
                    "dataset_id": dataset.id,
                    "code": item.node.code,
                    "reason_count": len(item.reject_reasons),
                },
            )
        rejected_out = [
            {
                "code": e.node.code,
                "title": e.node.title,
                "record_id": e.node.id,
                "reasons": e.reject_reasons,
            }
            for e in rejected[:5]
        ]
        if not accepted:
            return None, UnmatchedConcept(
                clinical_concept=concept.text,
                concept_status=concept.status.value,
                reason="No candidate in this dataset is supported by the documentation."
                if rejected
                else "No matching record found in this dataset.",
                rejected_candidates=rejected_out,
            )

        db_failures = 0
        for best in accepted:
            resolved = await self._resolve(dataset, concept, others, best)
            if resolved is None:
                continue
            final, missing, resolution_alternatives, resolution = resolved
            # MANDATORY hallucination guard: the final code must be a row of this dataset.
            validation_started = time.perf_counter()
            verified = await self._records.get_by_code(
                dataset.id, final.code or "", classification_only=True
            )
            if timings is not None:
                timings["db_validation_ms"] = timings.get("db_validation_ms", 0.0) + _elapsed_ms(
                    validation_started
                )
            if (
                verified is None
                or verified.id != final.id
                or verified.dataset_id != dataset.id
                or verified.status != NodeStatus.ACTIVE
            ):
                db_failures += 1
                logger.warning(
                    "unsupported code rejected at final database validation",
                    extra={"dataset_id": dataset.id, "code": final.code},
                )
                continue
            suggestion = await self._build(
                dataset,
                concept,
                best,
                verified,
                missing,
                resolution,
                alternatives=self._alternatives(
                    [e for e in accepted if e is not best],
                    resolution_alternatives,
                    rejected,
                    final,
                    top_k,
                ),
            )
            return suggestion, None

        return None, UnmatchedConcept(
            clinical_concept=concept.text,
            concept_status=concept.status.value,
            reason="Selected code failed database validation and was rejected."
            if db_failures
            else "No candidate in this dataset is supported by the documentation.",
            rejected_candidates=rejected_out,
        )

    async def _evaluate(
        self,
        dataset: IcdDataset,
        concept: ClinicalConcept,
        others: list[ClinicalConcept],
        candidates: list[RetrievalCandidate],
        timings: dict[str, float] | None = None,
    ) -> list[Evaluated]:
        """Rules, specificity and rerank features for each candidate. `timings` (evaluation
        only) accumulates the time spent per stage."""
        if not candidates:
            return []
        clock = _StageClock(timings)
        with clock("rule_data_ms"):
            node_ids = {c.node.id for c in candidates} | {
                a.id for c in candidates for a in c.ancestors
            }
            details = await self._records.details_of_many(dataset.id, sorted(node_ids))
            target_codes = {
                r.target_code for d in details.values() for r in d["rules"] if r.target_code
            }
            known_codes = await self._records.get_by_codes(dataset.id, sorted(target_codes))
        results: list[Evaluated] = []
        for candidate in candidates:
            node_details = details.get(candidate.node.id, {})
            own, inclusions = _own_terms(node_details)
            parent = _classification_parent(candidate.ancestors)
            category = _category(candidate.node, candidate.ancestors)
            with clock("specificity_ms"):
                # Baseline = the category: whatever the code adds beyond it, at any level
                # (e.g. a subtype added by an intermediate subdivision), must be documented.
                # A code written in the note is not documentation of the details it encodes.
                specificity = check_specificity(
                    concept,
                    candidate.node.title,
                    category.title if category is not candidate.node else None,
                    own,
                )
            with clock("rules_ms"):
                rules = self._rules.evaluate(
                    concept, others, candidate.node, candidate.ancestors, details, known_codes
                )
            with clock("rerank_ms"):
                features = self._reranker.features(
                    concept, candidate, own, inclusions, specificity, rules
                )
                rerank = self._reranker.score(features)
            evaluated = Evaluated(
                candidate=candidate,
                parent=parent,
                specificity=specificity,
                rules=rules,
                features=features,
                rerank=rerank,
            )
            if rules.rejected:
                evaluated.reject_reasons += [
                    f.message for f in rules.findings if f.severity == "reject"
                ]
            if specificity.contradicted:
                evaluated.reject_reasons += specificity.conflicts
            evaluated.reject_reasons += self._status_gate(concept, candidate.node, own)
            evaluated.condition, reasons = self._condition_gate(
                concept, candidate, details, own, specificity, rules, features
            )
            evaluated.reject_reasons += reasons
            if not rules.inclusion_match and not self._has_evidence(candidate):
                evaluated.reject_reasons.append(
                    "Insufficient retrieval evidence (no signal reaches the minimum: "
                    f"lexical >= {self._settings.suggestion_min_lexical_evidence:.2f} or "
                    f"exact/fuzzy/semantic/index term >= "
                    f"{self._settings.suggestion_min_evidence:.2f})."
                )
            results.append(evaluated)
        # When the note's own words point to other conditions (candidates sharing words with
        # it, all short of support), a meaning-only match to an unrelated category is a guess:
        # "kidney disease" must not become a glucose-regulation code.
        if any(e.condition == "partial" for e in results):
            for item in results:
                if item.condition == "none" and not item.reject_reasons:
                    item.reject_reasons.append(
                        "The documented words point to other conditions; meaning-only evidence "
                        f"for {item.node.code} is not enough."
                    )
        return results

    def _condition_gate(
        self,
        concept: ClinicalConcept,
        candidate: RetrievalCandidate,
        details: dict[int, dict[str, Any]],
        own: list[str],
        specificity: SpecificityCheck,
        rules: RuleEvaluation,
        features: RerankFeatures,
    ) -> tuple[str, list[str]]:
        """(verdict, reject reasons). The documentation must name the candidate's condition
        (app.coding.condition), not only share a word with it. An exact code, title or
        source-term match settles it. A candidate sharing no word at all needs meaning-based
        (semantic) evidence, and never for a history/family-history mention or a symptom (a
        symptom never becomes the diagnosis that would explain it)."""
        if (
            candidate.scores.exact >= 1.0
            or rules.inclusion_match
            or features.exact_terminology
            or specificity.supported_by_term
        ):
            return "exempt", []
        texts = [candidate.node.title, *own]
        for ancestor in candidate.ancestors:
            if ancestor.node_type in CLASSIFICATION_NODE_TYPES:
                texts += [ancestor.title, *_own_terms(details.get(ancestor.id, {}))[0]]
        support = condition_support(concept, texts)
        code = candidate.node.code
        if support.verdict == "partial":
            return support.verdict, [
                f"The documented concept names a different or less specific condition than {code}."
            ]
        if support.verdict == "none":
            if concept.status in _STATUS_MENTIONS:
                return support.verdict, [
                    "History/family-history mention: the documented condition does not match "
                    f"{code}."
                ]
            if concept.concept_type == ConceptType.SYMPTOM:
                return support.verdict, [
                    f"Symptom: no documented word of {code}'s condition; a symptom is not "
                    "coded as a diagnosis on meaning-only evidence."
                ]
            semantic = candidate.scores.semantic or 0.0
            if not self._meaning_based or semantic < self._settings.suggestion_min_evidence:
                return support.verdict, [
                    f"No documented word of {code}'s condition, and no meaning-based (semantic) "
                    "evidence."
                ]
        return support.verdict, []

    def _has_evidence(self, candidate: RetrievalCandidate) -> bool:
        """At least one retrieval signal is strong enough to support a suggestion.

        Hierarchy context is never evidence on its own; lexical needs about half of the
        concept's words; exact/fuzzy/semantic/index-term scores are already calibrated above
        their noise floors.
        """
        scores = candidate.scores
        minimum = self._settings.suggestion_min_evidence
        return (
            scores.lexical >= self._settings.suggestion_min_lexical_evidence
            or scores.exact >= 1.0
            or scores.fuzzy >= minimum
            or scores.index_term >= minimum
            or (scores.semantic or 0.0) >= minimum
        )

    @staticmethod
    def _status_gate(concept: ClinicalConcept, node: IcdNode, own_terms: list[str]) -> list[str]:
        texts = [node.title.lower(), *(t.lower() for t in own_terms)]
        is_family_code = any("family history" in t for t in texts)
        is_history_code = not is_family_code and any(
            any(word in t for word in _HISTORY_WORDS) for t in texts
        )
        concept_mentions_history = "history" in concept.text.lower()
        if concept.status == AssertionStatus.FAMILY_HISTORY and not is_family_code:
            return ["Family-history mention: only family-history codes apply."]
        if concept.status == AssertionStatus.HISTORY and not (is_history_code or is_family_code):
            return ["Personal-history mention: an active-condition code does not apply."]
        if concept.status == AssertionStatus.HISTORY and is_family_code:
            return ["Personal-history mention: family-history codes do not apply."]
        if (
            concept.status not in (AssertionStatus.HISTORY, AssertionStatus.FAMILY_HISTORY)
            and (is_history_code or is_family_code)
            and not concept_mentions_history
            and not any(normalize_text(concept.text) == normalize_text(t) for t in own_terms)
        ):
            return ["History code, but the documentation describes a current condition."]
        return []

    # --- hierarchy-aware resolution ------------------------------------------------------------

    async def _rejection_reasons(
        self,
        dataset: IcdDataset,
        concept: ClinicalConcept,
        others: list[ClinicalConcept],
        node: IcdNode,
        ancestors: list[IcdNode],
    ) -> list[str]:
        """Status gate + coding rules for a record reached by fallback/descent (it was not
        necessarily among the retrieved candidates, so it has not been checked yet)."""
        details = await self._records.details_of_many(
            dataset.id, [node.id, *(a.id for a in ancestors)]
        )
        target_codes = {
            r.target_code for d in details.values() for r in d["rules"] if r.target_code
        }
        known_codes = await self._records.get_by_codes(dataset.id, sorted(target_codes))
        own, _ = _own_terms(details.get(node.id, {}))
        reasons = self._status_gate(concept, node, own)
        rules = self._rules.evaluate(concept, others, node, ancestors, details, known_codes)
        if rules.rejected:
            reasons += [f.message for f in rules.findings if f.severity == "reject"]
        return reasons

    async def _resolve(
        self,
        dataset: IcdDataset,
        concept: ClinicalConcept,
        others: list[ClinicalConcept],
        best: Evaluated,
    ) -> tuple[IcdNode, list[str], list[Alternative], str] | None:
        """Pick the final record for an accepted candidate, or None if it cannot be resolved
        to a record that is both supported by the documentation and allowed by the rules.

        * supported candidate -> itself (then descend if it is a non-selectable category)
        * unsupported -> nearest classification ancestor, then descend to the child that the
          documentation supports (typically the "unspecified" subdivision).
        Every record reached by fallback or descent passes the same status gate and coding
        rules as a retrieved candidate.
        """
        missing = list(best.specificity.missing)
        alternatives: list[Alternative] = []
        resolution = "direct"
        target = best.node
        ancestors = list(best.candidate.ancestors)
        if not best.specificity.supported:
            fallback = await self._supported_ancestor(dataset, concept, ancestors)
            if fallback is None:
                return None  # never return an unsupported code
            parent, parent_ancestors = fallback
            if await self._rejection_reasons(dataset, concept, others, parent, parent_ancestors):
                return None
            alternatives.append(
                Alternative(
                    record_id=best.node.id,
                    code=best.node.code,
                    title=best.node.title,
                    reason="More specific code; requires documentation of: "
                    + ", ".join(best.specificity.missing),
                    rerank_score=best.rerank,
                    is_selectable=best.node.is_selectable,
                )
            )
            target, ancestors, resolution = parent, parent_ancestors, "parent_fallback"
        elif best.specificity.is_unspecified_variant and best.parent is not None:
            # An "unspecified" code is right when the detail is undocumented; say which detail
            # the specific siblings would need.
            _, sibling_missing, sibling_alternatives = await self._descend(
                dataset, concept, others, best.parent, ancestors[:-1]
            )
            missing += [m for m in sibling_missing if m not in missing]
            alternatives += [a for a in sibling_alternatives if a.record_id != best.node.id]
        if not target.is_selectable:
            descended, child_missing, child_alternatives = await self._descend(
                dataset, concept, others, target, ancestors
            )
            alternatives += child_alternatives
            for item in child_missing:
                if item not in missing:
                    missing.append(item)
            if descended is not None:
                target = descended
                resolution = (
                    "unspecified_subdivision"
                    if resolution == "direct"
                    else "parent_then_unspecified"
                )
        return target, missing, alternatives, resolution

    async def _supported_ancestor(
        self, dataset: IcdDataset, concept: ClinicalConcept, ancestors: list[IcdNode]
    ) -> tuple[IcdNode, list[IcdNode]] | None:
        """Nearest classification ancestor whose own specificity (beyond its category) is
        documented: falling back from C00.10 skips C00.1 ("type 1" undocumented) and lands on
        the category C00. Returns (ancestor, its ancestors) or None."""
        classification = [a for a in ancestors if a.node_type in CLASSIFICATION_NODE_TYPES]
        if not classification:
            return None
        category = _category(classification[0], ancestors)
        details = await self._records.details_of_many(dataset.id, [a.id for a in classification])
        for node in reversed(classification):
            own, _ = _own_terms(details.get(node.id, {}))
            if (
                node is category
                or check_specificity(concept, node.title, category.title, own).supported
            ):
                return node, ancestors[: ancestors.index(node)]
        return None

    async def _descend(
        self,
        dataset: IcdDataset,
        concept: ClinicalConcept,
        others: list[ClinicalConcept],
        node: IcdNode,
        node_ancestors: list[IcdNode],
    ) -> tuple[IcdNode | None, list[str], list[Alternative]]:
        """The single child of `node` the documentation supports (if any), the details the
        other children would need, and those children as alternatives. Children rejected by
        coding rules or the status gate are neither selected nor offered."""
        children = (await self._records.children_of_many(dataset.id, [node.id]))[node.id]
        children = [c for c in children if c.status == NodeStatus.ACTIVE]
        if not children:
            return None, [], []
        lineage = [*node_ancestors, node]
        details = await self._records.details_of_many(
            dataset.id, [*(c.id for c in children), *(a.id for a in lineage)]
        )
        target_codes = {
            r.target_code for d in details.values() for r in d["rules"] if r.target_code
        }
        known_codes = await self._records.get_by_codes(dataset.id, sorted(target_codes))
        supported: list[tuple[IcdNode, SpecificityCheck]] = []
        missing: list[str] = []
        alternatives: list[Alternative] = []
        for child in children:
            own, _ = _own_terms(details.get(child.id, {}))
            if self._status_gate(concept, child, own):
                continue
            if self._rules.evaluate(concept, others, child, lineage, details, known_codes).rejected:
                continue
            check = check_specificity(
                concept, child.title, _category(node, node_ancestors).title, own
            )
            if check.supported:
                supported.append((child, check))
            elif not check.contradicted:
                for item in check.missing:
                    if item not in missing:
                        missing.append(item)
                alternatives.append(
                    Alternative(
                        record_id=child.id,
                        code=child.code,
                        title=child.title,
                        reason="Requires documentation of: " + ", ".join(check.missing),
                        is_selectable=child.is_selectable,
                    )
                )
        specific = [(c, k) for c, k in supported if not k.is_unspecified_variant]
        unspecified = [(c, k) for c, k in supported if k.is_unspecified_variant]
        if len(specific) == 1:
            return specific[0][0], [], alternatives
        if not specific and len(unspecified) == 1:
            return unspecified[0][0], missing, alternatives
        return None, missing, alternatives

    # --- output -------------------------------------------------------------------------------

    @staticmethod
    def _alternatives(
        accepted: list[Evaluated],
        resolution: list[Alternative],
        rejected: list[Evaluated],
        final: IcdNode,
        top_k: int,
    ) -> list[Alternative]:
        seen = {final.id}
        result: list[Alternative] = []
        for alternative in resolution:
            if alternative.record_id not in seen:
                seen.add(alternative.record_id)
                result.append(alternative)
        for item in accepted:
            if item.node.id in seen or item.rerank < MIN_ALTERNATIVE_SCORE:
                continue
            seen.add(item.node.id)
            note = (
                "Alternative candidate"
                if item.specificity.supported
                else "Alternative; requires documentation of: "
                + ", ".join(item.specificity.missing)
            )
            result.append(
                Alternative(
                    record_id=item.node.id,
                    code=item.node.code,
                    title=item.node.title,
                    reason=note,
                    rerank_score=item.rerank,
                    is_selectable=item.node.is_selectable,
                )
            )
        for item in rejected:
            for finding in item.rules.findings:
                if (
                    finding.severity == "reject"
                    and finding.target_record_id
                    and (finding.target_record_id not in seen)
                ):
                    seen.add(finding.target_record_id)
                    result.append(
                        Alternative(
                            record_id=finding.target_record_id,
                            code=finding.target_code,
                            title=finding.target_title or "",
                            reason=f"{item.node.code} excludes this documentation; the exclusion "
                            f"refers to {finding.target_code}.",
                        )
                    )
        return result[: max(top_k - 1, 0) + 3]

    def _confidence(
        self,
        concept: ClinicalConcept,
        best: Evaluated,
        final: IcdNode,
        missing: list[str],
        resolution: str,
    ) -> str:
        strong_evidence = (
            best.features.exact_terminology
            or best.features.inclusion_match
            or best.specificity.supported_by_term
            or (best.candidate.scores.lexical >= 0.8 and best.candidate.scores.fuzzy >= 0.8)
        )
        level = "LOW"
        if best.rerank >= MEDIUM_CONFIDENCE:
            level = "MEDIUM"
        if (
            best.rerank >= HIGH_CONFIDENCE
            and strong_evidence
            and not missing
            and resolution == "direct"
            and best.rules.status == "passed"
            and final.is_selectable
        ):
            level = "HIGH"
        if level == "HIGH" and concept.status in _UNCERTAIN:
            level = "MEDIUM"
        if not final.is_selectable:
            level = "LOW"
        return level

    async def _build(
        self,
        dataset: IcdDataset,
        concept: ClinicalConcept,
        best: Evaluated,
        final: IcdNode,
        missing: list[str],
        resolution: str,
        *,
        alternatives: list[Alternative],
    ) -> Suggestion:
        ancestors = await self._records.ancestors(dataset.id, final.id)
        details = (await self._records.details_of_many(dataset.id, [final.id]))[final.id]
        evidence: list[dict[str, Any]] = [
            {
                "type": "clinical_text",
                "text": concept.evidence,
                "start": concept.start,
                "end": concept.end,
                "status_cues": concept.cues,
            }
        ]
        evidence += [
            {
                "type": "matched_term",
                "text": m.matched_text,
                "match_type": m.match_type,
                "score": round(m.score, 4),
                "record_code": best.node.code,
            }
            for m in best.candidate.evidence
        ]
        if best.rules.inclusion_match:
            evidence.append({"type": "inclusion_term", "text": best.rules.inclusion_match})
        if best.specificity.matched:
            evidence.append({"type": "documented_specificity", "details": best.specificity.matched})

        status_policy = None
        if concept.status in _UNCERTAIN:
            status_policy = (
                "Uncertain/suspected diagnosis: apply the applicable coding standard for "
                "uncertain diagnoses before assigning this code."
            )
        elif concept.status == AssertionStatus.HISTORY:
            status_policy = "Personal-history mention: only history codes were considered."
        elif concept.status == AssertionStatus.FAMILY_HISTORY:
            status_policy = "Family-history mention: only family-history codes were considered."

        return Suggestion(
            clinical_concept=concept.text,
            concept_status=concept.status.value,
            concept_type=concept.concept_type.value,
            code=final.code or "",
            title=final.title,
            record_id=final.id,
            evidence=evidence,
            icd_reference={
                "record_id": final.id,
                "dataset_id": dataset.id,
                "coding_system": dataset.system,
                "version": dataset.version,
                "code": final.code,
                "normalized_code": final.normalized_code,
                "title": final.title,
                "level": final.node_type.value,
                "is_selectable": final.is_selectable,
                "hierarchy": [hierarchy_item(a).model_dump() for a in ancestors],
            },
            alternatives=alternatives,
            missing_information=missing,
            confidence=self._confidence(concept, best, final, missing, resolution),  # type: ignore[arg-type]
            retrieval_scores={
                "retrieved_record_id": best.node.id,
                "retrieved_code": best.node.code,
                **best.candidate.scores.as_dict(),
                "semantic_raw": best.candidate.scores.semantic_raw,
                "hybrid": best.candidate.hybrid_score,
                "rerank": best.rerank,
                "rerank_features": best.features.as_dict(),
                **self._semantic,
            },
            validation={
                "db_verified": True,
                "dataset_match": final.dataset_id == dataset.id,
                "record_status": final.status.value,
                "selectable": final.is_selectable,
                "resolution": resolution,
                "rule_status": best.rules.status,
                "rule_checks": [f.as_dict() for f in best.rules.findings],
                "specificity": {
                    "status": best.specificity.status,
                    "matched": best.specificity.matched,
                    "missing": best.specificity.missing,
                    "conflicts": best.specificity.conflicts,
                },
                "concept_status_policy": status_policy,
                **(
                    {}
                    if final.is_selectable
                    else {
                        "warning": "Category-level record: not a valid final code; a more "
                        "specific subdivision is required once documented."
                    }
                ),
            },
            source_provenance=provenance(final, details),
        )
