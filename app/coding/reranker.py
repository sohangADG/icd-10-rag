"""Candidate reranking with explicit, inspectable features.

rerank = w_hybrid*hybrid + w_exact*exact_terminology + w_overlap*concept_overlap
       + w_inclusion*inclusion_match + w_specific*documented_specificity
       + w_selectable*selectable - w_unsupported*unsupported_specificity
       - w_less_specific*less_specific_than_documented - w_rule_warning*rule_warning

Exclusion conflicts and contradicted specificity are not "scored down" — they reject the
candidate (handled by the suggestion service). Semantic similarity contributes only through the
hybrid score, so a candidate can never win on vector similarity alone.
"""

from dataclasses import asdict, dataclass

from app.clinical.models import ClinicalConcept
from app.coding.rules import RuleEvaluation, inclusion_matches
from app.coding.specificity import SpecificityCheck, stems
from app.retrieval.hybrid import RetrievalCandidate

_FUNCTION_WORDS = frozenset({"a", "an", "the", "of", "in", "on", "to", "for", "by"})
_OVERLAP_STOP = frozenset(
    [
        "a",
        "an",
        "and",
        "of",
        "the",
        "in",
        "on",
        "to",
        "with",
        "or",
        "for",
        "by",
        "other",
        "unspecified",
    ]
)


@dataclass(frozen=True)
class RerankWeights:
    hybrid: float = 0.45
    exact_terminology: float = 0.25
    concept_overlap: float = 0.15
    inclusion_match: float = 0.10
    documented_specificity: float = 0.05
    selectable: float = 0.05
    unsupported_specificity: float = 0.20
    less_specific_than_documented: float = 0.10
    rule_warning: float = 0.05


@dataclass
class RerankFeatures:
    hybrid: float
    exact_terminology: float
    concept_overlap: float
    inclusion_match: float
    documented_specificity: float
    selectable: float
    unsupported_specificity: float
    less_specific_than_documented: float
    rule_warning: float

    def as_dict(self) -> dict[str, float]:
        return {k: round(v, 4) for k, v in asdict(self).items()}


def exact_terminology(
    concept: ClinicalConcept, candidate: RetrievalCandidate, terms: list[str]
) -> float:
    """1.0 when the concept (or an abbreviation expansion) equals the title or one of the
    record's own source terms after normalization (function words such as "the" ignored)."""

    def key(text: str) -> frozenset[str]:
        return frozenset(stems(text) - _FUNCTION_WORDS)

    variants = {key(concept.text), *(key(v) for v in concept.expansions)}
    targets = {key(candidate.node.title), *(key(t) for t in terms)}
    return 1.0 if (variants & targets) - {frozenset()} else 0.0


def concept_overlap(
    concept: ClinicalConcept, candidate: RetrievalCandidate, terms: list[str]
) -> float:
    """Jaccard overlap between concept words and the candidate's title + own terms."""
    concept_words = stems(concept.text)
    for variant in concept.expansions:
        concept_words |= stems(variant)
    concept_words -= _OVERLAP_STOP
    best = 0.0
    for text in [candidate.node.title, *terms]:
        words = stems(text) - _OVERLAP_STOP
        if words and concept_words:
            best = max(best, len(words & concept_words) / len(words | concept_words))
    return best


class Reranker:
    def __init__(self, weights: RerankWeights | None = None) -> None:
        self.weights = weights or RerankWeights()

    def features(
        self,
        concept: ClinicalConcept,
        candidate: RetrievalCandidate,
        own_terms: list[str],
        inclusion_terms: list[str],
        specificity: SpecificityCheck,
        rules: RuleEvaluation,
    ) -> RerankFeatures:
        return RerankFeatures(
            hybrid=candidate.hybrid_score,
            exact_terminology=exact_terminology(concept, candidate, own_terms),
            concept_overlap=concept_overlap(concept, candidate, own_terms),
            inclusion_match=1.0
            if rules.inclusion_match or any(inclusion_matches(concept, t) for t in inclusion_terms)
            else 0.0,
            documented_specificity=1.0 if specificity.matched else 0.0,
            selectable=1.0 if candidate.node.is_selectable else 0.0,
            unsupported_specificity=0.0 if specificity.supported else 1.0,
            less_specific_than_documented=1.0 if specificity.less_specific_than_documented else 0.0,
            rule_warning=1.0 if rules.status == "warning" else 0.0,
        )

    def score(self, features: RerankFeatures) -> float:
        w = self.weights
        value = (
            w.hybrid * features.hybrid
            + w.exact_terminology * features.exact_terminology
            + w.concept_overlap * features.concept_overlap
            + w.inclusion_match * features.inclusion_match
            + w.documented_specificity * features.documented_specificity
            + w.selectable * features.selectable
            - w.unsupported_specificity * features.unsupported_specificity
            - w.less_specific_than_documented * features.less_specific_than_documented
            - w.rule_warning * features.rule_warning
        )
        return round(max(0.0, min(1.0, value)), 4)
