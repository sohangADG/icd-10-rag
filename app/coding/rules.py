"""Coding-rule evaluation for a candidate code against the documented concepts.

Rules attached to the candidate *and its classification ancestors* apply (an exclusion on a
category covers its subdivisions). Outcomes:

* EXCLUDES — if the concept being coded matches the exclusion text, the candidate is REJECTED
  and the excluded-to code (when the source states exactly one) is offered instead. A match
  against a *different* concept in the note is reported as a warning.
* INCLUDES / inclusion terms — a concept matching an inclusion term supports the candidate.
* CODE_FIRST / USE_ADDITIONAL_CODE / CODE_ALSO — returned as instructions; referenced codes are
  checked to exist in the same dataset, and linked to other concepts in the note when possible.
* NOTE / SEE / SEE_ALSO / OTHER — returned for the coder to read; never auto-applied.

Matching is token-based and conservative; anything ambiguous is reported, not decided.
"""

from dataclasses import dataclass, field
from typing import Any

from app.clinical.models import ClinicalConcept
from app.coding.specificity import stems
from app.core.constants import CLASSIFICATION_NODE_TYPES, RULE_TO_INSTRUCTION, RuleType, TermType
from app.core.text import normalize_text
from app.ingestion.codes import find_code_references
from app.models import IcdNode, IcdRule, IcdTerm

_MATCH_STOP = frozenset(
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
        "nos",
    ]
)
EXCLUSION_MATCH_THRESHOLD = 0.75


@dataclass
class RuleFinding:
    rule_type: str
    text: str
    severity: str  # reject | warn | info
    message: str
    attached_to: str | None  # code of the record carrying the rule
    target_code: str | None = None
    target_record_id: int | None = None
    target_exists: bool | None = None
    target_title: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class RuleEvaluation:
    rejected: bool = False
    findings: list[RuleFinding] = field(default_factory=list)
    inclusion_match: str | None = None
    exclusion_conflict: bool = False
    redirect_codes: list[str] = field(default_factory=list)  # excluded-to codes to consider

    @property
    def status(self) -> str:
        if self.rejected:
            return "rejected"
        return "warning" if any(f.severity == "warn" for f in self.findings) else "passed"


def _content_stems(text: str) -> set[str]:
    without_codes = text
    for code in find_code_references(text).all_referenced:
        without_codes = without_codes.replace(code, " ")
    return stems(without_codes.replace("(", " ").replace(")", " ")) - _MATCH_STOP


def matches_text(concept: ClinicalConcept, text: str, threshold: float) -> bool:
    target = _content_stems(text)
    if not target:
        return False
    concept_words = stems(concept.text)
    for variant in concept.expansions:
        concept_words |= stems(variant)
    return len(target & concept_words) / len(target) >= threshold


def inclusion_matches(concept: ClinicalConcept, term: str) -> bool:
    normalized = normalize_text(term)
    variants = [normalize_text(concept.text), *(normalize_text(v) for v in concept.expansions)]
    if normalized in variants:
        return True
    term_stems = _content_stems(term)
    return bool(term_stems) and any(term_stems == (stems(v) - _MATCH_STOP) for v in variants)


class RuleEngine:
    def evaluate(
        self,
        concept: ClinicalConcept,
        other_concepts: list[ClinicalConcept],
        candidate: IcdNode,
        ancestors: list[IcdNode],
        details: dict[int, dict[str, Any]],
        known_codes: dict[str, IcdNode],
    ) -> RuleEvaluation:
        evaluation = RuleEvaluation()
        scope = [candidate] + [
            a for a in reversed(ancestors) if a.node_type in CLASSIFICATION_NODE_TYPES
        ]
        own_terms: list[IcdTerm] = details.get(candidate.id, {}).get("terms", [])
        for term in own_terms:
            if term.term_type == TermType.INCLUSION and inclusion_matches(concept, term.term):
                evaluation.inclusion_match = term.term
                evaluation.findings.append(
                    RuleFinding(
                        "INCLUDES",
                        term.term,
                        "info",
                        f"Concept matches inclusion term '{term.term}'.",
                        candidate.code,
                    )
                )
                break

        for node in scope:
            rules: list[IcdRule] = details.get(node.id, {}).get("rules", [])
            for rule in rules:
                target = known_codes.get(rule.target_code or "")
                target_info = {
                    "target_code": rule.target_code,
                    "target_record_id": target.id if target else rule.target_node_id,
                    "target_exists": (target is not None or rule.target_node_id is not None)
                    if rule.target_code
                    else None,
                    "target_title": target.title if target else None,
                }
                instruction = RULE_TO_INSTRUCTION.get(rule.rule_type)
                label = instruction.value if instruction else rule.rule_type.value
                if rule.rule_type == RuleType.EXCLUDE:
                    if matches_text(concept, rule.rule_text, EXCLUSION_MATCH_THRESHOLD):
                        evaluation.rejected = True
                        evaluation.exclusion_conflict = True
                        if rule.target_code:
                            evaluation.redirect_codes.append(rule.target_code)
                        evaluation.findings.append(
                            RuleFinding(
                                label,
                                rule.rule_text,
                                "reject",
                                f"Documented concept matches an exclusion of {node.code}: "
                                f"'{rule.rule_text}'.",
                                node.code,
                                **target_info,
                            )
                        )
                        continue
                    conflicting = [
                        other.text
                        for other in other_concepts
                        if other.codable
                        and matches_text(other, rule.rule_text, EXCLUSION_MATCH_THRESHOLD)
                    ]
                    if conflicting:
                        evaluation.findings.append(
                            RuleFinding(
                                label,
                                rule.rule_text,
                                "warn",
                                f"Another documented concept ({'; '.join(conflicting)}) matches "
                                f"an exclusion of {node.code}; review the combination.",
                                node.code,
                                **target_info,
                            )
                        )
                    continue
                if rule.rule_type in (
                    RuleType.CODE_FIRST,
                    RuleType.USE_ADDITIONAL_CODE,
                    RuleType.CODE_ALSO,
                ):
                    message = {
                        RuleType.CODE_FIRST: "Sequencing instruction: code the referenced "
                        "condition first if documented.",
                        RuleType.USE_ADDITIONAL_CODE: "An additional code may be required.",
                        RuleType.CODE_ALSO: "Code also the referenced condition if documented.",
                    }[rule.rule_type]
                    evaluation.findings.append(
                        RuleFinding(
                            label, rule.rule_text, "info", message, node.code, **target_info
                        )
                    )
                    continue
                evaluation.findings.append(
                    RuleFinding(
                        label,
                        rule.rule_text,
                        "info",
                        "Coding note for the reviewer.",
                        node.code,
                        **target_info,
                    )
                )
        return evaluation
