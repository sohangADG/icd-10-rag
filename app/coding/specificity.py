"""Specificity protection: never select a more specific code than the documentation supports.

A subdivision (subcategory / code) is compared with its classification parent. Whatever the
child *adds* to the parent's title — laterality, acuity, severity, subtype, stage, presence or
absence of a complication, encounter, anatomical site, or any other qualifier — must be stated
in the clinical concept. Missing details make the candidate unsupported (the caller falls back
to an "unspecified" sibling or the parent and reports the missing information); contradicting
details (documented "right", code says "left") reject it outright.

A concept that matches one of the candidate's own source inclusion terms or synonyms is
supported by the classification itself (e.g. "smoker" listed under "Current tobacco use").
"""

import re
from dataclasses import dataclass, field

from app.clinical.lexicons import ANATOMY, LATERALITY, SEVERITY
from app.clinical.models import AssertionStatus, ClinicalConcept
from app.core.text import normalize_text, tokenize

_GENERIC_STOP = frozenset(
    [
        "a",
        "an",
        "and",
        "as",
        "at",
        "by",
        "for",
        "from",
        "in",
        "into",
        "of",
        "on",
        "or",
        "other",
        "the",
        "to",
        "with",
        "without",
        "due",
        "not",
        "specified",
        "unspecified",
        "nos",
    ]
)
_ACUITY_WORDS = {"acute", "chronic", "subacute"}
_ENCOUNTER_WORDS = {"initial", "subsequent", "sequela", "sequelae"}
_UNSPECIFIED = re.compile(r"\b(?:unspecified|not otherwise specified|nos)\b", re.I)
_TYPE = re.compile(r"\btype\s*(\d+|i{1,2})\b", re.I)
_STAGE = re.compile(r"\bstage\s*([0-9]+[a-b]?|i{1,3}v?|iv|v)\b", re.I)
_WITH = re.compile(r"\bwith\s+(?!out\b)([a-z][a-z\s-]*?)(?=$|,|;|\bwithout\b)", re.I)
_WITHOUT = re.compile(r"\bwithout\s+([a-z][a-z\s-]*?)(?=$|,|;)", re.I)


def _stem(token: str) -> str:
    for suffix in ("ies", "es", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return token


def stems(text: str) -> set[str]:
    return {_stem(t) for t in tokenize(text)}


@dataclass
class SpecificityCheck:
    supported: bool = True
    contradicted: bool = False
    missing: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    matched: list[str] = field(default_factory=list)
    is_unspecified_variant: bool = False
    # The concept documents a detail this (unspecified) variant does not use.
    less_specific_than_documented: list[str] = field(default_factory=list)
    supported_by_term: str | None = None

    @property
    def status(self) -> str:
        if self.contradicted:
            return "contradicted"
        return "supported" if self.supported else "unsupported"


_STATUS_WORDS = {
    AssertionStatus.HISTORY: {"personal", "history", "past", "former", "previous"},
    AssertionStatus.FAMILY_HISTORY: {"family", "history"},
}


def _concept_words(concept: ClinicalConcept) -> set[str]:
    """Words the documentation states: the concept, its abbreviation expansions, the clause as
    written (status cues such as "current" are stripped from the concept text but are still
    documentation) and the words its assertion status stands for ("personal history")."""
    words = stems(concept.text) | stems(concept.evidence)
    for variant in concept.expansions:
        words |= stems(variant)
    return words | _STATUS_WORDS.get(concept.status, set())


def check_specificity(
    concept: ClinicalConcept,
    title: str,
    parent_title: str | None,
    own_terms: list[str] | None = None,
) -> SpecificityCheck:
    result = SpecificityCheck()
    attrs = concept.attributes
    concept_norm = normalize_text(concept.text)
    for term in own_terms or []:
        if normalize_text(term) == concept_norm or any(
            normalize_text(term) == normalize_text(v) for v in concept.expansions
        ):
            result.supported_by_term = term
            result.matched.append(f"source term '{term}'")
            return result
    if parent_title is None:
        return result

    title_l, parent_l = title.lower(), parent_title.lower()
    added = [t for t in tokenize(title) if _stem(t) not in stems(parent_title)]
    added_stems = {_stem(t) for t in added}
    concept_words = _concept_words(concept)
    covered: set[str] = set()

    if _UNSPECIFIED.search(title_l) and not _UNSPECIFIED.search(parent_l):
        result.is_unspecified_variant = True
        unspecified_of = {
            "side": "laterality",
            "severity": "severity",
            "stage": "stage",
            "type": "subtype",
        }
        for word, attribute in unspecified_of.items():
            if word in added_stems and getattr(attrs, attribute):
                result.less_specific_than_documented.append(attribute)
        return result

    def require(attribute: str, required: str, documented: str | None) -> None:
        if documented is None:
            # Contradicting documentation ("left and right") is unsupported, never resolved
            # by picking one of the values.
            conflicting = attribute in attrs.conflicts
            result.missing.append(f"{attribute} (conflicting)" if conflicting else attribute)
        elif documented != required:
            result.conflicts.append(
                f"{attribute}: documented '{documented}', code specifies '{required}'"
            )
        else:
            result.matched.append(f"{attribute}={required}")

    laterality = [LATERALITY[t] for t in added if t in LATERALITY]
    if laterality:
        require("laterality", laterality[0], attrs.laterality)
        covered |= {_stem(t) for t in added if t in LATERALITY}

    if "acute on chronic" in title_l and "acute on chronic" not in parent_l:
        require("acuity", "acute on chronic", attrs.acuity)
        covered |= {"acute", "on", "chronic"}
    elif acuity := [t for t in added if t in _ACUITY_WORDS]:
        require("acuity", acuity[0], attrs.acuity)
        covered |= set(acuity)

    if severity := [SEVERITY[t] for t in added if t in SEVERITY]:
        require("severity", severity[0], attrs.severity)
        covered |= set(severity)

    if (match := _TYPE.search(title_l)) and not _TYPE.search(parent_l):
        require("subtype", f"type {match.group(1)}", attrs.subtype)
        covered |= {"type", match.group(1)}
    if (match := _STAGE.search(title_l)) and not _STAGE.search(parent_l):
        require("stage", match.group(1), attrs.stage)
        covered |= {"stage", match.group(1)}

    for match in _WITH.finditer(title_l):
        phrase = match.group(1).strip()
        if phrase in parent_l:
            continue
        phrase_words = stems(phrase) - _GENERIC_STOP
        documented = attrs.complications or [
            c for c in [concept.text] if phrase_words and phrase_words <= concept_words
        ]
        if attrs.explicit_absence and not attrs.complications:
            result.conflicts.append(f"complication: documented absent, code requires '{phrase}'")
        elif not documented:
            result.missing.append(f"complication ({phrase})")
        else:
            specific = phrase_words - {"complication", "involvement"}
            if specific and not specific <= concept_words:
                result.missing.append(f"complication type ({phrase})")
            else:
                result.matched.append(f"with {phrase}")
        covered |= phrase_words | {"with"}

    for match in _WITHOUT.finditer(title_l):
        phrase = match.group(1).strip()
        if phrase in parent_l:
            continue
        if attrs.complications:
            result.conflicts.append(
                f"complication: documented {attrs.complications}, code requires none"
            )
        elif not attrs.explicit_absence:
            result.missing.append(f"absence of {phrase}")
        else:
            result.matched.append(f"without {phrase}")
        covered |= stems(phrase) | {"without"}

    if encounter := [t for t in added if t in _ENCOUNTER_WORDS]:
        require("encounter", encounter[0].rstrip("e"), attrs.encounter)
        # "initial encounter" / "initial visit": the encounter attribute covers the wording.
        covered |= set(encounter) | {"encounter", "visit"}

    for token in added:
        site = ANATOMY.get(token)
        if site and _stem(token) not in covered:
            if site not in attrs.anatomy and _stem(token) not in concept_words:
                result.missing.append(f"anatomical site ({token})")
            covered.add(_stem(token))

    leftovers = sorted(
        s for s in added_stems - covered - _GENERIC_STOP if s not in concept_words and len(s) > 2
    )
    if leftovers:
        result.missing.append("detail (" + ", ".join(leftovers) + ")")

    result.contradicted = bool(result.conflicts)
    result.supported = not result.missing and not result.conflicts
    return result
