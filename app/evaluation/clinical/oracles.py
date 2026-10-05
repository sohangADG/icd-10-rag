"""Independent evaluation oracles.

These checks deliberately do NOT reuse the extractor, the specificity guard or the rule engine:
they re-derive "is this detail documented?" from the raw words of the note so that a defect in
the pipeline cannot hide itself. They are simple and lexical on purpose (synthetic scenarios).
"""

import re
import statistics
from dataclasses import dataclass, field

from app.core.text import normalize_text, tokenize

# Words a code title can add to its parent, grouped by the specificity attribute they express.
_SIDE = {
    "left": {"left", "lt"},
    "right": {"right", "rt"},
    "both": {"both", "bilateral", "bilaterally"},
    "bilateral": {"both", "bilateral", "bilaterally"},
}
_SEVERITY = {"mild", "moderate", "severe"}
_ACUITY = {"acute", "chronic", "subacute"}
_ENCOUNTER = {"initial", "subsequent", "sequela", "sequelae"}
_SITES = {
    "knee": {"knee", "knees"},
    "hip": {"hip", "hips"},
    "lung": {"lung", "lungs", "pulmonary"},
    "lungs": {"lung", "lungs", "pulmonary"},
    "kidney": {"kidney", "kidneys", "renal", "ckd"},
    "forearm": {"forearm", "forearms", "radius", "ulna"},
}
_GENERIC = {"complication", "complications", "involvement", "disorder", "a", "an", "the", "of"}
_TYPE = re.compile(r"\btype\s*(\d+|i{1,2})\b", re.I)
_STAGE = re.compile(r"\bstage\s*(\d+|i{1,3}v?|iv|v)\b", re.I)
_WITH = re.compile(r"\bwith\s+(?!out\b)([a-z][a-z\s-]*?)(?=$|,|;|\bwithout\b)", re.I)
_WITHOUT = re.compile(r"\bwithout\s+([a-z][a-z\s-]*?)(?=$|,|;)", re.I)
_DUE_TO = re.compile(r"\bdue to\s+([a-z][a-z\s-]*?)(?=$|,|;)", re.I)
_ROMAN = {"i": "1", "ii": "2", "one": "1", "two": "2"}
_ABSENCE_WORDS = {"without", "no", "uncomplicated", "denies"}
_SUBTYPE_ABBREVIATIONS = {"t1dm": "1", "t2dm": "2"}


def _stem(word: str) -> str:
    return word[:-1] if len(word) > 3 and word.endswith("s") else word


def _words(text: str) -> set[str]:
    return {_stem(w) for w in tokenize(text)}


@dataclass
class SpecificityFinding:
    attribute: str
    detail: str


def unsupported_specificity(
    title: str, parent_title: str | None, evidence: str, source_terms: list[str]
) -> list[SpecificityFinding]:
    """Details the code title adds to its parent that the evidence text does not document.

    `evidence` is the clause the suggestion cites. A clause that contains one of the record's
    own source terms (inclusion/synonym) is supported by the classification itself.
    """
    if parent_title is None:
        return []
    clause = normalize_text(evidence)
    for term in source_terms:
        if normalize_text(term) and normalize_text(term) in clause:
            return []
    title_l, parent_l = title.lower(), parent_title.lower()
    if re.search(r"\b(unspecified|nos)\b", title_l):
        return []
    note_words = _words(evidence)
    raw_words = set(tokenize(evidence))
    added = [w for w in tokenize(title) if _stem(w) not in _words(parent_title)]
    findings: list[SpecificityFinding] = []

    for word in added:
        if word in _SIDE and not (_SIDE[word] & raw_words):
            findings.append(SpecificityFinding("laterality", word))
        if word in _SEVERITY and word not in raw_words:
            findings.append(SpecificityFinding("severity", word))
        if word in _ENCOUNTER and not ({word, word.rstrip("e"), word + "e"} & raw_words):
            findings.append(SpecificityFinding("encounter", word))
        if word in _SITES and not (_SITES[word] & raw_words):
            findings.append(SpecificityFinding("anatomy", word))
    if "acute on chronic" in title_l and "acute on chronic" not in parent_l:
        if not re.search(r"acute[\s-]+on[\s-]+chronic", evidence, re.I):
            findings.append(SpecificityFinding("acuity", "acute on chronic"))
    else:
        for word in added:
            if word in _ACUITY:
                documented = word in raw_words or (
                    word == "chronic" and re.search(r"long[\s-]standing", evidence, re.I)
                )
                if not documented:
                    findings.append(SpecificityFinding("acuity", word))
    if (match := _TYPE.search(title_l)) and not _TYPE.search(parent_l):
        wanted = _ROMAN.get(match.group(1), match.group(1))
        documented = {
            _ROMAN.get(m.group(1).lower(), m.group(1).lower())
            for m in re.finditer(r"\btype\s*(\d+|i{1,2}|one|two)\b", evidence, re.I)
        } | {v for k, v in _SUBTYPE_ABBREVIATIONS.items() if k in raw_words}
        if wanted not in documented:
            findings.append(SpecificityFinding("subtype", f"type {wanted}"))
    if (match := _STAGE.search(title_l)) and not _STAGE.search(parent_l):
        documented = {m.group(1).lower() for m in _STAGE.finditer(evidence)}
        if match.group(1) not in documented:
            findings.append(SpecificityFinding("stage", match.group(1)))
    for match in _WITH.finditer(title_l):
        phrase = match.group(1).strip()
        if phrase in parent_l:
            continue
        needed = _words(phrase) - _GENERIC
        linked = re.search(r"\b(with|complicated by|due to|secondary to)\b", evidence, re.I)
        if not linked or not needed <= note_words:
            findings.append(SpecificityFinding("complication", f"with {phrase}"))
    for match in _WITHOUT.finditer(title_l):
        phrase = match.group(1).strip()
        if phrase in parent_l:
            continue
        if not (_ABSENCE_WORDS & raw_words):
            findings.append(SpecificityFinding("complication", f"without {phrase}"))
    for match in _DUE_TO.finditer(title_l):
        phrase = match.group(1).strip()
        if phrase in parent_l:
            continue
        if not (_words(phrase) - _GENERIC) <= note_words:
            findings.append(SpecificityFinding("cause", f"due to {phrase}"))
    return findings


# --- missing information ---------------------------------------------------------------------

_MISSING_ATTRIBUTE = re.compile(
    r"^(laterality|severity|acuity|subtype|stage|encounter|anatomical site|complication|"
    r"complication type|absence of|detail)\b",
    re.I,
)


def missing_item_is_genuine(item: str, evidence: str) -> bool:
    """False when `item` (a missing_information entry) is in fact documented in `evidence`."""
    raw = set(tokenize(evidence))
    if "(conflicting)" in item:
        return True  # contradicting values: the detail is genuinely not established
    match = _MISSING_ATTRIBUTE.match(item.strip())
    if not match:
        return True
    attribute = match.group(1).lower()
    inner = re.search(r"\(([^)]*)\)", item)
    inner_words = _words(inner.group(1)) - _GENERIC if inner else set()
    if attribute == "laterality":
        return not (raw & {"left", "right", "lt", "rt", "bilateral", "both"})
    if attribute == "severity":
        return not (raw & _SEVERITY)
    if attribute == "acuity":
        return not (raw & _ACUITY) and not re.search(r"long[\s-]standing", evidence, re.I)
    if attribute == "subtype":
        return not re.search(r"\btype\s*(\d+|i{1,2}|one|two)\b", evidence, re.I) and not (
            raw & set(_SUBTYPE_ABBREVIATIONS)
        )
    if attribute == "stage":
        return not _STAGE.search(evidence)
    if attribute == "encounter":
        return not (raw & (_ENCOUNTER | {"sequela"}))
    if attribute == "absence of":
        return not (raw & _ABSENCE_WORDS)
    # anatomical site / complication / detail: genuinely missing unless all its words appear
    return not inner_words or not inner_words <= _words(evidence)


# --- concepts ------------------------------------------------------------------------------


MAX_CONTEXT_WORDS = 3


def concept_similarity(expected: str, extracted: str) -> float:
    left, right = _words(expected), _words(extracted)
    if not left or not right:
        return 0.0
    jaccard = len(left & right) / len(left | right)
    containment = len(left & right) / len(left)
    # A clause that contains the whole expected concept plus a little context still counts.
    return max(jaccard, containment if len(right - left) <= MAX_CONTEXT_WORDS else 0.0)


CONCEPT_MATCH = 0.6


def match_concepts(expected: list[str], extracted: list[list[str]]) -> dict[int, int]:
    """Greedy one-to-one matching: expected index -> extracted index. Each extracted concept
    is given as its readings (normalised concept text, clause as written); the best counts."""
    pairs = sorted(
        (
            (max(concept_similarity(e, reading) for reading in readings), i, j)
            for i, e in enumerate(expected)
            for j, readings in enumerate(extracted)
        ),
        reverse=True,
    )
    matched: dict[int, int] = {}
    used: set[int] = set()
    for score, i, j in pairs:
        if score < CONCEPT_MATCH or i in matched or j in used:
            continue
        matched[i] = j
        used.add(j)
    return matched


# --- statistics --------------------------------------------------------------------------------


@dataclass
class LatencySummary:
    values: list[float] = field(default_factory=list)

    def as_dict(self) -> dict[str, float | int]:
        if not self.values:
            return {"n": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
        ordered = sorted(self.values)
        return {
            "n": len(ordered),
            "mean": round(statistics.fmean(ordered), 2),
            "p50": round(percentile(ordered, 50), 2),
            "p95": round(percentile(ordered, 95), 2),
            "max": round(ordered[-1], 2),
        }


def percentile(ordered: list[float], pct: float) -> float:
    """Linear-interpolated percentile of an already sorted list."""
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * pct / 100
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def rate(numerator: int, denominator: int) -> float | None:
    """None (not 0) when nothing was measured, so empty categories are visible as such."""
    return round(numerator / denominator, 4) if denominator else None
