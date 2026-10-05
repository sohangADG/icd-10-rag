"""Code-format utilities shared by adapters, validators, retrieval and the suggestion engine.

Nothing here knows about a specific coding system's valid code set: format checks only.
Per-system code patterns live in app.ingestion.validator.
"""

import re
from dataclasses import dataclass

from app.core.constants import CLASSIFICATION_NODE_TYPES, NodeType

# Typographic markers that decorate codes in printed classifications (dagger/asterisk/etc.).
_CODE_DECORATIONS = str.maketrans("", "", "†‡*✝+")
_DASHES = str.maketrans({"–": "-", "—": "-", "‐": "-", "‑": "-", "−": "-"})

# Generic ICD-10-family code token: letter, two alphanumerics, optional dot + 1-4 alphanumerics.
# Used to *find* code references in text; validity is decided by the dataset's validator.
CODE_TOKEN = r"[A-Z][0-9][0-9A-Z](?:\.[0-9A-Z]{1,4})?"
CODE_TOKEN_RE = re.compile(rf"(?<![A-Za-z0-9.])({CODE_TOKEN})(\.-|-(?![A-Z0-9]))?(?![A-Za-z0-9])")
RANGE_RE = re.compile(rf"(?<![A-Za-z0-9.])({CODE_TOKEN})\s*-\s*({CODE_TOKEN})(?![A-Za-z0-9])")
_CODE_KEY_RE = re.compile(r"^([A-Z])([0-9]{2})(.*)$")


def clean_code(code: str) -> str:
    """Display form: trimmed, upper-case, decorations and inner whitespace removed."""
    return re.sub(r"\s+", "", code.translate(_CODE_DECORATIONS).translate(_DASHES)).upper()


def normalize_code(code: str, level: NodeType | None = None) -> str:
    """Lookup form. Classification codes drop all punctuation ("A00.0" -> "A000"); grouping
    codes (chapters, block ranges such as "A00-A09") keep their hyphen."""
    cleaned = clean_code(code)
    if level is not None and level not in CLASSIFICATION_NODE_TYPES:
        return cleaned
    if RANGE_RE.fullmatch(cleaned):
        return cleaned
    return re.sub(r"[^A-Z0-9]", "", cleaned)


def record_key(level: NodeType | None, code: str | None, fallback: str) -> str:
    """Stable identity of a normalized record within one source."""
    if code:
        return f"{level.value if level else 'UNKNOWN'}:{normalize_code(code, level)}"
    return f"{level.value if level else 'UNKNOWN'}:#{fallback}"


@dataclass(frozen=True)
class CodeReferences:
    codes: list[str]  # explicit single codes, in order of appearance ("B00.-" -> "B00")
    ranges: list[tuple[str, str]]

    @property
    def single_target(self) -> str | None:
        """The target code only when the text references exactly one code and no range.

        Anything else is ambiguous and must not be turned into a structured target.
        """
        if len(self.codes) == 1 and not self.ranges:
            return self.codes[0]
        return None

    @property
    def all_referenced(self) -> list[str]:
        return self.codes + [f"{start}-{end}" for start, end in self.ranges]


def find_code_references(text: str) -> CodeReferences:
    """Codes explicitly written in a piece of instruction text. Never invents codes."""
    normalized = text.translate(_DASHES)
    ranges = [(m.group(1), m.group(2)) for m in RANGE_RE.finditer(normalized)]
    without_ranges = RANGE_RE.sub(" ", normalized)
    codes: list[str] = []
    for match in CODE_TOKEN_RE.finditer(without_ranges):
        code = match.group(1)
        if code not in codes:
            codes.append(code)
    return CodeReferences(codes=codes, ranges=ranges)


def parse_range(code: str) -> tuple[str, str] | None:
    match = RANGE_RE.fullmatch(clean_code(code))
    return (match.group(1), match.group(2)) if match else None


def _sort_key(code: str) -> tuple[str, int, str]:
    match = _CODE_KEY_RE.match(normalize_code(code))
    if not match:
        return (code, 0, "")
    return (match.group(1), int(match.group(2)), match.group(3))


def code_in_range(code: str, start: str, end: str) -> bool:
    """Whether a category-level code falls inside an inclusive block range like A00-A09."""
    key = _sort_key(code)[:2]
    return _sort_key(start)[:2] <= key <= _sort_key(end)[:2]


def truncation_parents(code: str) -> list[str]:
    """Candidate parents by removing trailing characters, nearest first.

    "A00.10" -> ["A00.1", "A00"]. Only a *fallback*: hierarchy is taken from the source first,
    and any inferred link must still resolve to an existing record (see hierarchy.py).
    """
    cleaned = clean_code(code)
    if "." not in cleaned:
        return []
    stem, _, suffix = cleaned.partition(".")
    candidates = [f"{stem}.{suffix[:i]}" for i in range(len(suffix) - 1, 0, -1)]
    return [*candidates, stem]


def level_from_code_shape(code: str) -> NodeType:
    """Default level for a classification code by shape: A00 -> CATEGORY, A00.0 ->
    SUBCATEGORY, A00.00+ -> CODE. Adapters may override from explicit source data."""
    cleaned = clean_code(code)
    if "." not in cleaned:
        return NodeType.CATEGORY
    return NodeType.SUBCATEGORY if len(cleaned.partition(".")[2]) == 1 else NodeType.CODE


def looks_like_code(text: str) -> bool:
    return CODE_TOKEN_RE.fullmatch(clean_code(text)) is not None
