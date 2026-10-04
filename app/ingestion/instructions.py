"""Recognition of coding-instruction markers ("Includes:", "Excludes:", "Code first"...) and
construction of normalized rule objects from instruction text.

Only markers that are explicitly present are recognised; nothing is inferred from prose.
"""

import re
from dataclasses import dataclass

from app.core.constants import InstructionType
from app.ingestion.codes import find_code_references
from app.ingestion.models import (
    NormalizedExclusion,
    NormalizedInstruction,
    SourceProvenance,
)


@dataclass(frozen=True)
class MarkerMatch:
    instruction_type: InstructionType
    exclusion_type: str | None
    # Text after a label marker ("Excludes: foo" -> "foo"). For phrase markers that are part of
    # the instruction itself ("Code first underlying ...") the full line is kept.
    remainder: str
    is_label: bool


# Order matters: longer / more specific markers first.
_LABEL_MARKERS: tuple[tuple[re.Pattern[str], InstructionType, str | None], ...] = (
    (re.compile(r"^excludes\s*1\s*:?\s*", re.I), InstructionType.EXCLUDES, "EXCLUDES1"),
    (re.compile(r"^excludes\s*2\s*:?\s*", re.I), InstructionType.EXCLUDES, "EXCLUDES2"),
    (re.compile(r"^(?:excludes|excl\.)\s*:\s*", re.I), InstructionType.EXCLUDES, None),
    (re.compile(r"^excludes\s*$", re.I), InstructionType.EXCLUDES, None),
    (re.compile(r"^(?:includes|incl\.)\s*:\s*", re.I), InstructionType.INCLUDES, None),
    (re.compile(r"^includes\s*$", re.I), InstructionType.INCLUDES, None),
    (re.compile(r"^notes?\s*:\s*", re.I), InstructionType.NOTE, None),
    (re.compile(r"^notes?\s*$", re.I), InstructionType.NOTE, None),
)
_PHRASE_MARKERS: tuple[tuple[re.Pattern[str], InstructionType], ...] = (
    (re.compile(r"^use\s+additional\s+code\b", re.I), InstructionType.USE_ADDITIONAL_CODE),
    (re.compile(r"^code\s+first\b", re.I), InstructionType.CODE_FIRST),
    (re.compile(r"^code\s+also\b", re.I), InstructionType.CODE_ALSO),
    (re.compile(r"^see\s+also\b", re.I), InstructionType.SEE_ALSO),
    (re.compile(r"^see\b(?!\s+also)", re.I), InstructionType.SEE),
)

_ALIASES = {
    "INCLUDE": InstructionType.INCLUDES,
    "INCLUSION": InstructionType.INCLUDES,
    "EXCLUDE": InstructionType.EXCLUDES,
    "EXCLUSION": InstructionType.EXCLUDES,
    "EXCLUDES1": InstructionType.EXCLUDES,
    "EXCLUDES2": InstructionType.EXCLUDES,
    "NOTES": InstructionType.NOTE,
    "CODEALSO": InstructionType.CODE_ALSO,
    "USEADDITIONALCODE": InstructionType.USE_ADDITIONAL_CODE,
    "CODEFIRST": InstructionType.CODE_FIRST,
    "SEEALSO": InstructionType.SEE_ALSO,
}


def match_marker(line: str) -> MarkerMatch | None:
    stripped = line.strip()
    for pattern, instruction_type, exclusion_type in _LABEL_MARKERS:
        match = pattern.match(stripped)
        if match:
            return MarkerMatch(instruction_type, exclusion_type, stripped[match.end() :], True)
    for pattern, instruction_type in _PHRASE_MARKERS:
        if pattern.match(stripped):
            return MarkerMatch(instruction_type, None, stripped, False)
    return None


def parse_instruction_type(value: str) -> InstructionType | None:
    """Map an explicit type label from structured data (e.g. a CSV column) to the vocabulary."""
    key = re.sub(r"[^A-Z0-9]", "", value.strip().upper().replace(" ", ""))
    for member in InstructionType:
        if key == member.value.replace("_", ""):
            return member
    return _ALIASES.get(key)


def explicit_exclusion_type(value: str) -> str | None:
    key = re.sub(r"[^A-Z0-9]", "", value.upper())
    return key if key in {"EXCLUDES1", "EXCLUDES2"} else None


def make_exclusion(
    text: str,
    *,
    exclusion_type: str | None = None,
    explicit_target: str | None = None,
    provenance: SourceProvenance | None = None,
) -> NormalizedExclusion:
    references = find_code_references(text)
    return NormalizedExclusion(
        text=text.strip(),
        exclusion_type=exclusion_type,
        target_code=explicit_target or references.single_target,
        referenced_codes=references.all_referenced,
        provenance=provenance,
    )


def make_instruction(
    instruction_type: InstructionType,
    text: str,
    *,
    explicit_target: str | None = None,
    provenance: SourceProvenance | None = None,
) -> NormalizedInstruction:
    references = find_code_references(text)
    return NormalizedInstruction(
        instruction_type=instruction_type,
        text=text.strip(),
        target_code=explicit_target or references.single_target,
        referenced_codes=references.all_referenced,
        provenance=provenance,
    )
