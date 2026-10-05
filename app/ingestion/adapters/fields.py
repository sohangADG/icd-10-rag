"""Mapping of source fields onto normalized records, shared by every structured adapter.

A *row* is a dict of canonical field name -> raw value, produced by a format-specific reader
using an operator-configured field mapping. This module turns rows into NormalizedICDRecords and
attaches rows from separate rule / index-term tables to their owners.
"""

from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel, Field

from app.core.constants import InstructionType, NodeStatus, NodeType
from app.ingestion.codes import clean_code, level_from_code_shape, normalize_code, record_key
from app.ingestion.instructions import (
    explicit_exclusion_type,
    make_exclusion,
    make_instruction,
    parse_instruction_type,
)
from app.ingestion.models import (
    IndexTermKind,
    NormalizedICDRecord,
    NormalizedInclusion,
    NormalizedIndexTerm,
    Severity,
    SourceProvenance,
    ValidationIssue,
)

# Canonical record fields an adapter mapping may point at.
RECORD_FIELDS = (
    "code",
    "title",
    "description",
    "level",
    "parent_code",
    "parent_level",
    "range_start",
    "range_end",
    "selectable",
    "status",
    "sort_order",
    "inclusions",
    "exclusions",
    "includes_notes",
    "notes",
    "code_also",
    "use_additional_code",
    "code_first",
    "see",
    "see_also",
    "other_instructions",
    "index_terms",
    "synonyms",
    "abbreviations",
)
# Canonical fields of a long-format rules table (one row per rule / term).
RULE_FIELDS = ("owner_code", "owner_level", "rule_type", "text", "target_code", "exclusion_type")

INSTRUCTION_FIELDS: dict[str, InstructionType] = {
    "includes_notes": InstructionType.INCLUDES,
    "notes": InstructionType.NOTE,
    "code_also": InstructionType.CODE_ALSO,
    "use_additional_code": InstructionType.USE_ADDITIONAL_CODE,
    "code_first": InstructionType.CODE_FIRST,
    "see": InstructionType.SEE,
    "see_also": InstructionType.SEE_ALSO,
    "other_instructions": InstructionType.OTHER,
}
_TERM_FIELDS: dict[str, IndexTermKind] = {
    "index_terms": IndexTermKind.INDEX_TERM,
    "synonyms": IndexTermKind.SYNONYM,
    "abbreviations": IndexTermKind.ABBREVIATION,
}
_TRUE = {"1", "true", "t", "yes", "y", "x"}
_FALSE = {"0", "false", "f", "no", "n", ""}


class FieldMapping(BaseModel):
    """Which source field holds each canonical field. Unmapped canonical fields default to a
    source field of the same name (so a source using canonical headers needs no mapping)."""

    columns: dict[str, str] = Field(default_factory=dict)
    rule_columns: dict[str, str] = Field(default_factory=dict)
    # Separator for multi-valued cells ("a | b | c"). Lists in JSON/XML need none.
    list_separator: str = "|"
    # Source level labels -> NodeType, e.g. {"chap": "CHAPTER", "cat": "CATEGORY"}.
    level_values: dict[str, NodeType] = Field(default_factory=dict)
    # When no level is given: derive CATEGORY/SUBCATEGORY/CODE from the code's shape.
    level_from_code_shape: bool = True

    def source_field(self, canonical: str) -> str:
        return self.columns.get(canonical, canonical)

    def rule_source_field(self, canonical: str) -> str:
        return self.rule_columns.get(canonical, canonical)


def as_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    text = " ".join(str(value).split())
    return text or None


def as_list(value: Any, separator: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list | tuple):
        items: Iterable[Any] = value
    else:
        text = str(value)
        items = text.split(separator) if separator and separator in text else [text]
    return [t for t in (as_text(item) for item in items) if t]


def as_bool(value: Any) -> bool | None:
    if value is None or isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return None if text == "" else False
    return None


class RecordAssembler:
    """Builds normalized records from canonical rows and collects row-level parser issues."""

    def __init__(self, mapping: FieldMapping) -> None:
        self.mapping = mapping
        self.issues: list[ValidationIssue] = []
        self._records: dict[str, NormalizedICDRecord] = {}  # first occurrence per key
        self._all: list[NormalizedICDRecord] = []
        self._by_code: dict[str, list[str]] = {}
        self._counter = 0

    # --- records ----------------------------------------------------------------------------

    def parse_level(self, value: Any, code: str | None) -> NodeType | None:
        text = as_text(value)
        if text:
            if text in self.mapping.level_values:
                return NodeType(self.mapping.level_values[text])
            upper = text.upper()
            if upper in NodeType.__members__:
                return NodeType[upper]
            return None
        if code and self.mapping.level_from_code_shape:
            return level_from_code_shape(code)
        return None

    def add_row(
        self, row: dict[str, Any], provenance: SourceProvenance, *, parent_code: str | None = None
    ) -> NormalizedICDRecord | None:
        get = row.get
        raw_code = as_text(get("code"))
        code = clean_code(raw_code) if raw_code else None
        raw_level = get("level")
        level = self.parse_level(raw_level, code)
        if as_text(raw_level) and level is None:
            self._issue(
                Severity.ERROR,
                "INVALID_LEVEL",
                f"Unknown hierarchy level {as_text(raw_level)!r}.",
                provenance,
                code,
            )
        self._counter += 1
        parent_level_text = as_text(get("parent_level"))
        record = NormalizedICDRecord(
            key=record_key(level, code, str(self._counter)),
            code=code,
            normalized_code=normalize_code(code, level) if code else None,
            title=as_text(get("title")),
            description=as_text(get("description")),
            level=level,
            parent_code=clean_code(p) if (p := as_text(get("parent_code"))) else parent_code,
            parent_level=self.parse_level(parent_level_text, None) if parent_level_text else None,
            range_start=as_text(get("range_start")),
            range_end=as_text(get("range_end")),
            is_selectable=as_bool(get("selectable")),
            status=NodeStatus.DISABLED
            if (as_text(get("status")) or "").lower() in {"disabled", "inactive", "retired"}
            else NodeStatus.ACTIVE,
            sort_order=self._counter,
            provenance=provenance,
            raw_text=as_text(get("raw_text")),
        )
        sep = self.mapping.list_separator
        record.inclusions = [
            NormalizedInclusion(text=t, provenance=provenance)
            for t in as_list(get("inclusions"), sep)
        ]
        record.exclusions = [
            make_exclusion(t, provenance=provenance) for t in as_list(get("exclusions"), sep)
        ]
        for field, instruction_type in INSTRUCTION_FIELDS.items():
            record.instructions.extend(
                make_instruction(instruction_type, t, provenance=provenance)
                for t in as_list(get(field), sep)
            )
        for field, kind in _TERM_FIELDS.items():
            record.index_terms.extend(
                NormalizedIndexTerm(term=t, kind=kind, provenance=provenance)
                for t in as_list(get(field), sep)
            )
        self._store(record)
        return record

    def add_record(self, record: NormalizedICDRecord) -> None:
        """Register a record built elsewhere (e.g. by the ClaML reader)."""
        self._counter += 1
        if record.sort_order is None:
            record.sort_order = self._counter
        self._store(record)

    def _store(self, record: NormalizedICDRecord) -> None:
        # Duplicates are kept: the hierarchy builder reports them with full provenance.
        self._all.append(record)
        if record.key not in self._records:
            self._records[record.key] = record
            if record.code:
                self._by_code.setdefault(normalize_code(record.code), []).append(record.key)

    def records(self) -> list[NormalizedICDRecord]:
        """All records in source order, duplicates included."""
        return list(self._all)

    # --- separate rule / term tables -----------------------------------------------------------

    def find_owner(self, code: str, level: NodeType | None) -> NormalizedICDRecord | None:
        keys = self._by_code.get(normalize_code(code), [])
        candidates = [self._records[k] for k in keys]
        if level is not None:
            candidates = [c for c in candidates if c.level == level] or candidates
        return candidates[0] if candidates else None

    def add_rule_row(self, row: dict[str, Any], provenance: SourceProvenance) -> None:
        owner_code = as_text(row.get("owner_code"))
        rule_type_text = as_text(row.get("rule_type"))
        text = as_text(row.get("text"))
        if not text:
            self._issue(
                Severity.ERROR, "EMPTY_RULE", "Rule row has no text.", provenance, owner_code
            )
            return
        if not owner_code:
            self._issue(
                Severity.ERROR, "RULE_WITHOUT_OWNER", "Rule row has no owner code.", provenance
            )
            return
        owner_level_text = as_text(row.get("owner_level"))
        owner = self.find_owner(
            clean_code(owner_code),
            self.parse_level(owner_level_text, None) if owner_level_text else None,
        )
        if owner is None:
            self._issue(
                Severity.ERROR,
                "RULE_WITHOUT_OWNER",
                f"Rule owner {owner_code!r} does not exist in the source.",
                provenance,
                owner_code,
            )
            return
        explicit_target = as_text(row.get("target_code"))
        explicit_target = clean_code(explicit_target) if explicit_target else None
        kind = (rule_type_text or "").upper().replace(" ", "_")
        if kind in {"INCLUSION", "INCLUSION_TERM"}:
            owner.inclusions.append(NormalizedInclusion(text=text, provenance=provenance))
            return
        if kind in IndexTermKind.__members__:
            owner.index_terms.append(
                NormalizedIndexTerm(
                    term=text,
                    kind=IndexTermKind[kind],
                    target_code=explicit_target,
                    provenance=provenance,
                )
            )
            return
        instruction_type = parse_instruction_type(rule_type_text or "")
        if instruction_type is None:
            self._issue(
                Severity.ERROR,
                "UNKNOWN_RULE_TYPE",
                f"Unknown rule type {rule_type_text!r}.",
                provenance,
                owner_code,
            )
            return
        if instruction_type is InstructionType.EXCLUDES:
            exclusion_type = as_text(row.get("exclusion_type")) or as_text(rule_type_text)
            owner.exclusions.append(
                make_exclusion(
                    text,
                    exclusion_type=explicit_exclusion_type(exclusion_type or ""),
                    explicit_target=explicit_target,
                    provenance=provenance,
                )
            )
        else:
            owner.instructions.append(
                make_instruction(
                    instruction_type, text, explicit_target=explicit_target, provenance=provenance
                )
            )

    def _issue(
        self,
        severity: Severity,
        code: str,
        message: str,
        provenance: SourceProvenance,
        record_code: str | None = None,
    ) -> None:
        self.issues.append(
            ValidationIssue(
                severity=severity,
                code=code,
                message=message,
                record_code=record_code,
                locator=provenance.to_locator(),
            )
        )


def canonical_row(source_row: dict[str, Any], mapping: FieldMapping) -> dict[str, Any]:
    """Project a source row (keys = source column names) onto canonical record fields."""
    return {
        field: source_row.get(mapping.source_field(field))
        for field in RECORD_FIELDS
        if mapping.source_field(field) in source_row
    }


def canonical_rule_row(source_row: dict[str, Any], mapping: FieldMapping) -> dict[str, Any]:
    return {
        field: source_row.get(mapping.rule_source_field(field))
        for field in RULE_FIELDS
        if mapping.rule_source_field(field) in source_row
    }
