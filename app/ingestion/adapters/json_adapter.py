"""JSON adapter: configurable record path, nested children (hierarchy by nesting), nested rule
lists, optional embedded dataset header.

Mapping:
    {"dataset_path": ["dataset"],        # optional: embedded dataset identity
     "records_path": ["chapters"],       # path to the top-level record array
     "children_key": "children",         # nested children => parent taken from nesting
     "columns": {"code": "id", "title": "label"},
     "rule_lists": {"exclusions": "excludes"}}   # canonical list field -> JSON key

Nested rule items may be strings or objects with "text" (+ optional "target_code",
"exclusion_type"); explicit targets are kept, others are parsed from the text.
"""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar

from pydantic import Field

from app.core.constants import SourceType
from app.ingestion.adapters.base import AdapterError, SourceAdapter, SourceManifest
from app.ingestion.adapters.fields import (
    INSTRUCTION_FIELDS,
    FieldMapping,
    RecordAssembler,
    canonical_row,
)
from app.ingestion.codes import clean_code
from app.ingestion.instructions import explicit_exclusion_type, make_exclusion, make_instruction
from app.ingestion.models import (
    NormalizedICDRecord,
    Severity,
    SourceProvenance,
    ValidationIssue,
)

_STRUCTURED_LIST_FIELDS = (
    "exclusions",
    "includes_notes",
    "notes",
    "code_also",
    "use_additional_code",
    "code_first",
    "see",
    "see_also",
    "other_instructions",
)


class JsonMapping(FieldMapping):
    dataset_path: list[str] = Field(default_factory=list)
    records_path: list[str] = Field(default_factory=lambda: ["records"])
    children_key: str | None = "children"
    max_bytes: int = 512 * 1024 * 1024


def _dig(document: Any, path: list[str]) -> Any:
    current = document
    for part in path:
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


class JsonAdapter(SourceAdapter):
    source_type: ClassVar[SourceType] = SourceType.JSON
    extensions: ClassVar[tuple[str, ...]] = (".json",)

    def __init__(self, path: Path, manifest: SourceManifest | None = None) -> None:
        super().__init__(path, manifest)
        self.mapping = JsonMapping.model_validate(self.manifest.mapping)
        self._document: Any = None

    def _load(self) -> Any:
        if self._document is None:
            if self.path.stat().st_size > self.mapping.max_bytes:
                raise AdapterError(f"{self.path.name} exceeds the configured JSON size limit")
            try:
                self._document = json.loads(self.path.read_text(encoding="utf-8-sig"))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise AdapterError(f"Invalid JSON in {self.path.name}: {exc}") from exc
        return self._document

    def _embedded_dataset_metadata(self) -> dict[str, Any]:
        if not self.mapping.dataset_path:
            return {}
        header = _dig(self._load(), self.mapping.dataset_path)
        return header if isinstance(header, dict) else {}

    def _inspection_details(self) -> dict[str, Any]:
        records = _dig(self._load(), self.mapping.records_path)
        return {"top_level_records": len(records) if isinstance(records, list) else None}

    def iterate_records(self) -> Iterator[NormalizedICDRecord]:
        records = _dig(self._load(), self.mapping.records_path)
        if not isinstance(records, list):
            raise AdapterError(
                f"records_path {'.'.join(self.mapping.records_path)!r} is not a list"
            )
        assembler = RecordAssembler(self.mapping)
        base_path = "/".join(self.mapping.records_path)
        stack: list[tuple[Any, str, str | None]] = [
            (item, f"{base_path}[{i}]", None) for i, item in reversed(list(enumerate(records)))
        ]
        while stack:
            item, element_path, parent_code = stack.pop()
            if not isinstance(item, dict):
                assembler.issues.append(
                    self._parser_issue(f"Record at {element_path} is not an object.", element_path)
                )
                continue
            provenance = SourceProvenance(
                source_filename=self.path.name, kind=self.source_type, element_path=element_path
            )
            row = canonical_row(item, self.mapping)
            structured = {
                f: row.pop(f) for f in _STRUCTURED_LIST_FIELDS if _is_object_list(row.get(f))
            }
            record = assembler.add_row(row, provenance, parent_code=parent_code)
            if record is not None:
                self._attach_structured(record, structured, provenance)
            children = item.get(self.mapping.children_key) if self.mapping.children_key else None
            if isinstance(children, list) and record is not None:
                stack.extend(
                    (child, f"{element_path}/{self.mapping.children_key}[{i}]", record.code)
                    for i, child in reversed(list(enumerate(children)))
                )
        self._issues.extend(assembler.issues)
        yield from assembler.records()

    @staticmethod
    def _attach_structured(
        record: NormalizedICDRecord,
        structured: dict[str, list[dict[str, Any]]],
        provenance: SourceProvenance,
    ) -> None:
        for field, items in structured.items():
            for item in items:
                text = " ".join(str(item.get("text", "")).split())
                if not text:
                    continue
                target = item.get("target_code")
                target = clean_code(str(target)) if target else None
                if field == "exclusions":
                    record.exclusions.append(
                        make_exclusion(
                            text,
                            exclusion_type=explicit_exclusion_type(
                                str(item.get("exclusion_type") or "")
                            ),
                            explicit_target=target,
                            provenance=provenance,
                        )
                    )
                else:
                    record.instructions.append(
                        make_instruction(
                            INSTRUCTION_FIELDS[field],
                            text,
                            explicit_target=target,
                            provenance=provenance,
                        )
                    )

    def _parser_issue(self, message: str, element_path: str) -> ValidationIssue:
        return ValidationIssue(
            severity=Severity.ERROR,
            code="PARSER_FAILURE",
            message=message,
            locator={"kind": self.source_type.value, "element_path": element_path},
        )


def _is_object_list(value: Any) -> bool:
    return isinstance(value, list) and any(isinstance(v, dict) for v in value)
