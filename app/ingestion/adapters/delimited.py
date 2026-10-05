"""CSV and TSV adapters with configurable column mapping, encoding handling and optional
separate long-format rules / index-term files.

Mapping (manifest "mapping" section):
    {"encoding": "auto",                 # or "utf-8", "cp1252", ...
     "delimiter": ",",                   # TSV adapter defaults to tab
     "columns": {"code": "Code", "title": "Description", "parent_code": "Parent"},
     "list_separator": "|",
     "level_values": {"C": "CATEGORY"},
     "rules_file": "rules.csv",          # optional, relative to the manifest
     "rule_columns": {"owner_code": "Code", "rule_type": "Type", "text": "Text"}}
"""

import csv
import io
from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar

from app.core.constants import SourceType
from app.ingestion.adapters.base import AdapterError, SourceAdapter, SourceManifest
from app.ingestion.adapters.fields import (
    FieldMapping,
    RecordAssembler,
    canonical_row,
    canonical_rule_row,
)
from app.ingestion.models import NormalizedICDRecord, SourceProvenance

AUTO_ENCODINGS = ("utf-8-sig", "cp1252", "latin-1")


class DelimitedMapping(FieldMapping):
    encoding: str = "auto"
    delimiter: str | None = None
    rules_file: str | None = None
    index_terms_file: str | None = None


def read_text(path: Path, encoding: str) -> tuple[str, str]:
    """Decode a file. "auto" tries UTF-8 (with BOM) first, then common legacy encodings."""
    raw = path.read_bytes()
    candidates = AUTO_ENCODINGS if encoding == "auto" else (encoding,)
    for candidate in candidates:
        try:
            return raw.decode(candidate), candidate
        except UnicodeDecodeError:
            continue
    raise AdapterError(f"Cannot decode {path.name} with encodings {', '.join(candidates)}")


class CsvAdapter(SourceAdapter):
    source_type: ClassVar[SourceType] = SourceType.CSV
    extensions: ClassVar[tuple[str, ...]] = (".csv",)
    default_delimiter: ClassVar[str] = ","

    def __init__(self, path: Path, manifest: SourceManifest | None = None) -> None:
        super().__init__(path, manifest)
        self.mapping = DelimitedMapping.model_validate(self.manifest.mapping)
        self.detected_encoding: str | None = None

    def _rows(self, path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
        text, encoding = read_text(path, self.mapping.encoding)
        if path == self.path:
            self.detected_encoding = encoding
        reader = csv.DictReader(
            io.StringIO(text, newline=""),
            delimiter=self.mapping.delimiter or self.default_delimiter,
        )
        if not reader.fieldnames:
            raise AdapterError(f"{path.name} has no header row")
        reader.fieldnames = [name.strip() for name in reader.fieldnames]
        # Data row numbers are 1-based and exclude the header (row 1 = first data line).
        for index, row in enumerate(reader, start=1):
            yield index, {k: v for k, v in row.items() if k is not None}

    def _inspection_details(self) -> dict[str, Any]:
        text, encoding = read_text(self.path, self.mapping.encoding)
        reader = csv.reader(
            io.StringIO(text, newline=""),
            delimiter=self.mapping.delimiter or self.default_delimiter,
        )
        header = next(reader, [])
        return {
            "encoding": encoding,
            "columns": [h.strip() for h in header],
            "data_rows": sum(1 for _ in reader),
        }

    def iterate_records(self) -> Iterator[NormalizedICDRecord]:
        assembler = RecordAssembler(self.mapping)
        for row_number, row in self._rows(self.path):
            provenance = SourceProvenance(
                source_filename=self.path.name, kind=self.source_type, row=row_number
            )
            assembler.add_row(canonical_row(row, self.mapping), provenance)
        for file_name in (self.mapping.rules_file, self.mapping.index_terms_file):
            if not file_name:
                continue
            rules_path = self._resolve_relative(file_name)
            if not rules_path.is_file():
                raise AdapterError(f"Rules file not found: {rules_path.name}")
            for row_number, row in self._rows(rules_path):
                provenance = SourceProvenance(
                    source_filename=rules_path.name, kind=self.source_type, row=row_number
                )
                assembler.add_rule_row(canonical_rule_row(row, self.mapping), provenance)
        self._issues.extend(assembler.issues)
        yield from assembler.records()


class TsvAdapter(CsvAdapter):
    source_type: ClassVar[SourceType] = SourceType.TSV
    extensions: ClassVar[tuple[str, ...]] = (".tsv", ".tab")
    default_delimiter: ClassVar[str] = "\t"
