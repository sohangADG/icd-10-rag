"""Source adapter interface.

An adapter turns one source file into normalized records. It owns everything format-specific
(PDF layout, CSV dialects, XML paths...) and nothing else: no validation policy, no hierarchy
inference, no database access.
"""

import hashlib
import json
from abc import ABC, abstractmethod
from collections.abc import Iterator
from functools import cached_property
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, Field, ValidationError

from app.core.constants import SourceType
from app.ingestion.hierarchy import HierarchyConfig
from app.ingestion.models import (
    DatasetMetadata,
    NormalizedICDRecord,
    Severity,
    SourceInspection,
    SourceProvenance,
    ValidationIssue,
)


class AdapterError(Exception):
    """The source cannot be read with this adapter/mapping (fatal for the run)."""


class SourceRestrictedError(AdapterError):
    """The source restricts the requested processing (e.g. a PDF whose permissions forbid text
    extraction). Adapters never attempt to bypass such restrictions."""


class SourceManifest(BaseModel):
    """Operator-supplied description of a source: dataset identity + how to read it.

    Example (JSON):
        {"adapter": "csv",
         "dataset": {"coding_system": "ICD-10-CA", "version": "2022", "country": "CA",
                     "language": "en", "publisher": "...",
                     "licence": {"basis": "CIHI licence #..."}},
         "mapping": {"columns": {"code": "Code", "title": "Description"}},
         "hierarchy": {"infer_parents": false}}
    """

    adapter: SourceType | None = None
    dataset: dict[str, Any] = Field(default_factory=dict)
    mapping: dict[str, Any] = Field(default_factory=dict)
    hierarchy: HierarchyConfig = Field(default_factory=HierarchyConfig)
    # Directory used to resolve relative paths inside `mapping` (e.g. a separate rules file).
    base_dir: Path | None = None

    @classmethod
    def from_file(cls, path: Path) -> "SourceManifest":
        data = json.loads(path.read_text(encoding="utf-8"))
        data.setdefault("base_dir", str(path.parent))
        return cls.model_validate(data)


def sha256_of_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


class SourceAdapter(ABC):
    source_type: ClassVar[SourceType]
    extensions: ClassVar[tuple[str, ...]] = ()

    def __init__(self, path: Path, manifest: SourceManifest | None = None) -> None:
        self.path = Path(path)
        self.manifest = manifest or SourceManifest()
        self._issues: list[ValidationIssue] = []
        if not self.path.is_file():
            raise AdapterError(f"Source file not found: {self.path.name}")

    # --- detection / inspection -------------------------------------------------------------

    @classmethod
    def detect(cls, path: Path) -> float:
        """Confidence (0..1) that this adapter can read `path`. Cheap: extension + magic."""
        return 0.6 if path.suffix.lower() in cls.extensions else 0.0

    def get_source_hash(self) -> str:
        return self._sha256

    @cached_property
    def _sha256(self) -> str:
        return sha256_of_file(self.path)

    def inspect_metadata(self) -> SourceInspection:
        return SourceInspection(
            source_type=self.source_type,
            source_filename=self.path.name,
            size_bytes=self.path.stat().st_size,
            sha256=self.get_source_hash(),
            details=self._inspection_details(),
        )

    def _inspection_details(self) -> dict[str, Any]:
        return {}

    def get_provenance(self) -> SourceProvenance:
        """Document-level provenance (records carry their own, finer-grained provenance)."""
        return SourceProvenance(source_filename=self.path.name, kind=self.source_type)

    def get_dataset_metadata(self) -> DatasetMetadata:
        """Manifest identity + facts measured from the source. Raises AdapterError if the
        manifest lacks required identity fields: identity is never guessed from a filename."""
        inspection = self.inspect_metadata()
        values = {
            "source_type": self.source_type,
            "source_filename": self.path.name,
            **self._embedded_dataset_metadata(),
            **self.manifest.dataset,
            "source_sha256": inspection.sha256,
        }
        if inspection.page_count is not None:
            values["source_page_count"] = inspection.page_count
        try:
            return DatasetMetadata.model_validate(values)
        except ValidationError as exc:
            missing = ", ".join(".".join(str(p) for p in e["loc"]) for e in exc.errors())
            raise AdapterError(f"Invalid or incomplete dataset metadata: {missing}") from exc

    def _embedded_dataset_metadata(self) -> dict[str, Any]:
        """Dataset identity stated inside the source itself (e.g. a JSON header)."""
        return {}

    # --- records ------------------------------------------------------------------------------

    @abstractmethod
    def iterate_records(self) -> Iterator[NormalizedICDRecord]:
        """Yield normalized records in source order. Parser problems are collected as issues
        (see validation_messages) rather than raised, unless the whole source is unreadable."""

    def validation_messages(self) -> list[ValidationIssue]:
        return list(self._issues)

    def _report(
        self,
        severity: Severity,
        code: str,
        message: str,
        *,
        locator: dict[str, Any] | None = None,
        record_code: str | None = None,
    ) -> None:
        self._issues.append(
            ValidationIssue(
                severity=severity,
                code=code,
                message=message,
                locator=locator,
                record_code=record_code,
            )
        )

    def _resolve_relative(self, value: str) -> Path:
        candidate = Path(value)
        if candidate.is_absolute():
            return candidate
        base = self.manifest.base_dir or self.path.parent
        return base / candidate
