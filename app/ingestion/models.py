"""Source-independent normalized models produced by every source adapter.

These are deliberately separate from the SQLAlchemy ORM models: adapters know nothing about the
database, and repositories know nothing about PDF/XML/CSV internals. The importer is the only
component that maps between the two.
"""

from datetime import date
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.core.constants import InstructionType, NodeStatus, NodeType, SourceType


class Severity(StrEnum):
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


class SourceProvenance(BaseModel):
    """Where a piece of normalized data came from. Every record must carry one."""

    model_config = ConfigDict(frozen=True)

    source_filename: str
    kind: SourceType
    page_start: int | None = None  # physical PDF page (1-based)
    page_end: int | None = None
    printed_page: str | None = None  # page label printed on the page, when determinable
    row: int | None = None  # 1-based data row for CSV/TSV/XLSX
    sheet: str | None = None
    element_path: str | None = None  # XML/JSON path of the source element
    line: int | None = None  # 1-based line within the (cleaned) text stream

    def to_locator(self) -> dict[str, Any]:
        """Compact JSON locator persisted next to each row (no None values)."""
        return {
            key: value
            for key, value in self.model_dump(mode="json").items()
            if value is not None and key != "source_filename"
        }


class NormalizedInclusion(BaseModel):
    text: str = Field(min_length=1)
    provenance: SourceProvenance | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class NormalizedExclusion(BaseModel):
    text: str = Field(min_length=1)
    # Only when the source distinguishes exclusion kinds explicitly (e.g. EXCLUDES1/EXCLUDES2).
    exclusion_type: str | None = None
    # Set only when the source states exactly one code; never inferred.
    target_code: str | None = None
    referenced_codes: list[str] = Field(default_factory=list)
    provenance: SourceProvenance | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class NormalizedInstruction(BaseModel):
    instruction_type: InstructionType
    text: str = Field(min_length=1)
    target_code: str | None = None
    referenced_codes: list[str] = Field(default_factory=list)
    provenance: SourceProvenance | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class IndexTermKind(StrEnum):
    INDEX_TERM = "INDEX_TERM"
    SYNONYM = "SYNONYM"
    ABBREVIATION = "ABBREVIATION"


class NormalizedIndexTerm(BaseModel):
    """A source-provided index term / synonym. Ingestion never generates these."""

    term: str = Field(min_length=1)
    kind: IndexTermKind = IndexTermKind.INDEX_TERM
    modifier: str | None = None
    # Code the term points to. None means "the owning record".
    target_code: str | None = None
    see: str | None = None  # "see <term>" cross-reference, as printed
    see_also: str | None = None
    provenance: SourceProvenance | None = None


class NormalizedICDRecord(BaseModel):
    """One logical ICD entity (chapter, block, category, subcategory or code)."""

    key: str  # stable identity within this source: "<LEVEL>:<normalized code>"
    code: str | None = None
    normalized_code: str | None = None
    title: str | None = None
    description: str | None = None
    level: NodeType | None = None
    # Explicit, source-provided parent. Hierarchy inference is a separate, configurable step.
    parent_code: str | None = None
    parent_level: NodeType | None = None
    range_start: str | None = None
    range_end: str | None = None
    is_selectable: bool | None = None  # None: decided by the hierarchy builder (leaf => True)
    status: NodeStatus = NodeStatus.ACTIVE
    sort_order: int | None = None
    inclusions: list[NormalizedInclusion] = Field(default_factory=list)
    exclusions: list[NormalizedExclusion] = Field(default_factory=list)
    instructions: list[NormalizedInstruction] = Field(default_factory=list)
    index_terms: list[NormalizedIndexTerm] = Field(default_factory=list)
    provenance: SourceProvenance | None = None
    raw_text: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DatasetMetadata(BaseModel):
    """Identity and source facts of one dataset. Supplied by the operator's manifest and/or
    inspected from the source (hash, page count). Never guessed from a filename."""

    model_config = ConfigDict(str_strip_whitespace=True)

    coding_system: str = Field(min_length=1, max_length=64)
    modification: str | None = Field(default=None, max_length=64)
    country: str = Field(min_length=2, max_length=8)
    version: str = Field(min_length=1, max_length=32)
    revision: str | None = Field(default=None, max_length=32)
    edition: str | None = Field(default=None, max_length=64)
    language: str = Field(min_length=2, max_length=16)
    title: str | None = None
    publisher: str = Field(min_length=1, max_length=255)
    publication_year: int | None = Field(default=None, gt=1900)
    source_type: SourceType | None = None
    source_filename: str | None = None
    source_identifier: str | None = None
    source_uri: str | None = None
    source_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    source_page_count: int | None = Field(default=None, ge=0)
    effective_from: date | None = None
    effective_to: date | None = None
    # Operator's licence statement for this source (see docs/licensing.md).
    licence: dict[str, Any] = Field(default_factory=dict)
    extra: dict[str, Any] = Field(default_factory=dict)

    @field_validator("coding_system")
    @classmethod
    def _upper_system(cls, value: str) -> str:
        return value.upper()


class ValidationIssue(BaseModel):
    severity: Severity
    code: str  # machine-readable issue code, e.g. DUPLICATE_CODE
    message: str
    record_key: str | None = None
    record_code: str | None = None
    locator: dict[str, Any] | None = None

    @property
    def is_fatal(self) -> bool:
        return self.severity is Severity.ERROR


class SourceInspection(BaseModel):
    """Format-level facts about a source, obtained without importing anything."""

    source_type: SourceType
    source_filename: str
    size_bytes: int
    sha256: str
    page_count: int | None = None
    encrypted: bool = False
    text_extraction_permitted: bool = True
    details: dict[str, Any] = Field(default_factory=dict)
