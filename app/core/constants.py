"""Domain vocabularies and well-known dataset identities.

Enums are persisted as VARCHAR + CHECK constraints (not native PostgreSQL enums), so adding a
value later is a single, reversible migration that rewrites the CHECK constraint.
"""

from enum import StrEnum


class DatasetStatus(StrEnum):
    """Dataset lifecycle. Only READY datasets are served by retrieval/suggestion APIs.

    PENDING -> PROCESSING -> INDEXING -> READY
                  |-> VALIDATION_FAILED   (fatal validation errors; no content persisted)
                  |-> FAILED              (unexpected failure; transaction rolled back)
    READY -> ARCHIVED                     (kept for history, no longer served)
    """

    PENDING = "pending"
    PROCESSING = "processing"
    VALIDATION_FAILED = "validation_failed"
    INDEXING = "indexing"
    READY = "ready"
    FAILED = "failed"
    ARCHIVED = "archived"


class NodeType(StrEnum):
    CHAPTER = "CHAPTER"
    BLOCK = "BLOCK"
    CATEGORY = "CATEGORY"
    SUBCATEGORY = "SUBCATEGORY"
    CODE = "CODE"


# Node types whose `code` is a classification code that must be unique within a dataset.
CLASSIFICATION_NODE_TYPES: tuple[NodeType, ...] = (
    NodeType.CATEGORY,
    NodeType.SUBCATEGORY,
    NodeType.CODE,
)

# Grouping levels: they organise codes but are never themselves assigned to an encounter.
GROUPING_NODE_TYPES: tuple[NodeType, ...] = (NodeType.CHAPTER, NodeType.BLOCK)

# Canonical top-down order, used for depth sanity checks (a child is never "above" its parent).
NODE_TYPE_RANK: dict[NodeType, int] = {
    NodeType.CHAPTER: 0,
    NodeType.BLOCK: 1,
    NodeType.CATEGORY: 2,
    NodeType.SUBCATEGORY: 3,
    NodeType.CODE: 4,
}


class NodeStatus(StrEnum):
    ACTIVE = "active"
    DISABLED = "disabled"


class TermType(StrEnum):
    OFFICIAL_TITLE = "OFFICIAL_TITLE"
    INCLUSION = "INCLUSION"
    SYNONYM = "SYNONYM"
    INDEX_TERM = "INDEX_TERM"
    ABBREVIATION = "ABBREVIATION"
    ALTERNATIVE_TERM = "ALTERNATIVE_TERM"


class RuleType(StrEnum):
    """Persisted coding-instruction type (icd_rules.rule_type).

    INCLUDE/EXCLUDE are the Phase 1 spellings of the INCLUDES/EXCLUDES instruction types; see
    InstructionType for the source-independent vocabulary and the mapping between them.
    """

    INCLUDE = "INCLUDE"
    EXCLUDE = "EXCLUDE"
    NOTE = "NOTE"
    USE_ADDITIONAL_CODE = "USE_ADDITIONAL_CODE"
    CODE_SEPARATELY = "CODE_SEPARATELY"
    CODE_ALSO = "CODE_ALSO"
    CODE_FIRST = "CODE_FIRST"
    DAGGER = "DAGGER"
    ASTERISK = "ASTERISK"
    CROSS_REFERENCE = "CROSS_REFERENCE"
    SEE = "SEE"
    SEE_ALSO = "SEE_ALSO"
    INSTRUCTION = "INSTRUCTION"
    OTHER = "OTHER"


class InstructionType(StrEnum):
    """Source-independent coding instruction vocabulary used by adapters and the API."""

    INCLUDES = "INCLUDES"
    EXCLUDES = "EXCLUDES"
    NOTE = "NOTE"
    CODE_ALSO = "CODE_ALSO"
    USE_ADDITIONAL_CODE = "USE_ADDITIONAL_CODE"
    CODE_FIRST = "CODE_FIRST"
    SEE = "SEE"
    SEE_ALSO = "SEE_ALSO"
    OTHER = "OTHER"


INSTRUCTION_TO_RULE: dict[InstructionType, RuleType] = {
    InstructionType.INCLUDES: RuleType.INCLUDE,
    InstructionType.EXCLUDES: RuleType.EXCLUDE,
    InstructionType.NOTE: RuleType.NOTE,
    InstructionType.CODE_ALSO: RuleType.CODE_ALSO,
    InstructionType.USE_ADDITIONAL_CODE: RuleType.USE_ADDITIONAL_CODE,
    InstructionType.CODE_FIRST: RuleType.CODE_FIRST,
    InstructionType.SEE: RuleType.SEE,
    InstructionType.SEE_ALSO: RuleType.SEE_ALSO,
    InstructionType.OTHER: RuleType.OTHER,
}
RULE_TO_INSTRUCTION: dict[RuleType, InstructionType] = {
    rule: instruction for instruction, rule in INSTRUCTION_TO_RULE.items()
}


class RelationshipType(StrEnum):
    PARENT = "PARENT"
    CHILD = "CHILD"
    EXCLUDES = "EXCLUDES"
    INCLUDES = "INCLUDES"
    DAGGER_ASTERISK = "DAGGER_ASTERISK"
    ADDITIONAL_CODE = "ADDITIONAL_CODE"
    CROSS_REFERENCE = "CROSS_REFERENCE"
    SEE = "SEE"
    SEE_ALSO = "SEE_ALSO"


class CrossReferenceType(StrEnum):
    SEE = "SEE"
    SEE_ALSO = "SEE_ALSO"
    SEE_CONDITION = "SEE_CONDITION"


class IngestionRunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIALLY_COMPLETED = "partially_completed"
    VALIDATION_FAILED = "validation_failed"
    SKIPPED = "skipped"


class SourceType(StrEnum):
    PDF = "pdf"
    PDF_OCR = "pdf_ocr"
    TEXT = "text"
    CSV = "csv"
    TSV = "tsv"
    JSON = "json"
    XML = "xml"
    XLSX = "xlsx"


# Search document type for the one-retrieval-unit-per-ICD-entity representation.
SEARCH_DOCUMENT_TYPE_RECORD = "icd_record"

# Identity of the first target dataset. Only metadata — no classification content.
ICD10CA_2022 = {
    "system": "ICD-10-CA",
    "country": "CA",
    "version": "2022",
    "publisher": "Canadian Institute for Health Information",
    "publication_year": 2022,
    "language": "en",
    "source_filename": "ICD10CA_2022_final.pdf",
}
