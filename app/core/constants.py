"""Domain vocabularies and well-known dataset identities.

Enums are persisted as VARCHAR + CHECK constraints (not native PostgreSQL enums), so adding a
value later is a single, reversible migration that rewrites the CHECK constraint.
"""

from enum import StrEnum


class DatasetStatus(StrEnum):
    PENDING = "pending"
    INGESTING = "ingesting"
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
    INCLUDE = "INCLUDE"
    EXCLUDE = "EXCLUDE"
    NOTE = "NOTE"
    USE_ADDITIONAL_CODE = "USE_ADDITIONAL_CODE"
    CODE_SEPARATELY = "CODE_SEPARATELY"
    DAGGER = "DAGGER"
    ASTERISK = "ASTERISK"
    CROSS_REFERENCE = "CROSS_REFERENCE"
    INSTRUCTION = "INSTRUCTION"


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
