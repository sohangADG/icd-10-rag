"""ORM models. Importing this package registers every table on Base.metadata (used by Alembic)."""

from app.models.base import Base
from app.models.dataset import IcdDataset
from app.models.icd_index_entry import IcdIndexEntry
from app.models.icd_node import IcdNode
from app.models.icd_relationship import IcdRelationship
from app.models.icd_rule import IcdRule
from app.models.icd_source_ref import IcdSourceRef
from app.models.icd_term import IcdTerm
from app.models.ingestion_run import IcdIngestionError, IcdIngestionRun
from app.models.search_document import IcdSearchDocument

__all__ = [
    "Base",
    "IcdDataset",
    "IcdIndexEntry",
    "IcdIngestionError",
    "IcdIngestionRun",
    "IcdNode",
    "IcdRelationship",
    "IcdRule",
    "IcdSearchDocument",
    "IcdSourceRef",
    "IcdTerm",
]
