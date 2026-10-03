from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    ForeignKey,
    Index,
    Integer,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.core.constants import CrossReferenceType
from app.models.base import Base, CreatedAtMixin, IdMixin, same_dataset_fk, str_enum


class IcdIndexEntry(IdMixin, CreatedAtMixin, Base):
    """An Alphabetical Index entry, kept hierarchical (lead term -> modifiers -> sub-modifiers).

    The Alphabetical Index is a separate knowledge structure from the Tabular List; candidate_code
    is what the index *points to* and still requires Tabular List verification.
    """

    __tablename__ = "icd_index_entries"
    __table_args__ = (
        UniqueConstraint("dataset_id", "id", name="uq_icd_index_entries_dataset_id_id"),
        same_dataset_fk("parent_id", "icd_index_entries", ondelete="CASCADE"),
        same_dataset_fk("target_node_id", "icd_nodes"),
        Index("ix_icd_index_entries_dataset_id_normalized_term", "dataset_id", "normalized_term"),
        CheckConstraint("depth >= 0", name="depth_non_negative"),
        CheckConstraint("parent_id IS NULL OR parent_id <> id", name="not_own_parent"),
        CheckConstraint("source_page > 0", name="source_page_positive"),
    )

    dataset_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("icd_datasets.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    parent_id: Mapped[int | None] = mapped_column(BigInteger, index=True)
    lead_term: Mapped[str] = mapped_column(Text, nullable=False)
    normalized_term: Mapped[str] = mapped_column(Text, nullable=False)
    modifier: Mapped[str | None] = mapped_column(Text)
    depth: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    candidate_code: Mapped[str | None] = mapped_column(String(32), index=True)
    target_node_id: Mapped[int | None] = mapped_column(BigInteger, index=True)
    cross_reference_type: Mapped[CrossReferenceType | None] = mapped_column(
        str_enum(CrossReferenceType, "cross_reference_type")
    )
    cross_reference_target: Mapped[str | None] = mapped_column(Text)
    source_page: Mapped[int] = mapped_column(Integer, nullable=False)
