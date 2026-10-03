from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    ForeignKey,
    Index,
    Integer,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.core.constants import CLASSIFICATION_NODE_TYPES, NodeStatus, NodeType
from app.models.base import Base, IdMixin, TimestampMixin, same_dataset_fk, str_enum

_CLASSIFICATION_TYPES_SQL = ", ".join(f"'{t.value}'" for t in CLASSIFICATION_NODE_TYPES)


class IcdNode(IdMixin, TimestampMixin, Base):
    """A node in the Tabular List hierarchy: chapter, block, category, subcategory or code.

    `code` is free text: ICD-10-CA extends codes to 5th/6th characters, so no fixed length is
    assumed. Chapters/blocks typically carry a range (code_from/code_to) instead of a code.
    Retired codes are kept with status=disabled rather than deleted.
    """

    __tablename__ = "icd_nodes"
    __table_args__ = (
        # Target of same-dataset composite FKs from every other knowledge table.
        UniqueConstraint("dataset_id", "id", name="uq_icd_nodes_dataset_id_id"),
        same_dataset_fk("parent_id", "icd_nodes"),
        # A classification code (A00, A00.0, Canadian extensions...) exists once per dataset.
        Index(
            "uq_icd_nodes_dataset_classification_code",
            "dataset_id",
            "code",
            unique=True,
            postgresql_where=text(
                f"code IS NOT NULL AND node_type IN ({_CLASSIFICATION_TYPES_SQL})"
            ),
        ),
        # Chapter/block codes (e.g. "I", "A00-A09") are unique per type within a dataset.
        Index(
            "uq_icd_nodes_dataset_grouping_code",
            "dataset_id",
            "node_type",
            "code",
            unique=True,
            postgresql_where=text(
                f"code IS NOT NULL AND node_type NOT IN ({_CLASSIFICATION_TYPES_SQL})"
            ),
        ),
        Index("ix_icd_nodes_dataset_id_code", "dataset_id", "code"),
        CheckConstraint("depth >= 0", name="depth_non_negative"),
        CheckConstraint("parent_id IS NULL OR parent_id <> id", name="not_own_parent"),
        CheckConstraint(
            "source_page_end IS NULL OR source_page_start IS NULL "
            "OR source_page_end >= source_page_start",
            name="source_page_range",
        ),
    )

    dataset_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("icd_datasets.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    parent_id: Mapped[int | None] = mapped_column(BigInteger, index=True)
    node_type: Mapped[NodeType] = mapped_column(
        str_enum(NodeType, "node_type"), nullable=False, index=True
    )
    code: Mapped[str | None] = mapped_column(String(32), index=True)
    code_from: Mapped[str | None] = mapped_column(String(32))
    code_to: Mapped[str | None] = mapped_column(String(32))
    title: Mapped[str] = mapped_column(Text, nullable=False)
    depth: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    is_selectable: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=text("false")
    )
    is_canadian_extension: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=text("false")
    )
    status: Mapped[NodeStatus] = mapped_column(
        str_enum(NodeStatus, "status"),
        nullable=False,
        default=NodeStatus.ACTIVE,
        server_default=NodeStatus.ACTIVE.value,
    )
    introduced_version: Mapped[str | None] = mapped_column(String(32))
    disabled_version: Mapped[str | None] = mapped_column(String(32))
    source_page_start: Mapped[int | None] = mapped_column(Integer)
    source_page_end: Mapped[int | None] = mapped_column(Integer)
