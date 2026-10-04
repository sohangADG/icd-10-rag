from typing import Any

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
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.core.constants import CLASSIFICATION_NODE_TYPES, NodeStatus, NodeType
from app.models.base import Base, IdMixin, TimestampMixin, same_dataset_fk, str_enum

_CLASSIFICATION_TYPES_SQL = ", ".join(f"'{t.value}'" for t in CLASSIFICATION_NODE_TYPES)


class IcdNode(IdMixin, TimestampMixin, Base):
    """An ICD record: a node in the Tabular List hierarchy (chapter, block, category, code...).

    `code` is free text: ICD-10-CA extends codes to 5th/6th characters, so no fixed length is
    assumed. Chapters/blocks typically carry a range (code_from/code_to) instead of a code.
    `normalized_code` is the punctuation-free, upper-case form used for exact lookup.
    chapter_id/block_id/category_id are denormalised ancestor shortcuts (same dataset) so
    filtering by chapter or block needs no recursive query.
    Retired codes are kept with status=disabled rather than deleted.
    """

    __tablename__ = "icd_nodes"
    __table_args__ = (
        # Target of same-dataset composite FKs from every other knowledge table.
        UniqueConstraint("dataset_id", "id", name="uq_icd_nodes_dataset_id_id"),
        same_dataset_fk("parent_id", "icd_nodes"),
        same_dataset_fk("chapter_id", "icd_nodes"),
        same_dataset_fk("block_id", "icd_nodes"),
        same_dataset_fk("category_id", "icd_nodes"),
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
        # ...and so does its normalized form ("A00.0" and "A000" cannot both exist).
        Index(
            "uq_icd_nodes_dataset_classification_normalized_code",
            "dataset_id",
            "normalized_code",
            unique=True,
            postgresql_where=text(
                f"normalized_code IS NOT NULL AND node_type IN ({_CLASSIFICATION_TYPES_SQL})"
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
        Index("ix_icd_nodes_dataset_id_normalized_code", "dataset_id", "normalized_code"),
        Index("ix_icd_nodes_dataset_id_node_type", "dataset_id", "node_type"),
        Index(
            "ix_icd_nodes_title_trgm",
            "title",
            postgresql_using="gin",
            postgresql_ops={"title": "gin_trgm_ops"},
        ),
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
    chapter_id: Mapped[int | None] = mapped_column(BigInteger, index=True)
    block_id: Mapped[int | None] = mapped_column(BigInteger, index=True)
    category_id: Mapped[int | None] = mapped_column(BigInteger, index=True)
    node_type: Mapped[NodeType] = mapped_column(
        str_enum(NodeType, "node_type"), nullable=False, index=True
    )
    code: Mapped[str | None] = mapped_column(String(32), index=True)
    normalized_code: Mapped[str | None] = mapped_column(String(32))
    code_from: Mapped[str | None] = mapped_column(String(32))
    code_to: Mapped[str | None] = mapped_column(String(32))
    title: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    depth: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    sort_order: Mapped[int | None] = mapped_column(Integer)
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
    # Format-specific provenance, e.g. {"kind": "pdf", "page": 12} or {"kind": "csv", "row": 7}.
    source_locator: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    raw_text: Mapped[str | None] = mapped_column(Text)
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict, server_default="{}"
    )
