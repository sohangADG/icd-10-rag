from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.core.constants import RuleType
from app.models.base import Base, CreatedAtMixin, IdMixin, same_dataset_fk, str_enum


class IcdRule(IdMixin, CreatedAtMixin, Base):
    """A structured coding instruction (includes/excludes/notes/code first/see...).

    node_id is nullable and may point at any hierarchy level (chapter, block, category...),
    since instructions are not restricted to final codes. target_code preserves the code
    exactly as printed, and is only set when the source states exactly one code; ambiguous
    references keep target_code NULL and list what was seen in metadata["referenced_codes"].
    """

    __tablename__ = "icd_rules"
    __table_args__ = (
        UniqueConstraint("dataset_id", "id", name="uq_icd_rules_dataset_id_id"),
        same_dataset_fk("node_id", "icd_nodes", ondelete="CASCADE"),
        same_dataset_fk("target_node_id", "icd_nodes"),
        CheckConstraint("source_page > 0", name="source_page_positive"),
    )

    dataset_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("icd_datasets.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    node_id: Mapped[int | None] = mapped_column(BigInteger, index=True)
    rule_type: Mapped[RuleType] = mapped_column(
        str_enum(RuleType, "rule_type"), nullable=False, index=True
    )
    rule_text: Mapped[str] = mapped_column(Text, nullable=False)
    # Only for EXCLUDE rules, and only when the source distinguishes kinds (e.g. Excludes1/2).
    exclusion_type: Mapped[str | None] = mapped_column(String(32))
    target_code: Mapped[str | None] = mapped_column(String(32), index=True)
    target_node_id: Mapped[int | None] = mapped_column(BigInteger, index=True)
    is_mandatory: Mapped[bool | None] = mapped_column(Boolean)
    scope: Mapped[str | None] = mapped_column(String(64))
    source_page: Mapped[int | None] = mapped_column(Integer)
    source_locator: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict, server_default="{}"
    )
