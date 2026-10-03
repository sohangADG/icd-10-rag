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
from sqlalchemy.orm import Mapped, mapped_column

from app.core.constants import RuleType
from app.models.base import Base, CreatedAtMixin, IdMixin, same_dataset_fk, str_enum


class IcdRule(IdMixin, CreatedAtMixin, Base):
    """A structured coding instruction (includes/excludes/notes/dagger-asterisk...).

    node_id is nullable and may point at any hierarchy level (chapter, block, category...),
    since instructions are not restricted to final codes. target_code preserves the code
    exactly as printed even when it cannot (yet) be resolved to target_node_id.
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
    target_code: Mapped[str | None] = mapped_column(String(32), index=True)
    target_node_id: Mapped[int | None] = mapped_column(BigInteger, index=True)
    is_mandatory: Mapped[bool | None] = mapped_column(Boolean)
    scope: Mapped[str | None] = mapped_column(String(64))
    source_page: Mapped[int] = mapped_column(Integer, nullable=False)
