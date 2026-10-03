from sqlalchemy import BigInteger, CheckConstraint, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.core.constants import RelationshipType
from app.models.base import Base, CreatedAtMixin, IdMixin, same_dataset_fk, str_enum


class IcdRelationship(IdMixin, CreatedAtMixin, Base):
    """An explicit, typed edge between nodes of the same dataset."""

    __tablename__ = "icd_relationships"
    __table_args__ = (
        same_dataset_fk("source_node_id", "icd_nodes", ondelete="CASCADE"),
        same_dataset_fk("target_node_id", "icd_nodes"),
        CheckConstraint("target_node_id IS NOT NULL OR target_code IS NOT NULL", name="has_target"),
    )

    dataset_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("icd_datasets.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    source_node_id: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True)
    target_node_id: Mapped[int | None] = mapped_column(BigInteger, index=True)
    target_code: Mapped[str | None] = mapped_column(String(32))
    relationship_type: Mapped[RelationshipType] = mapped_column(
        str_enum(RelationshipType, "relationship_type"), nullable=False, index=True
    )
    sequence_order: Mapped[int | None] = mapped_column(Integer)
