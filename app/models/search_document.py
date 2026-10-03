from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import BigInteger, CheckConstraint, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, IdMixin, TimestampMixin, same_dataset_fk


class IcdSearchDocument(IdMixin, TimestampMixin, Base):
    """A retrieval record (logical chunk) derived from authoritative relational data.

    Retrieval aid only: a hit here never establishes a valid code; codes are always re-verified
    against icd_nodes.

    `embedding` is a dimensionless pgvector column because no embedding model has been chosen.
    Once one is, a migration will ALTER it to vector(<dim>) and add an ANN (HNSW) index, which
    requires a fixed dimension. `embedding_model` records which model produced each vector so
    vectors from different models are never compared.
    """

    __tablename__ = "icd_search_documents"
    __table_args__ = (
        same_dataset_fk("node_id", "icd_nodes", ondelete="CASCADE"),
        CheckConstraint(
            "embedding IS NULL OR embedding_model IS NOT NULL", name="embedding_has_model"
        ),
    )

    dataset_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("icd_datasets.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    node_id: Mapped[int | None] = mapped_column(BigInteger, index=True)
    document_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    # "metadata" is reserved on declarative classes, so the attribute is named differently.
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict, server_default="{}"
    )
    embedding: Mapped[list[float] | None] = mapped_column(Vector())
    embedding_model: Mapped[str | None] = mapped_column(String(128))
