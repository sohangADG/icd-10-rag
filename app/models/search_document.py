from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, IdMixin, TimestampMixin, same_dataset_fk

# Prefix of the per-(model, dimension) HNSW expression indexes the indexer creates at runtime.
# Alembic's autogenerate comparison ignores indexes with this prefix (see migrations/env.py).
HNSW_INDEX_PREFIX = "ix_icd_search_documents_hnsw_"


class IcdSearchDocument(IdMixin, TimestampMixin, Base):
    """A retrieval unit derived from authoritative relational data: one per logical ICD entity.

    Retrieval aid only: a hit here never establishes a valid code; codes are always re-verified
    against icd_nodes.

    `search_vector` is a weighted tsvector (code/title/terms > inclusions > context). Exclusion
    text is deliberately left out of it so a query can never match a code via what it excludes.

    `embedding` stays a dimensionless pgvector column: the embedding model is configurable, and
    each vector is labelled with its model and dimension. ANN search uses per-model partial HNSW
    expression indexes over `embedding::vector(<dim>)` created by the indexer.
    `embedding_content_hash` hashes the embedding space (provider, model, dimension,
    normalisation) together with `semantic_text`, so unchanged documents are never re-embedded
    and any change of text or space triggers regeneration.
    """

    __tablename__ = "icd_search_documents"
    __table_args__ = (
        same_dataset_fk("node_id", "icd_nodes", ondelete="CASCADE"),
        UniqueConstraint(
            "dataset_id",
            "node_id",
            "document_type",
            name="uq_icd_search_documents_dataset_node_type",
        ),
        Index(
            "ix_icd_search_documents_search_vector",
            "search_vector",
            postgresql_using="gin",
        ),
        CheckConstraint(
            "embedding IS NULL OR embedding_model IS NOT NULL", name="embedding_has_model"
        ),
        CheckConstraint(
            "embedding IS NULL OR embedding_dimension IS NOT NULL", name="embedding_has_dimension"
        ),
        CheckConstraint(
            "embedding IS NULL OR (embedding_provider IS NOT NULL "
            "AND embedding_normalized IS NOT NULL)",
            name="embedding_has_space",
        ),
        # The database itself refuses a vector whose length differs from its declared dimension.
        CheckConstraint(
            "embedding IS NULL OR vector_dims(embedding) = embedding_dimension",
            name="embedding_dimension_matches",
        ),
    )

    dataset_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("icd_datasets.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    node_id: Mapped[int | None] = mapped_column(BigInteger, index=True)
    document_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str | None] = mapped_column(String(64))
    # Exactly the text that is embedded: positive retrieval evidence only (no exclusions, no
    # instructions naming other conditions, no subdivision list). See app/indexing/indexer.py.
    semantic_text: Mapped[str | None] = mapped_column(Text)
    search_vector: Mapped[str | None] = mapped_column(TSVECTOR)
    # "metadata" is reserved on declarative classes, so the attribute is named differently.
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict, server_default="{}"
    )
    embedding: Mapped[list[float] | None] = mapped_column(Vector())
    embedding_provider: Mapped[str | None] = mapped_column(String(64))
    embedding_model: Mapped[str | None] = mapped_column(String(128), index=True)
    embedding_normalized: Mapped[bool | None] = mapped_column(Boolean)
    embedding_dimension: Mapped[int | None] = mapped_column(Integer)
    embedding_content_hash: Mapped[str | None] = mapped_column(String(64))
    embedded_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
