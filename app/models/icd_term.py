from typing import Any

from sqlalchemy import BigInteger, CheckConstraint, ForeignKey, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.core.constants import TermType
from app.models.base import Base, CreatedAtMixin, IdMixin, same_dataset_fk, str_enum


class IcdTerm(IdMixin, CreatedAtMixin, Base):
    """A term attached to a Tabular List node (inclusion term, source synonym, abbreviation...).

    Only source-provided terms are stored: ingestion never generates synonyms.
    """

    __tablename__ = "icd_terms"
    __table_args__ = (
        same_dataset_fk("node_id", "icd_nodes", ondelete="CASCADE"),
        Index("ix_icd_terms_dataset_id_normalized_term", "dataset_id", "normalized_term"),
        Index(
            "ix_icd_terms_normalized_term_trgm",
            "normalized_term",
            postgresql_using="gin",
            postgresql_ops={"normalized_term": "gin_trgm_ops"},
        ),
        CheckConstraint("source_page > 0", name="source_page_positive"),
    )

    dataset_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("icd_datasets.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    node_id: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True)
    term: Mapped[str] = mapped_column(Text, nullable=False)
    normalized_term: Mapped[str] = mapped_column(Text, nullable=False)
    term_type: Mapped[TermType] = mapped_column(
        str_enum(TermType, "term_type"), nullable=False, index=True
    )
    language: Mapped[str] = mapped_column(String(16), nullable=False)
    # Nullable since non-paginated sources (CSV, XML...) have no page; see source_locator.
    source_page: Mapped[int | None] = mapped_column(Integer)
    source_locator: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict, server_default="{}"
    )
