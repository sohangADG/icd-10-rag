from sqlalchemy import BigInteger, CheckConstraint, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.core.constants import TermType
from app.models.base import Base, CreatedAtMixin, IdMixin, same_dataset_fk, str_enum


class IcdTerm(IdMixin, CreatedAtMixin, Base):
    """A term attached to a Tabular List node (official title, inclusion term, synonym...)."""

    __tablename__ = "icd_terms"
    __table_args__ = (
        same_dataset_fk("node_id", "icd_nodes", ondelete="CASCADE"),
        Index("ix_icd_terms_dataset_id_normalized_term", "dataset_id", "normalized_term"),
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
    source_page: Mapped[int] = mapped_column(Integer, nullable=False)
