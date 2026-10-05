from typing import Any

from sqlalchemy import BigInteger, CheckConstraint, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, CreatedAtMixin, IdMixin, same_dataset_fk


class IcdSourceRef(IdMixin, CreatedAtMixin, Base):
    """Provenance: where in which source document a node, rule or index entry came from.

    pdf_page/printed_page are filled for paginated sources; `locator` carries the format-specific
    position for every source type (CSV row, XML element path, XLSX sheet+row...).
    """

    __tablename__ = "icd_source_refs"
    __table_args__ = (
        same_dataset_fk("node_id", "icd_nodes", ondelete="CASCADE"),
        same_dataset_fk("rule_id", "icd_rules", ondelete="CASCADE"),
        same_dataset_fk("index_entry_id", "icd_index_entries", ondelete="CASCADE"),
        CheckConstraint("pdf_page > 0", name="pdf_page_positive"),
    )

    dataset_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("icd_datasets.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    node_id: Mapped[int | None] = mapped_column(BigInteger, index=True)
    rule_id: Mapped[int | None] = mapped_column(BigInteger, index=True)
    index_entry_id: Mapped[int | None] = mapped_column(BigInteger, index=True)
    source_filename: Mapped[str] = mapped_column(Text, nullable=False)
    pdf_page: Mapped[int | None] = mapped_column(Integer)
    printed_page: Mapped[str | None] = mapped_column(String(32))
    section: Mapped[str | None] = mapped_column(Text)
    raw_text: Mapped[str | None] = mapped_column(Text)
    locator: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
