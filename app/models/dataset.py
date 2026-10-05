from datetime import date, datetime
from typing import Any

from sqlalchemy import (
    CheckConstraint,
    Date,
    DateTime,
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

from app.core.constants import DatasetStatus, SourceType
from app.models.base import Base, IdMixin, TimestampMixin, str_enum

DATASET_IDENTITY_COLUMNS = ("system", "country", "version", "revision", "edition", "language")


class IcdDataset(IdMixin, TimestampMixin, Base):
    """One ICD classification release (system + country + version + revision/edition + language).

    Every ICD knowledge row carries a dataset_id, so releases coexist without ever sharing an
    unversioned namespace. `system` is the coding system (e.g. ICD-10-CA); `modification`
    optionally names the national modification separately from the system label.
    """

    __tablename__ = "icd_datasets"
    __table_args__ = (
        # NULLS NOT DISTINCT: (ICD-10-CA, CA, 2022, NULL, NULL, en) may exist only once.
        UniqueConstraint(
            *DATASET_IDENTITY_COLUMNS,
            name="uq_icd_datasets_identity",
            postgresql_nulls_not_distinct=True,
        ),
        # One source file (by SHA-256) can back at most one dataset identity.
        Index(
            "uq_icd_datasets_source_checksum",
            "source_checksum",
            unique=True,
            postgresql_where=text("source_checksum IS NOT NULL"),
        ),
        CheckConstraint(
            "effective_to IS NULL OR effective_from IS NULL OR effective_to >= effective_from",
            name="effective_range",
        ),
        CheckConstraint("publication_year IS NULL OR publication_year > 1900", name="pub_year"),
        CheckConstraint(
            "source_page_count IS NULL OR source_page_count >= 0", name="page_count_non_negative"
        ),
    )

    system: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    modification: Mapped[str | None] = mapped_column(String(64))
    country: Mapped[str] = mapped_column(String(8), nullable=False)
    version: Mapped[str] = mapped_column(String(32), nullable=False)
    revision: Mapped[str | None] = mapped_column(String(32))
    edition: Mapped[str | None] = mapped_column(String(64))
    title: Mapped[str | None] = mapped_column(Text)
    publisher: Mapped[str] = mapped_column(String(255), nullable=False)
    publication_year: Mapped[int | None] = mapped_column(SmallInteger)
    language: Mapped[str] = mapped_column(String(16), nullable=False)
    source_type: Mapped[SourceType | None] = mapped_column(str_enum(SourceType, "source_type"))
    source_filename: Mapped[str | None] = mapped_column(Text)
    source_identifier: Mapped[str | None] = mapped_column(Text)
    source_uri: Mapped[str | None] = mapped_column(Text)
    # SHA-256 (hex) of the source file. Named "checksum" since Phase 1.
    source_checksum: Mapped[str | None] = mapped_column(String(128))
    source_page_count: Mapped[int | None] = mapped_column(Integer)
    status: Mapped[DatasetStatus] = mapped_column(
        str_enum(DatasetStatus, "status"),
        nullable=False,
        default=DatasetStatus.PENDING,
        server_default=DatasetStatus.PENDING.value,
        index=True,
    )
    imported_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    effective_from: Mapped[date | None] = mapped_column(Date)
    effective_to: Mapped[date | None] = mapped_column(Date)
    # "metadata" is reserved on declarative classes, so the attribute is named differently.
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict, server_default="{}"
    )
