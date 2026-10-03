from datetime import date

from sqlalchemy import CheckConstraint, Date, SmallInteger, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.core.constants import DatasetStatus
from app.models.base import Base, IdMixin, TimestampMixin, str_enum

DATASET_IDENTITY_COLUMNS = ("system", "country", "version", "revision", "edition", "language")


class IcdDataset(IdMixin, TimestampMixin, Base):
    """One ICD classification release (system + country + version + revision/edition + language).

    Every ICD knowledge row carries a dataset_id, so releases coexist without ever sharing an
    unversioned namespace.
    """

    __tablename__ = "icd_datasets"
    __table_args__ = (
        # NULLS NOT DISTINCT: (ICD-10-CA, CA, 2022, NULL, NULL, en) may exist only once.
        UniqueConstraint(
            *DATASET_IDENTITY_COLUMNS,
            name="uq_icd_datasets_identity",
            postgresql_nulls_not_distinct=True,
        ),
        CheckConstraint(
            "effective_to IS NULL OR effective_from IS NULL OR effective_to >= effective_from",
            name="effective_range",
        ),
        CheckConstraint("publication_year IS NULL OR publication_year > 1900", name="pub_year"),
    )

    system: Mapped[str] = mapped_column(String(64), nullable=False)
    country: Mapped[str] = mapped_column(String(8), nullable=False)
    version: Mapped[str] = mapped_column(String(32), nullable=False)
    revision: Mapped[str | None] = mapped_column(String(32))
    edition: Mapped[str | None] = mapped_column(String(64))
    publisher: Mapped[str] = mapped_column(String(255), nullable=False)
    publication_year: Mapped[int | None] = mapped_column(SmallInteger)
    language: Mapped[str] = mapped_column(String(16), nullable=False)
    source_filename: Mapped[str | None] = mapped_column(Text)
    source_checksum: Mapped[str | None] = mapped_column(String(128))
    status: Mapped[DatasetStatus] = mapped_column(
        str_enum(DatasetStatus, "status"),
        nullable=False,
        default=DatasetStatus.PENDING,
        server_default=DatasetStatus.PENDING.value,
    )
    effective_from: Mapped[date | None] = mapped_column(Date)
    effective_to: Mapped[date | None] = mapped_column(Date)
