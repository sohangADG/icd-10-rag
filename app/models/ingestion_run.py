import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    Uuid,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.core.constants import IngestionRunStatus
from app.models.base import Base, CreatedAtMixin, IdMixin, str_enum


def _counter() -> Mapped[int]:
    return mapped_column(Integer, nullable=False, default=0, server_default=text("0"))


class IcdIngestionRun(IdMixin, CreatedAtMixin, Base):
    """One attempt to import a source document into a dataset."""

    __tablename__ = "icd_ingestion_runs"
    __table_args__ = (
        CheckConstraint(
            "records_seen >= 0 AND records_inserted >= 0 "
            "AND records_updated >= 0 AND records_failed >= 0",
            name="counters_non_negative",
        ),
        CheckConstraint(
            "completed_at IS NULL OR completed_at >= started_at", name="completed_after_start"
        ),
    )

    dataset_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("icd_datasets.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    run_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False, unique=True, default=uuid.uuid4)
    status: Mapped[IngestionRunStatus] = mapped_column(
        str_enum(IngestionRunStatus, "status"),
        nullable=False,
        default=IngestionRunStatus.PENDING,
        server_default=IngestionRunStatus.PENDING.value,
        index=True,
    )
    source_filename: Mapped[str] = mapped_column(Text, nullable=False)
    source_checksum: Mapped[str | None] = mapped_column(String(128))
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    records_seen: Mapped[int] = _counter()
    records_inserted: Mapped[int] = _counter()
    records_updated: Mapped[int] = _counter()
    records_failed: Mapped[int] = _counter()
    error_summary: Mapped[str | None] = mapped_column(Text)
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict, server_default="{}"
    )


class IcdIngestionError(IdMixin, CreatedAtMixin, Base):
    """A single record-level validation issue or failure within an ingestion run.

    `error_type` holds the validation issue code (e.g. DUPLICATE_CODE); `severity` is
    ERROR / WARNING / INFO. Raw payloads never contain source content beyond the offending
    record's identifiers.
    """

    __tablename__ = "icd_ingestion_errors"
    __table_args__ = (CheckConstraint("severity IN ('ERROR', 'WARNING', 'INFO')", name="severity"),)

    ingestion_run_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("icd_ingestion_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    severity: Mapped[str] = mapped_column(
        String(16), nullable=False, default="ERROR", server_default="ERROR"
    )
    source_page: Mapped[int | None] = mapped_column(Integer)
    entity_type: Mapped[str | None] = mapped_column(String(64))
    entity_identifier: Mapped[str | None] = mapped_column(Text)
    error_type: Mapped[str] = mapped_column(String(64), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    locator: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    raw_payload: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
