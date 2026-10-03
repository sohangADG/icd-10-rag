from datetime import datetime
from enum import StrEnum

from sqlalchemy import BigInteger, DateTime, Enum, ForeignKeyConstraint, MetaData, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# Deterministic constraint names so Alembic migrations are stable and reviewable.
NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_N_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


class IdMixin:
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)


class CreatedAtMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class TimestampMixin(CreatedAtMixin):
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


def str_enum(enum_cls: type[StrEnum], name: str) -> Enum:
    """Persist a StrEnum as VARCHAR guarded by a named CHECK constraint (no native PG enum)."""
    return Enum(
        enum_cls,
        name=name,
        native_enum=False,
        create_constraint=True,
        length=32,
        values_callable=lambda members: [member.value for member in members],
        validate_strings=True,
    )


def same_dataset_fk(
    local_column: str, target_table: str, *, ondelete: str | None = None
) -> ForeignKeyConstraint:
    """FK on (dataset_id, <local_column>) -> <target_table>(dataset_id, id).

    Makes it impossible for a row to reference a node/rule/index entry from another dataset
    (e.g. an ICD-10-CA 2022 node pointing at an ICD-10-CM node). With MATCH SIMPLE semantics the
    constraint is skipped while the referencing column is NULL, so optional links stay optional.
    """
    return ForeignKeyConstraint(
        ["dataset_id", local_column],
        [f"{target_table}.dataset_id", f"{target_table}.id"],
        ondelete=ondelete,
    )
