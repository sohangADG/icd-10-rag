from collections.abc import Sequence
from typing import Any

from sqlalchemy import and_, delete, func, select, text, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.constants import DatasetStatus
from app.models import (
    IcdIndexEntry,
    IcdNode,
    IcdRelationship,
    IcdRule,
    IcdSearchDocument,
    IcdSourceRef,
    IcdTerm,
)
from app.models.dataset import DATASET_IDENTITY_COLUMNS, IcdDataset
from app.schemas.dataset import DatasetCreate

# Tables holding a dataset's knowledge content, in safe deletion order (children first).
_CONTENT_TABLES = (
    IcdSearchDocument,
    IcdSourceRef,
    IcdRelationship,
    IcdIndexEntry,
    IcdRule,
    IcdTerm,
)


class DatasetRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # --- identity --------------------------------------------------------------------------

    async def get_by_identity(self, data: DatasetCreate) -> IcdDataset | None:
        return await self.get_by_identity_values(
            {column: getattr(data, column) for column in DATASET_IDENTITY_COLUMNS}
        )

    async def get_by_identity_values(self, identity: dict[str, Any]) -> IcdDataset | None:
        conditions = [
            # IS NOT DISTINCT FROM matches NULL revision/edition the same way the unique key does.
            getattr(IcdDataset, column).is_not_distinct_from(identity.get(column))
            for column in DATASET_IDENTITY_COLUMNS
        ]
        result = await self._session.execute(select(IcdDataset).where(and_(*conditions)))
        return result.scalar_one_or_none()

    async def get_by_checksum(self, checksum: str) -> IcdDataset | None:
        result = await self._session.execute(
            select(IcdDataset).where(IcdDataset.source_checksum == checksum)
        )
        return result.scalar_one_or_none()

    async def insert_if_absent(self, data: DatasetCreate) -> int | None:
        """Insert the dataset unless its identity already exists. Returns the new id, or None.

        Relies on the database unique constraint (ON CONFLICT DO NOTHING), so concurrent
        registrations cannot create duplicates.
        """
        statement = (
            insert(IcdDataset)
            .values(**data.model_dump())
            .on_conflict_do_nothing(constraint="uq_icd_datasets_identity")
            .returning(IcdDataset.id)
        )
        result = await self._session.execute(statement)
        return result.scalar_one_or_none()

    async def lock_identity(self, identity_key: str) -> None:
        """Transaction-scoped advisory lock serialising imports of one dataset identity."""
        await self._session.execute(
            text("SELECT pg_advisory_xact_lock(hashtextextended(:key, 0))"),
            {"key": f"icd-dataset:{identity_key}"},
        )

    # --- reads -----------------------------------------------------------------------------

    async def get(self, dataset_id: int) -> IcdDataset | None:
        return await self._session.get(IcdDataset, dataset_id)

    async def list_datasets(
        self,
        *,
        system: str | None = None,
        version: str | None = None,
        country: str | None = None,
        language: str | None = None,
        statuses: Sequence[DatasetStatus] | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[IcdDataset]:
        statement = select(IcdDataset)
        if system:
            statement = statement.where(func.upper(IcdDataset.system) == system.upper())
        if version:
            statement = statement.where(IcdDataset.version == version)
        if country:
            statement = statement.where(func.upper(IcdDataset.country) == country.upper())
        if language:
            statement = statement.where(IcdDataset.language == language)
        if statuses:
            statement = statement.where(IcdDataset.status.in_(list(statuses)))
        statement = (
            statement.order_by(IcdDataset.system, IcdDataset.version.desc(), IcdDataset.id)
            .limit(limit)
            .offset(offset)
        )
        return list((await self._session.execute(statement)).scalars())

    async def systems(self) -> list[str]:
        result = await self._session.execute(select(IcdDataset.system).distinct())
        return [row.upper() for row in result.scalars()]

    # --- writes ----------------------------------------------------------------------------

    async def set_status(self, dataset_id: int, status: DatasetStatus, **values: Any) -> None:
        await self._session.execute(
            update(IcdDataset)
            .where(IcdDataset.id == dataset_id)
            .values(status=status, updated_at=func.now(), **values)
        )

    async def delete_content(self, dataset_id: int) -> None:
        """Remove every knowledge row of a dataset (used before a retry re-import).

        icd_nodes references itself (parent/chapter/block/category), so the shortcut columns
        are cleared first and nodes are then deleted in one statement.
        """
        for model in _CONTENT_TABLES:
            await self._session.execute(delete(model).where(model.dataset_id == dataset_id))
        await self._session.execute(
            update(IcdNode)
            .where(IcdNode.dataset_id == dataset_id)
            .values(parent_id=None, chapter_id=None, block_id=None, category_id=None)
        )
        await self._session.execute(delete(IcdNode).where(IcdNode.dataset_id == dataset_id))

    async def content_counts(self, dataset_id: int) -> dict[str, int]:
        counts: dict[str, int] = {}
        for model in (IcdNode, IcdTerm, IcdRule, IcdIndexEntry, IcdSourceRef, IcdSearchDocument):
            counts[model.__tablename__] = (
                await self._session.execute(
                    select(func.count()).select_from(model).where(model.dataset_id == dataset_id)
                )
            ).scalar_one()
        return counts
