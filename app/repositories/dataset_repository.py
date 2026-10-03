from sqlalchemy import and_, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.dataset import DATASET_IDENTITY_COLUMNS, IcdDataset
from app.schemas.dataset import DatasetCreate


class DatasetRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_by_identity(self, data: DatasetCreate) -> IcdDataset | None:
        conditions = [
            # IS NOT DISTINCT FROM matches NULL revision/edition the same way the unique key does.
            getattr(IcdDataset, column).is_not_distinct_from(getattr(data, column))
            for column in DATASET_IDENTITY_COLUMNS
        ]
        result = await self._session.execute(select(IcdDataset).where(and_(*conditions)))
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
