import logging

from sqlalchemy.ext.asyncio import AsyncSession

from app.repositories.dataset_repository import DatasetRepository
from app.schemas.dataset import DatasetCreate, DatasetRead, DatasetRegistration

logger = logging.getLogger(__name__)


class DatasetService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._datasets = DatasetRepository(session)

    async def register(self, data: DatasetCreate) -> DatasetRegistration:
        """Idempotently register a dataset identity. Re-registering returns the existing row.

        Only identity/metadata is stored; an existing row is never modified here.
        """
        new_id = await self._datasets.insert_if_absent(data)
        dataset = await self._datasets.get_by_identity(data)
        if dataset is None:
            raise RuntimeError("Dataset missing immediately after idempotent insert")
        # Serialize before commit: a commit may expire ORM attributes, and async sessions
        # cannot lazy-load them afterwards.
        registration = DatasetRegistration(
            dataset=DatasetRead.model_validate(dataset), created=new_id is not None
        )
        await self._session.commit()

        logger.info(
            "dataset registered" if registration.created else "dataset already registered",
            extra={
                "dataset_id": registration.dataset.id,
                "system": registration.dataset.system,
                "country": registration.dataset.country,
                "version": registration.dataset.version,
                "language": registration.dataset.language,
            },
        )
        return registration
