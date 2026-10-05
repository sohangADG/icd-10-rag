"""Resolve a request's (coding system, version, country, language) or dataset id to exactly one
READY dataset — or fail with a precise, mappable error. Datasets are never mixed."""

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.constants import DatasetStatus
from app.core.exceptions import (
    AmbiguousDatasetError,
    DatasetNotFound,
    DatasetNotReady,
    UnsupportedCodingSystem,
    UnsupportedDatasetVersion,
)
from app.models import IcdDataset
from app.repositories.dataset_repository import DatasetRepository


class DatasetResolver:
    def __init__(self, session: AsyncSession) -> None:
        self._datasets = DatasetRepository(session)

    async def resolve(
        self,
        *,
        dataset_id: int | None = None,
        coding_system: str | None = None,
        version: str | None = None,
        country: str | None = None,
        language: str | None = None,
        require_ready: bool = True,
    ) -> IcdDataset:
        if dataset_id is not None:
            dataset = await self._datasets.get(dataset_id)
            if dataset is None:
                raise DatasetNotFound(f"Dataset {dataset_id} does not exist")
            if coding_system and dataset.system.upper() != coding_system.upper():
                raise DatasetNotFound(
                    f"Dataset {dataset_id} is {dataset.system}, not {coding_system}"
                )
            if version and dataset.version != version:
                raise DatasetNotFound(f"Dataset {dataset_id} is version {dataset.version}")
            return self._ready(dataset) if require_ready else dataset

        if not coding_system:
            raise DatasetNotFound("Either dataset_id or coding_system is required")
        if coding_system.upper() not in await self._datasets.systems():
            raise UnsupportedCodingSystem(
                f"No dataset is registered for coding system {coding_system}",
                details={"coding_system": coding_system},
            )
        candidates = await self._datasets.list_datasets(
            system=coding_system, version=version, country=country, language=language
        )
        if not candidates:
            raise UnsupportedDatasetVersion(
                f"No {coding_system} dataset matches version={version!r}, country={country!r}, "
                f"language={language!r}",
                details={"coding_system": coding_system, "version": version},
            )
        ready = [d for d in candidates if d.status == DatasetStatus.READY]
        if require_ready and not ready:
            raise DatasetNotReady(
                f"{coding_system} {version or ''} exists but is not ready",
                details={"statuses": sorted({d.status.value for d in candidates})},
            )
        pool = ready if require_ready else candidates
        if len(pool) > 1:
            raise AmbiguousDatasetError(
                "Several datasets match; specify version, country, language or dataset_id",
                details={
                    "datasets": [
                        {
                            "id": d.id,
                            "version": d.version,
                            "country": d.country,
                            "language": d.language,
                        }
                        for d in pool
                    ]
                },
            )
        return pool[0]

    @staticmethod
    def _ready(dataset: IcdDataset) -> IcdDataset:
        if dataset.status != DatasetStatus.READY:
            raise DatasetNotReady(
                f"Dataset {dataset.id} is {dataset.status.value}, not ready",
                details={"dataset_id": dataset.id, "status": dataset.status.value},
            )
        return dataset
