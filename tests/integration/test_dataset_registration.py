import pytest
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.constants import ICD10CA_2022, DatasetStatus
from app.models import IcdDataset
from app.schemas.dataset import DatasetCreate
from app.services.dataset_service import DatasetService
from tests.integration.factories import make_dataset


async def _dataset_count(session: AsyncSession) -> int:
    return (await session.execute(select(func.count()).select_from(IcdDataset))).scalar_one()


async def test_icd10ca_2022_registration_is_idempotent(session: AsyncSession) -> None:
    service = DatasetService(session)
    data = DatasetCreate(**ICD10CA_2022)

    first = await service.register(data)
    second = await service.register(data)

    assert first.created is True
    assert second.created is False
    assert second.dataset.id == first.dataset.id
    assert await _dataset_count(session) == 1
    assert first.dataset.status == DatasetStatus.PENDING
    assert first.dataset.system == "ICD-10-CA"
    assert first.dataset.version == "2022"
    assert first.dataset.source_checksum is None


async def test_other_versions_and_systems_register_separately(session: AsyncSession) -> None:
    service = DatasetService(session)

    await service.register(DatasetCreate(**ICD10CA_2022))
    await service.register(DatasetCreate(**{**ICD10CA_2022, "version": "2026"}))
    await service.register(DatasetCreate(**{**ICD10CA_2022, "language": "fr"}))
    await service.register(
        DatasetCreate(**{**ICD10CA_2022, "system": "ICD-10-CM", "country": "US", "version": "2025"})
    )

    assert await _dataset_count(session) == 4


async def test_duplicate_dataset_identity_is_rejected_by_database(session: AsyncSession) -> None:
    await make_dataset(session, system="ICD-10-CA", country="CA", version="2022")

    # revision/edition are NULL on both rows: NULLS NOT DISTINCT must still treat them as equal.
    with pytest.raises(IntegrityError, match="uq_icd_datasets_identity"):
        async with session.begin_nested():
            await make_dataset(session, system="ICD-10-CA", country="CA", version="2022")


async def test_distinct_revision_is_a_distinct_identity(session: AsyncSession) -> None:
    await make_dataset(session, version="2022")
    await make_dataset(session, version="2022", revision="r2")

    assert await _dataset_count(session) == 2


async def test_unknown_dataset_status_is_rejected(session: AsyncSession) -> None:
    with pytest.raises(IntegrityError, match="ck_icd_datasets_status"):
        async with session.begin_nested():
            await session.execute(
                text(
                    "INSERT INTO icd_datasets "
                    "(system, country, version, publisher, language, status) "
                    "VALUES ('TEST-ICD', 'XX', '0000', 'Test', 'en', 'bogus')"
                )
            )
