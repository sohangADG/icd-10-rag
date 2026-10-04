"""Protected administrative operations. Disabled (404) unless ADMIN_API_TOKEN is set.

There is deliberately no upload/import endpoint: ingesting a source is an operator action
performed with the CLI, where licence attestation is recorded (docs/licensing.md).
"""

from typing import Annotated, Any

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import ProviderHandle, embedding_provider, require_admin
from app.core.constants import DatasetStatus
from app.core.database import get_session
from app.core.exceptions import DatasetNotFound, DatasetNotReady
from app.ingestion.service import IngestionService
from app.repositories.dataset_repository import DatasetRepository

router = APIRouter(prefix="/api/v1/admin", tags=["admin"], dependencies=[Depends(require_admin)])


@router.post("/datasets/{dataset_id}/reindex")
async def reindex(
    dataset_id: int,
    session: Annotated[AsyncSession, Depends(get_session)],
    provider: Annotated[ProviderHandle, Depends(embedding_provider)],
) -> dict[str, Any]:
    result = await IngestionService(session, provider.provider).reindex(dataset_id)
    return {"dataset_id": dataset_id, **result}


@router.post("/datasets/{dataset_id}/archive")
async def archive(
    dataset_id: int, session: Annotated[AsyncSession, Depends(get_session)]
) -> dict[str, Any]:
    repository = DatasetRepository(session)
    dataset = await repository.get(dataset_id)
    if dataset is None:
        raise DatasetNotFound(f"Dataset {dataset_id} does not exist")
    if dataset.status != DatasetStatus.READY:
        raise DatasetNotReady(f"Only READY datasets can be archived (is {dataset.status.value})")
    await repository.set_status(dataset_id, DatasetStatus.ARCHIVED)
    await session.commit()
    return {"dataset_id": dataset_id, "status": DatasetStatus.ARCHIVED.value}
