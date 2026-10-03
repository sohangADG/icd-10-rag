import logging
from typing import Annotated

from fastapi import APIRouter, Depends, Response, status
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.core.database import get_session
from app.repositories.health_repository import HealthRepository
from app.schemas.health import DatabaseHealthResponse, HealthResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/health", tags=["health"])


@router.get("", response_model=HealthResponse)
async def health(settings: Annotated[Settings, Depends(get_settings)]) -> HealthResponse:
    """Liveness: the process is up. Does not touch the database."""
    return HealthResponse(status="ok", service=settings.app_name)


@router.get(
    "/db",
    response_model=DatabaseHealthResponse,
    responses={503: {"model": DatabaseHealthResponse}},
)
async def database_health(
    session: Annotated[AsyncSession, Depends(get_session)], response: Response
) -> DatabaseHealthResponse:
    """Readiness: PostgreSQL reachable and the pgvector extension installed."""
    repository = HealthRepository(session)
    try:
        await repository.ping()
    except (SQLAlchemyError, OSError) as exc:
        # Log the exception class only: driver messages can include host/user details.
        logger.error("database health check failed", extra={"error_type": type(exc).__name__})
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return DatabaseHealthResponse(status="error", database="unavailable", pgvector=False)

    pgvector_installed = await repository.pgvector_version() is not None
    if not pgvector_installed:
        logger.error("pgvector extension is not installed; run `alembic upgrade head`")
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return DatabaseHealthResponse(
        status="ok" if pgvector_installed else "error",
        database="connected",
        pgvector=pgvector_installed,
    )
