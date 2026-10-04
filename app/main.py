import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app.api.errors import register_exception_handlers
from app.api.health import router as health_router
from app.api.middleware import RequestContextMiddleware
from app.api.v1.admin import router as admin_router
from app.api.v1.icd import router as icd_router
from app.core.config import get_settings
from app.core.database import dispose_engine, get_engine
from app.core.logging import configure_logging

logger = logging.getLogger(__name__)


async def verify_database() -> None:
    """Startup connectivity probe. Failures are logged, not fatal: /health/db reports readiness."""
    try:
        async with get_engine().connect() as connection:
            await connection.execute(text("SELECT 1"))
    except (SQLAlchemyError, OSError) as exc:
        # Exception class only: driver messages may include connection details.
        logger.error(
            "database unreachable at startup; /health/db will report unavailable",
            extra={"error_type": type(exc).__name__},
        )
    else:
        logger.info("database connection verified")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    logger.info("starting service", extra={"service": settings.app_name, "env": settings.app_env})
    await verify_database()
    yield
    await dispose_engine()
    logger.info("service stopped")


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)
    app = FastAPI(title=settings.app_name, version="0.2.0", lifespan=lifespan)
    app.add_middleware(RequestContextMiddleware)
    register_exception_handlers(app)
    app.include_router(health_router)
    app.include_router(icd_router)
    app.include_router(admin_router)
    return app


app = create_app()
