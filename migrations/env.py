import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import URL, Connection
from sqlalchemy.ext.asyncio import create_async_engine

from app.core.config import get_settings
from app.models import Base
from app.models.search_document import HNSW_INDEX_PREFIX

config = context.config

if config.config_file_name is not None and config.attributes.get("configure_logger", True):
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def include_object(
    obj: object, name: str | None, type_: str, reflected: bool, compare_to: object
) -> bool:
    """Per-embedding-model HNSW indexes are created at runtime by the indexer (their dimension
    depends on configuration), so autogenerate must neither drop nor recreate them."""
    return not (type_ == "index" and name is not None and name.startswith(HNSW_INDEX_PREFIX))


def _database_url() -> URL:
    # Tests pass an explicit URL object via config.attributes; otherwise use app settings.
    # A URL object (not a string) is used so the password is never interpolated into config.
    url = config.attributes.get("database_url")
    return url if url is not None else get_settings().sqlalchemy_url()


def run_migrations_offline() -> None:
    # Offline mode only emits SQL; it needs the dialect, not a connection URL or credentials.
    context.configure(
        dialect_name="postgresql",
        target_metadata=target_metadata,
        include_object=include_object,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def _run_sync_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        include_object=include_object,
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    engine = create_async_engine(_database_url(), poolclass=pool.NullPool)
    try:
        async with engine.connect() as connection:
            await connection.run_sync(_run_sync_migrations)
    finally:
        await engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
