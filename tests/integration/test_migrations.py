import asyncio
from typing import Any

from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import inspect, pool, text
from sqlalchemy.engine import URL, Connection
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from app.core.config import get_settings
from app.models import Base
from tests.integration.db import alembic_config, create_fresh_database, drop_database

PHASE1_TABLES = {
    "icd_datasets",
    "icd_nodes",
    "icd_terms",
    "icd_rules",
    "icd_relationships",
    "icd_index_entries",
    "icd_source_refs",
    "icd_search_documents",
    "icd_ingestion_runs",
    "icd_ingestion_errors",
}


async def _table_names(engine: AsyncEngine) -> set[str]:
    async with engine.connect() as connection:
        return set(await connection.run_sync(lambda c: inspect(c).get_table_names()))


async def test_database_is_at_alembic_head(engine: AsyncEngine, database_url: URL) -> None:
    head = ScriptDirectory.from_config(alembic_config(database_url)).get_current_head()

    async with engine.connect() as connection:
        current = (
            await connection.execute(text("SELECT version_num FROM alembic_version"))
        ).scalar_one()

    assert current == head


async def test_all_phase1_tables_exist(engine: AsyncEngine) -> None:
    assert await _table_names(engine) >= PHASE1_TABLES


async def test_migrations_match_orm_models(engine: AsyncEngine) -> None:
    """Autogenerate against the migrated DB must find nothing: models and migrations agree."""

    def diff(connection: Connection) -> list[Any]:
        context = MigrationContext.configure(connection, opts={"compare_type": True})
        return compare_metadata(context, Base.metadata)

    async with engine.connect() as connection:
        assert await connection.run_sync(diff) == []


def test_downgrade_to_base_and_upgrade_again() -> None:
    # Synchronous test: Alembic's env.py runs its own event loop via asyncio.run().
    name = f"{get_settings().database_name}_test_migrations"
    url = create_fresh_database(name)
    config = alembic_config(url)

    async def table_names() -> set[str]:
        engine = create_async_engine(url, poolclass=pool.NullPool)
        try:
            return await _table_names(engine)
        finally:
            await engine.dispose()

    try:
        command.upgrade(config, "head")
        command.downgrade(config, "base")
        assert not PHASE1_TABLES & asyncio.run(table_names())

        command.upgrade(config, "head")
        assert asyncio.run(table_names()) >= PHASE1_TABLES
    finally:
        drop_database(name)
