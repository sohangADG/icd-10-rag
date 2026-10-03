"""Integration fixtures: a dedicated, migration-built test database with per-test rollback.

The schema is created by running the real Alembic migrations (not metadata.create_all), so the
tests exercise exactly what production gets. Each test runs inside an outer transaction that is
rolled back afterwards; the session uses savepoints so code under test may call commit().
"""

from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest
from alembic import command
from sqlalchemy import pool
from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine

from app.core.config import get_settings
from tests.integration.db import alembic_config, create_fresh_database, drop_database

INTEGRATION_DIR = Path(__file__).parent


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if item.path.is_relative_to(INTEGRATION_DIR):
            item.add_marker(pytest.mark.integration)


@pytest.fixture(scope="session")
def database_url() -> Iterator[URL]:
    # Synchronous fixture: Alembic's env.py runs its own event loop via asyncio.run().
    name = f"{get_settings().database_name}_test"
    url = create_fresh_database(name)
    command.upgrade(alembic_config(url), "head")
    yield url
    drop_database(name)


@pytest.fixture
async def engine(database_url: URL) -> AsyncIterator[AsyncEngine]:
    engine = create_async_engine(database_url, poolclass=pool.NullPool)
    yield engine
    await engine.dispose()


@pytest.fixture
async def session(engine: AsyncEngine) -> AsyncIterator[AsyncSession]:
    async with engine.connect() as connection:
        transaction = await connection.begin()
        session = AsyncSession(
            bind=connection, join_transaction_mode="create_savepoint", expire_on_commit=False
        )
        try:
            yield session
        finally:
            await session.close()
            await transaction.rollback()
