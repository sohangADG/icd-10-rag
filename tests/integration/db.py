"""Helpers for creating throwaway PostgreSQL databases and running Alembic against them."""

import asyncio
from pathlib import Path

import pytest
from alembic.config import Config
from sqlalchemy import pool, text
from sqlalchemy.engine import URL
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import create_async_engine

from app.core.config import get_settings

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def alembic_config(database_url: URL) -> Config:
    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    # A URL object (not a string), so the password is never interpolated into config text.
    config.attributes["database_url"] = database_url
    config.attributes["configure_logger"] = False
    return config


async def _admin_execute(*statements: str) -> None:
    engine = create_async_engine(
        get_settings().sqlalchemy_url("postgres"),
        poolclass=pool.NullPool,
        isolation_level="AUTOCOMMIT",
    )
    try:
        async with engine.connect() as connection:
            for statement in statements:
                await connection.execute(text(statement))
    finally:
        await engine.dispose()


def create_fresh_database(name: str) -> URL:
    """(Re)create an empty database. Must be called outside a running event loop."""
    try:
        asyncio.run(
            _admin_execute(
                f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)', f'CREATE DATABASE "{name}"'
            )
        )
    except (OSError, SQLAlchemyError) as exc:
        pytest.fail(
            f"PostgreSQL is not reachable ({type(exc).__name__}). Check that the server is "
            "running and that DATABASE_* in .env are correct (see README)."
        )
    return get_settings().sqlalchemy_url(name)


def drop_database(name: str) -> None:
    asyncio.run(_admin_execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
