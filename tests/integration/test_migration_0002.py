"""Migration 0002 on a database that already holds Phase 1 (0001) data."""

import asyncio
from collections.abc import Iterator

import pytest
from alembic import command
from sqlalchemy import pool, text
from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import create_async_engine

from app.core.config import get_settings
from tests.integration.db import alembic_config, create_fresh_database, drop_database


def _run(url: URL, *statements: str) -> list[list[tuple]]:
    async def go() -> list[list[tuple]]:
        engine = create_async_engine(url, poolclass=pool.NullPool)
        results: list[list[tuple]] = []
        try:
            async with engine.begin() as connection:
                for statement in statements:
                    result = await connection.execute(text(statement))
                    results.append([tuple(r) for r in result] if result.returns_rows else [])
        finally:
            await engine.dispose()
        return results

    return asyncio.run(go())


@pytest.fixture
def phase1_database() -> Iterator[URL]:
    name = f"{get_settings().database_name}_test_upgrade"
    url = create_fresh_database(name)
    command.upgrade(alembic_config(url), "0001")
    _run(
        url,
        "INSERT INTO icd_datasets (id, system, country, version, publisher, language, status) "
        "VALUES (1, 'ICD-10-CA', 'CA', '2022', 'p', 'en', 'ingesting')",
        "INSERT INTO icd_nodes (id, dataset_id, node_type, code, title, depth) "
        "VALUES (1, 1, 'CATEGORY', 'A00', 'Existing', 0), "
        "(2, 1, 'SUBCATEGORY', 'A00.1', 'Existing child', 1)",
        "INSERT INTO icd_rules (dataset_id, node_id, rule_type, rule_text, source_page) "
        "VALUES (1, 1, 'EXCLUDE', 'x', 3)",
    )
    yield url
    drop_database(name)


def test_upgrade_existing_phase1_data(phase1_database: URL) -> None:
    command.upgrade(alembic_config(phase1_database), "head")
    status, codes, extensions = _run(
        phase1_database,
        "SELECT status FROM icd_datasets WHERE id = 1",
        "SELECT code, normalized_code FROM icd_nodes ORDER BY id",
        "SELECT extname FROM pg_extension WHERE extname IN ('vector', 'pg_trgm') ORDER BY 1",
    )
    assert status == [("processing",)]  # 'ingesting' mapped onto the new lifecycle
    assert codes == [("A00", "A00"), ("A00.1", "A001")]  # normalized_code backfilled
    assert extensions == [("pg_trgm",), ("vector",)]


def test_downgrade_restores_phase1_schema(phase1_database: URL) -> None:
    config = alembic_config(phase1_database)
    command.upgrade(config, "head")
    command.downgrade(config, "0001")
    status, columns = _run(
        phase1_database,
        "SELECT status FROM icd_datasets WHERE id = 1",
        "SELECT count(*) FROM information_schema.columns WHERE table_name = 'icd_nodes' "
        "AND column_name IN ('normalized_code', 'chapter_id', 'raw_text')",
    )
    assert status == [("ingesting",)] and columns == [(0,)]
    command.upgrade(config, "head")  # and forward again


def test_downgrade_refuses_to_invent_page_numbers(phase1_database: URL) -> None:
    config = alembic_config(phase1_database)
    command.upgrade(config, "head")
    _run(
        phase1_database,
        "INSERT INTO icd_terms (dataset_id, node_id, term, normalized_term, term_type, language) "
        "VALUES (1, 1, 'from a CSV row', 'from a csv row', 'INCLUSION', 'en')",
    )
    with pytest.raises(RuntimeError, match="pages are never fabricated"):
        command.downgrade(config, "0001")
