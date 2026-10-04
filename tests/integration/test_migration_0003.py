"""Migration 0003 (embedding spaces) on a database that already holds 0002-era embeddings."""

from collections.abc import Iterator

import pytest
from alembic import command
from sqlalchemy.engine import URL
from sqlalchemy.exc import IntegrityError

from app.core.config import get_settings
from tests.integration.db import alembic_config, create_fresh_database, drop_database
from tests.integration.test_migration_0002 import _run


@pytest.fixture
def database_0002() -> Iterator[URL]:
    name = f"{get_settings().database_name}_test_0003"
    url = create_fresh_database(name)
    command.upgrade(alembic_config(url), "0002")
    _run(
        url,
        "INSERT INTO icd_datasets (id, system, country, version, publisher, language, status) "
        "VALUES (1, 'SYNTH-ICD', 'XX', '2024', 'p', 'en', 'ready')",
        "INSERT INTO icd_nodes (id, dataset_id, node_type, code, normalized_code, title, depth) "
        "VALUES (1, 1, 'CATEGORY', 'A00', 'A00', 'Airway infection', 0)",
        "INSERT INTO icd_search_documents (dataset_id, node_id, document_type, content, "
        "embedding, embedding_model, embedding_dimension) VALUES "
        "(1, 1, 'icd_record', 'A00 Airway infection', '[0.6,0.8,0]', 'hashing-v1-3', 3)",
    )
    yield url
    drop_database(name)


def test_upgrade_backfills_space_and_enforces_dimensions(database_0002: URL) -> None:
    config = alembic_config(database_0002)
    command.upgrade(config, "head")
    (space,) = _run(
        database_0002,
        "SELECT embedding_provider, embedding_normalized, semantic_text, embedding_dimension "
        "FROM icd_search_documents",
    )
    # Legacy vectors are labelled from their model name; no semantic_text yet, so the next
    # `index` run regenerates them (their content hash cannot match).
    assert space == [("hashing", True, None, 3)]

    with pytest.raises(IntegrityError, match="embedding_dimension_matches"):
        _run(
            database_0002,
            "UPDATE icd_search_documents SET embedding = '[1,0,0,0]' WHERE node_id = 1",
        )
    with pytest.raises(IntegrityError, match="embedding_has_space"):
        _run(
            database_0002,
            "UPDATE icd_search_documents SET embedding_provider = NULL WHERE node_id = 1",
        )


def test_downgrade_and_reupgrade(database_0002: URL) -> None:
    config = alembic_config(database_0002)
    command.upgrade(config, "head")
    command.downgrade(config, "0002")
    (columns,) = _run(
        database_0002,
        "SELECT count(*) FROM information_schema.columns WHERE table_name = "
        "'icd_search_documents' AND column_name IN "
        "('semantic_text', 'embedding_provider', 'embedding_normalized')",
    )
    assert columns == [(0,)]
    (vectors,) = _run(database_0002, "SELECT count(*) FROM icd_search_documents")
    assert vectors == [(1,)]  # data kept
    command.upgrade(config, "head")
