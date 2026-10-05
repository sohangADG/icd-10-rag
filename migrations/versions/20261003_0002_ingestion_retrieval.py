"""Ingestion, provenance and retrieval schema: dataset metadata, record fields, rules, search.

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-03

Extends the Phase 1 tables in place (no parallel concepts):

- pg_trgm extension for fuzzy matching.
- icd_datasets: modification/title/source metadata, page count, imported_at, metadata JSON,
  the full status lifecycle (`ingesting` becomes `processing`), one dataset per source SHA-256.
- icd_nodes: normalized_code, description, raw_text, source_locator, sort_order, metadata and
  same-dataset chapter/block/category shortcuts; trigram index on title.
- icd_terms / icd_rules / icd_index_entries: format-independent provenance (source_page becomes
  optional, source_locator added); rules gain exclusion_type and new instruction types.
- icd_search_documents: weighted tsvector + GIN, content hashes, embedding dimension/timestamp.
- icd_ingestion_runs / icd_ingestion_errors: validation outcome statuses, severity, locator.

Allowed enum values are written out literally so this migration stays frozen.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

OLD_DATASET_STATUSES = ("pending", "ingesting", "ready", "failed", "archived")
NEW_DATASET_STATUSES = (
    "pending",
    "processing",
    "validation_failed",
    "indexing",
    "ready",
    "failed",
    "archived",
)
OLD_RULE_TYPES = (
    "INCLUDE",
    "EXCLUDE",
    "NOTE",
    "USE_ADDITIONAL_CODE",
    "CODE_SEPARATELY",
    "DAGGER",
    "ASTERISK",
    "CROSS_REFERENCE",
    "INSTRUCTION",
)
NEW_RULE_TYPES = (
    *OLD_RULE_TYPES,
    "CODE_ALSO",
    "CODE_FIRST",
    "SEE",
    "SEE_ALSO",
    "OTHER",
)
OLD_RUN_STATUSES = ("pending", "running", "completed", "failed", "partially_completed")
NEW_RUN_STATUSES = (*OLD_RUN_STATUSES, "validation_failed", "skipped")
SOURCE_TYPES = ("pdf", "pdf_ocr", "text", "csv", "tsv", "json", "xml", "xlsx")
CLASSIFICATION_NODE_TYPES = ("CATEGORY", "SUBCATEGORY", "CODE")


def _in_list(values: Sequence[str]) -> str:
    return ", ".join(f"'{value}'" for value in values)


def _replace_enum_check(table: str, column: str, values: Sequence[str]) -> None:
    name = f"ck_{table}_{column}"
    op.drop_constraint(op.f(name), table, type_="check")
    op.create_check_constraint(op.f(name), table, f"{column} IN ({_in_list(values)})")


def _jsonb(name: str, *, nullable: bool) -> sa.Column:
    if nullable:
        return sa.Column(name, postgresql.JSONB(astext_type=sa.Text()), nullable=True)
    return sa.Column(
        name,
        postgresql.JSONB(astext_type=sa.Text()),
        server_default=sa.text("'{}'::jsonb"),
        nullable=False,
    )


def _same_dataset_fk(table: str, column: str, target: str) -> None:
    op.create_foreign_key(
        op.f(f"fk_{table}_dataset_id_{column}_{target}"),
        table,
        target,
        ["dataset_id", column],
        ["dataset_id", "id"],
    )


def _trigram_index(table: str, column: str) -> None:
    op.create_index(
        f"ix_{table}_{column}_trgm",
        table,
        [column],
        postgresql_using="gin",
        postgresql_ops={column: "gin_trgm_ops"},
    )


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")

    # --- icd_datasets ---------------------------------------------------------------------
    op.add_column("icd_datasets", sa.Column("modification", sa.String(64), nullable=True))
    op.add_column("icd_datasets", sa.Column("title", sa.Text(), nullable=True))
    op.add_column("icd_datasets", sa.Column("source_type", sa.String(32), nullable=True))
    op.add_column("icd_datasets", sa.Column("source_identifier", sa.Text(), nullable=True))
    op.add_column("icd_datasets", sa.Column("source_uri", sa.Text(), nullable=True))
    op.add_column("icd_datasets", sa.Column("source_page_count", sa.Integer(), nullable=True))
    op.add_column(
        "icd_datasets", sa.Column("imported_at", sa.DateTime(timezone=True), nullable=True)
    )
    op.add_column("icd_datasets", _jsonb("metadata", nullable=False))
    op.create_check_constraint(
        op.f("ck_icd_datasets_source_type"),
        "icd_datasets",
        f"source_type IN ({_in_list(SOURCE_TYPES)})",
    )
    op.create_check_constraint(
        op.f("ck_icd_datasets_page_count_non_negative"),
        "icd_datasets",
        "source_page_count IS NULL OR source_page_count >= 0",
    )
    op.drop_constraint(op.f("ck_icd_datasets_status"), "icd_datasets", type_="check")
    op.execute("UPDATE icd_datasets SET status = 'processing' WHERE status = 'ingesting'")
    op.create_check_constraint(
        op.f("ck_icd_datasets_status"),
        "icd_datasets",
        f"status IN ({_in_list(NEW_DATASET_STATUSES)})",
    )
    op.create_index("ix_icd_datasets_system", "icd_datasets", ["system"])
    op.create_index("ix_icd_datasets_status", "icd_datasets", ["status"])
    op.create_index(
        "uq_icd_datasets_source_checksum",
        "icd_datasets",
        ["source_checksum"],
        unique=True,
        postgresql_where=sa.text("source_checksum IS NOT NULL"),
    )

    # --- icd_nodes ------------------------------------------------------------------------
    for column in ("chapter_id", "block_id", "category_id"):
        op.add_column("icd_nodes", sa.Column(column, sa.BigInteger(), nullable=True))
        _same_dataset_fk("icd_nodes", column, "icd_nodes")
        op.create_index(f"ix_icd_nodes_{column}", "icd_nodes", [column])
    op.add_column("icd_nodes", sa.Column("normalized_code", sa.String(32), nullable=True))
    op.add_column("icd_nodes", sa.Column("description", sa.Text(), nullable=True))
    op.add_column("icd_nodes", sa.Column("sort_order", sa.Integer(), nullable=True))
    op.add_column("icd_nodes", _jsonb("source_locator", nullable=True))
    op.add_column("icd_nodes", sa.Column("raw_text", sa.Text(), nullable=True))
    op.add_column("icd_nodes", _jsonb("metadata", nullable=False))
    # Backfill: punctuation-free upper-case code (same rule as app.ingestion.codes).
    op.execute(
        "UPDATE icd_nodes SET normalized_code = upper(regexp_replace(code, '[^A-Za-z0-9]', '', "
        "'g')) WHERE code IS NOT NULL"
    )
    op.create_index(
        "uq_icd_nodes_dataset_classification_normalized_code",
        "icd_nodes",
        ["dataset_id", "normalized_code"],
        unique=True,
        postgresql_where=sa.text(
            f"normalized_code IS NOT NULL AND node_type IN ({_in_list(CLASSIFICATION_NODE_TYPES)})"
        ),
    )
    op.create_index(
        "ix_icd_nodes_dataset_id_normalized_code", "icd_nodes", ["dataset_id", "normalized_code"]
    )
    op.create_index("ix_icd_nodes_dataset_id_node_type", "icd_nodes", ["dataset_id", "node_type"])
    _trigram_index("icd_nodes", "title")

    # --- icd_terms ------------------------------------------------------------------------
    op.alter_column("icd_terms", "source_page", existing_type=sa.Integer(), nullable=True)
    op.add_column("icd_terms", _jsonb("source_locator", nullable=True))
    op.add_column("icd_terms", _jsonb("metadata", nullable=False))
    _trigram_index("icd_terms", "normalized_term")

    # --- icd_rules ------------------------------------------------------------------------
    _replace_enum_check("icd_rules", "rule_type", NEW_RULE_TYPES)
    op.alter_column("icd_rules", "source_page", existing_type=sa.Integer(), nullable=True)
    op.add_column("icd_rules", sa.Column("exclusion_type", sa.String(32), nullable=True))
    op.add_column("icd_rules", _jsonb("source_locator", nullable=True))
    op.add_column("icd_rules", _jsonb("metadata", nullable=False))

    # --- icd_index_entries ----------------------------------------------------------------
    op.alter_column("icd_index_entries", "source_page", existing_type=sa.Integer(), nullable=True)
    op.add_column("icd_index_entries", _jsonb("source_locator", nullable=True))
    _trigram_index("icd_index_entries", "normalized_term")

    # --- icd_source_refs ------------------------------------------------------------------
    op.alter_column("icd_source_refs", "pdf_page", existing_type=sa.Integer(), nullable=True)
    op.add_column("icd_source_refs", _jsonb("locator", nullable=True))

    # --- icd_search_documents -------------------------------------------------------------
    op.add_column("icd_search_documents", sa.Column("content_hash", sa.String(64), nullable=True))
    op.add_column(
        "icd_search_documents", sa.Column("search_vector", postgresql.TSVECTOR(), nullable=True)
    )
    op.add_column(
        "icd_search_documents", sa.Column("embedding_dimension", sa.Integer(), nullable=True)
    )
    op.add_column(
        "icd_search_documents",
        sa.Column("embedding_content_hash", sa.String(64), nullable=True),
    )
    op.add_column(
        "icd_search_documents",
        sa.Column("embedded_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.execute(
        "UPDATE icd_search_documents SET embedding_dimension = vector_dims(embedding) "
        "WHERE embedding IS NOT NULL"
    )
    op.create_check_constraint(
        op.f("ck_icd_search_documents_embedding_has_dimension"),
        "icd_search_documents",
        "embedding IS NULL OR embedding_dimension IS NOT NULL",
    )
    op.create_unique_constraint(
        op.f("uq_icd_search_documents_dataset_node_type"),
        "icd_search_documents",
        ["dataset_id", "node_id", "document_type"],
    )
    op.create_index(
        "ix_icd_search_documents_search_vector",
        "icd_search_documents",
        ["search_vector"],
        postgresql_using="gin",
    )
    op.create_index(
        "ix_icd_search_documents_embedding_model", "icd_search_documents", ["embedding_model"]
    )

    # --- icd_ingestion_runs / errors ------------------------------------------------------
    _replace_enum_check("icd_ingestion_runs", "status", NEW_RUN_STATUSES)
    op.add_column(
        "icd_ingestion_errors",
        sa.Column("severity", sa.String(16), server_default="ERROR", nullable=False),
    )
    op.add_column("icd_ingestion_errors", _jsonb("locator", nullable=True))
    op.create_check_constraint(
        op.f("ck_icd_ingestion_errors_severity"),
        "icd_ingestion_errors",
        "severity IN ('ERROR', 'WARNING', 'INFO')",
    )


def _refuse_lossy_downgrade() -> None:
    """Phase 1 required page numbers everywhere. Never invent pages to satisfy NOT NULL."""
    bind = op.get_bind()
    for table, column in (
        ("icd_terms", "source_page"),
        ("icd_rules", "source_page"),
        ("icd_index_entries", "source_page"),
        ("icd_source_refs", "pdf_page"),
    ):
        count = bind.execute(sa.text(f"SELECT count(*) FROM {table} WHERE {column} IS NULL"))
        if count.scalar_one():
            raise RuntimeError(
                f"Cannot downgrade below 0002: {table}.{column} has NULL values (non-paginated "
                "sources). Delete those datasets first; pages are never fabricated."
            )
    for table, column, values in (
        ("icd_rules", "rule_type", NEW_RULE_TYPES[len(OLD_RULE_TYPES) :]),
        ("icd_ingestion_runs", "status", NEW_RUN_STATUSES[len(OLD_RUN_STATUSES) :]),
    ):
        count = bind.execute(
            sa.text(f"SELECT count(*) FROM {table} WHERE {column} IN ({_in_list(values)})")
        )
        if count.scalar_one():
            raise RuntimeError(
                f"Cannot downgrade below 0002: {table}.{column} uses values introduced in 0002."
            )


def downgrade() -> None:
    _refuse_lossy_downgrade()

    op.drop_constraint(
        op.f("ck_icd_ingestion_errors_severity"), "icd_ingestion_errors", type_="check"
    )
    op.drop_column("icd_ingestion_errors", "locator")
    op.drop_column("icd_ingestion_errors", "severity")
    _replace_enum_check("icd_ingestion_runs", "status", OLD_RUN_STATUSES)

    op.drop_index("ix_icd_search_documents_embedding_model", table_name="icd_search_documents")
    op.drop_index("ix_icd_search_documents_search_vector", table_name="icd_search_documents")
    op.drop_constraint(
        op.f("uq_icd_search_documents_dataset_node_type"), "icd_search_documents", type_="unique"
    )
    op.drop_constraint(
        op.f("ck_icd_search_documents_embedding_has_dimension"),
        "icd_search_documents",
        type_="check",
    )
    for column in (
        "embedded_at",
        "embedding_content_hash",
        "embedding_dimension",
        "search_vector",
        "content_hash",
    ):
        op.drop_column("icd_search_documents", column)

    op.drop_column("icd_source_refs", "locator")
    op.alter_column("icd_source_refs", "pdf_page", existing_type=sa.Integer(), nullable=False)

    op.drop_index("ix_icd_index_entries_normalized_term_trgm", table_name="icd_index_entries")
    op.drop_column("icd_index_entries", "source_locator")
    op.alter_column("icd_index_entries", "source_page", existing_type=sa.Integer(), nullable=False)

    for column in ("metadata", "source_locator", "exclusion_type"):
        op.drop_column("icd_rules", column)
    op.alter_column("icd_rules", "source_page", existing_type=sa.Integer(), nullable=False)
    _replace_enum_check("icd_rules", "rule_type", OLD_RULE_TYPES)

    op.drop_index("ix_icd_terms_normalized_term_trgm", table_name="icd_terms")
    op.drop_column("icd_terms", "metadata")
    op.drop_column("icd_terms", "source_locator")
    op.alter_column("icd_terms", "source_page", existing_type=sa.Integer(), nullable=False)

    op.drop_index("ix_icd_nodes_title_trgm", table_name="icd_nodes")
    op.drop_index("ix_icd_nodes_dataset_id_node_type", table_name="icd_nodes")
    op.drop_index("ix_icd_nodes_dataset_id_normalized_code", table_name="icd_nodes")
    op.drop_index("uq_icd_nodes_dataset_classification_normalized_code", table_name="icd_nodes")
    for column in ("metadata", "raw_text", "source_locator", "sort_order", "description"):
        op.drop_column("icd_nodes", column)
    op.drop_column("icd_nodes", "normalized_code")
    for column in ("category_id", "block_id", "chapter_id"):
        op.drop_index(f"ix_icd_nodes_{column}", table_name="icd_nodes")
        op.drop_constraint(
            op.f(f"fk_icd_nodes_dataset_id_{column}_icd_nodes"), "icd_nodes", type_="foreignkey"
        )
        op.drop_column("icd_nodes", column)

    op.drop_index("uq_icd_datasets_source_checksum", table_name="icd_datasets")
    op.drop_index("ix_icd_datasets_status", table_name="icd_datasets")
    op.drop_index("ix_icd_datasets_system", table_name="icd_datasets")
    op.drop_constraint(op.f("ck_icd_datasets_status"), "icd_datasets", type_="check")
    # Statuses that did not exist in 0001 collapse onto their nearest 0001 equivalent.
    op.execute(
        "UPDATE icd_datasets SET status = CASE status "
        "WHEN 'processing' THEN 'ingesting' WHEN 'indexing' THEN 'ingesting' "
        "WHEN 'validation_failed' THEN 'failed' ELSE status END"
    )
    op.create_check_constraint(
        op.f("ck_icd_datasets_status"),
        "icd_datasets",
        f"status IN ({_in_list(OLD_DATASET_STATUSES)})",
    )
    op.drop_constraint(
        op.f("ck_icd_datasets_page_count_non_negative"), "icd_datasets", type_="check"
    )
    op.drop_constraint(op.f("ck_icd_datasets_source_type"), "icd_datasets", type_="check")
    for column in (
        "metadata",
        "imported_at",
        "source_page_count",
        "source_uri",
        "source_identifier",
        "source_type",
        "title",
        "modification",
    ):
        op.drop_column("icd_datasets", column)
    # pg_trgm is left installed, like `vector` in 0001: CREATE EXTENSION IF NOT EXISTS makes
    # re-applying this migration safe.
