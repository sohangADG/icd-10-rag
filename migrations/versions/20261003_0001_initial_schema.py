"""Initial Phase 1 schema: pgvector + versioned ICD knowledge tables + ingestion tracking.

Revision ID: 0001
Revises:
Create Date: 2026-10-03

Enum-like columns are VARCHAR + CHECK. The allowed values are written out literally here (not
imported from app code) so this migration stays frozen even when the application enums evolve;
adding a value later means a new migration that replaces the CHECK constraint.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

DATASET_STATUSES = ("pending", "ingesting", "ready", "failed", "archived")
NODE_TYPES = ("CHAPTER", "BLOCK", "CATEGORY", "SUBCATEGORY", "CODE")
CLASSIFICATION_NODE_TYPES = ("CATEGORY", "SUBCATEGORY", "CODE")
NODE_STATUSES = ("active", "disabled")
TERM_TYPES = (
    "OFFICIAL_TITLE",
    "INCLUSION",
    "SYNONYM",
    "INDEX_TERM",
    "ABBREVIATION",
    "ALTERNATIVE_TERM",
)
RULE_TYPES = (
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
RELATIONSHIP_TYPES = (
    "PARENT",
    "CHILD",
    "EXCLUDES",
    "INCLUDES",
    "DAGGER_ASTERISK",
    "ADDITIONAL_CODE",
    "CROSS_REFERENCE",
    "SEE",
    "SEE_ALSO",
)
CROSS_REFERENCE_TYPES = ("SEE", "SEE_ALSO", "SEE_CONDITION")
INGESTION_RUN_STATUSES = ("pending", "running", "completed", "failed", "partially_completed")


def _in_list(values: Sequence[str]) -> str:
    return ", ".join(f"'{value}'" for value in values)


def _enum_check(table: str, column: str, values: Sequence[str]) -> sa.CheckConstraint:
    return sa.CheckConstraint(
        f"{column} IN ({_in_list(values)})", name=op.f(f"ck_{table}_{column}")
    )


def _timestamps(*, with_updated_at: bool) -> list[sa.Column]:
    columns = [
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        )
    ]
    if with_updated_at:
        columns.append(
            sa.Column(
                "updated_at",
                sa.DateTime(timezone=True),
                server_default=sa.text("now()"),
                nullable=False,
            )
        )
    return columns


def _dataset_fk(table: str) -> sa.ForeignKeyConstraint:
    return sa.ForeignKeyConstraint(
        ["dataset_id"],
        ["icd_datasets.id"],
        name=op.f(f"fk_{table}_dataset_id_icd_datasets"),
        ondelete="RESTRICT",
    )


def _same_dataset_fk(
    table: str, column: str, target: str, *, ondelete: str | None = None
) -> sa.ForeignKeyConstraint:
    return sa.ForeignKeyConstraint(
        ["dataset_id", column],
        [f"{target}.dataset_id", f"{target}.id"],
        name=op.f(f"fk_{table}_dataset_id_{column}_{target}"),
        ondelete=ondelete,
    )


def _indexes(table: str, *columns: str) -> None:
    for column in columns:
        op.create_index(op.f(f"ix_{table}_{column}"), table, [column])


def upgrade() -> None:
    # pgvector must exist before any vector column is created.
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # --- icd_datasets ---------------------------------------------------------------------
    op.create_table(
        "icd_datasets",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("system", sa.String(64), nullable=False),
        sa.Column("country", sa.String(8), nullable=False),
        sa.Column("version", sa.String(32), nullable=False),
        sa.Column("revision", sa.String(32), nullable=True),
        sa.Column("edition", sa.String(64), nullable=True),
        sa.Column("publisher", sa.String(255), nullable=False),
        sa.Column("publication_year", sa.SmallInteger(), nullable=True),
        sa.Column("language", sa.String(16), nullable=False),
        sa.Column("source_filename", sa.Text(), nullable=True),
        sa.Column("source_checksum", sa.String(128), nullable=True),
        sa.Column("status", sa.String(32), server_default="pending", nullable=False),
        sa.Column("effective_from", sa.Date(), nullable=True),
        sa.Column("effective_to", sa.Date(), nullable=True),
        *_timestamps(with_updated_at=True),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_icd_datasets")),
        sa.UniqueConstraint(
            "system",
            "country",
            "version",
            "revision",
            "edition",
            "language",
            name="uq_icd_datasets_identity",
            postgresql_nulls_not_distinct=True,
        ),
        sa.CheckConstraint(
            "effective_to IS NULL OR effective_from IS NULL OR effective_to >= effective_from",
            name=op.f("ck_icd_datasets_effective_range"),
        ),
        sa.CheckConstraint(
            "publication_year IS NULL OR publication_year > 1900",
            name=op.f("ck_icd_datasets_pub_year"),
        ),
        _enum_check("icd_datasets", "status", DATASET_STATUSES),
    )

    # --- icd_nodes ------------------------------------------------------------------------
    op.create_table(
        "icd_nodes",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("dataset_id", sa.BigInteger(), nullable=False),
        sa.Column("parent_id", sa.BigInteger(), nullable=True),
        sa.Column("node_type", sa.String(32), nullable=False),
        sa.Column("code", sa.String(32), nullable=True),
        sa.Column("code_from", sa.String(32), nullable=True),
        sa.Column("code_to", sa.String(32), nullable=True),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("depth", sa.SmallInteger(), nullable=False),
        sa.Column("is_selectable", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column(
            "is_canadian_extension", sa.Boolean(), server_default=sa.text("false"), nullable=False
        ),
        sa.Column("status", sa.String(32), server_default="active", nullable=False),
        sa.Column("introduced_version", sa.String(32), nullable=True),
        sa.Column("disabled_version", sa.String(32), nullable=True),
        sa.Column("source_page_start", sa.Integer(), nullable=True),
        sa.Column("source_page_end", sa.Integer(), nullable=True),
        *_timestamps(with_updated_at=True),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_icd_nodes")),
        sa.UniqueConstraint("dataset_id", "id", name=op.f("uq_icd_nodes_dataset_id_id")),
        _dataset_fk("icd_nodes"),
        _same_dataset_fk("icd_nodes", "parent_id", "icd_nodes"),
        sa.CheckConstraint("depth >= 0", name=op.f("ck_icd_nodes_depth_non_negative")),
        sa.CheckConstraint(
            "parent_id IS NULL OR parent_id <> id", name=op.f("ck_icd_nodes_not_own_parent")
        ),
        sa.CheckConstraint(
            "source_page_end IS NULL OR source_page_start IS NULL "
            "OR source_page_end >= source_page_start",
            name=op.f("ck_icd_nodes_source_page_range"),
        ),
        _enum_check("icd_nodes", "node_type", NODE_TYPES),
        _enum_check("icd_nodes", "status", NODE_STATUSES),
    )
    _indexes("icd_nodes", "dataset_id", "parent_id", "node_type", "code")
    op.create_index("ix_icd_nodes_dataset_id_code", "icd_nodes", ["dataset_id", "code"])
    op.create_index(
        "uq_icd_nodes_dataset_classification_code",
        "icd_nodes",
        ["dataset_id", "code"],
        unique=True,
        postgresql_where=sa.text(
            f"code IS NOT NULL AND node_type IN ({_in_list(CLASSIFICATION_NODE_TYPES)})"
        ),
    )
    op.create_index(
        "uq_icd_nodes_dataset_grouping_code",
        "icd_nodes",
        ["dataset_id", "node_type", "code"],
        unique=True,
        postgresql_where=sa.text(
            f"code IS NOT NULL AND node_type NOT IN ({_in_list(CLASSIFICATION_NODE_TYPES)})"
        ),
    )

    # --- icd_terms ------------------------------------------------------------------------
    op.create_table(
        "icd_terms",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("dataset_id", sa.BigInteger(), nullable=False),
        sa.Column("node_id", sa.BigInteger(), nullable=False),
        sa.Column("term", sa.Text(), nullable=False),
        sa.Column("normalized_term", sa.Text(), nullable=False),
        sa.Column("term_type", sa.String(32), nullable=False),
        sa.Column("language", sa.String(16), nullable=False),
        sa.Column("source_page", sa.Integer(), nullable=False),
        *_timestamps(with_updated_at=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_icd_terms")),
        _dataset_fk("icd_terms"),
        _same_dataset_fk("icd_terms", "node_id", "icd_nodes", ondelete="CASCADE"),
        sa.CheckConstraint("source_page > 0", name=op.f("ck_icd_terms_source_page_positive")),
        _enum_check("icd_terms", "term_type", TERM_TYPES),
    )
    _indexes("icd_terms", "dataset_id", "node_id", "term_type")
    op.create_index(
        "ix_icd_terms_dataset_id_normalized_term", "icd_terms", ["dataset_id", "normalized_term"]
    )

    # --- icd_rules ------------------------------------------------------------------------
    op.create_table(
        "icd_rules",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("dataset_id", sa.BigInteger(), nullable=False),
        sa.Column("node_id", sa.BigInteger(), nullable=True),
        sa.Column("rule_type", sa.String(32), nullable=False),
        sa.Column("rule_text", sa.Text(), nullable=False),
        sa.Column("target_code", sa.String(32), nullable=True),
        sa.Column("target_node_id", sa.BigInteger(), nullable=True),
        sa.Column("is_mandatory", sa.Boolean(), nullable=True),
        sa.Column("scope", sa.String(64), nullable=True),
        sa.Column("source_page", sa.Integer(), nullable=False),
        *_timestamps(with_updated_at=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_icd_rules")),
        sa.UniqueConstraint("dataset_id", "id", name=op.f("uq_icd_rules_dataset_id_id")),
        _dataset_fk("icd_rules"),
        _same_dataset_fk("icd_rules", "node_id", "icd_nodes", ondelete="CASCADE"),
        _same_dataset_fk("icd_rules", "target_node_id", "icd_nodes"),
        sa.CheckConstraint("source_page > 0", name=op.f("ck_icd_rules_source_page_positive")),
        _enum_check("icd_rules", "rule_type", RULE_TYPES),
    )
    _indexes("icd_rules", "dataset_id", "node_id", "rule_type", "target_code", "target_node_id")

    # --- icd_relationships ----------------------------------------------------------------
    op.create_table(
        "icd_relationships",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("dataset_id", sa.BigInteger(), nullable=False),
        sa.Column("source_node_id", sa.BigInteger(), nullable=False),
        sa.Column("target_node_id", sa.BigInteger(), nullable=True),
        sa.Column("target_code", sa.String(32), nullable=True),
        sa.Column("relationship_type", sa.String(32), nullable=False),
        sa.Column("sequence_order", sa.Integer(), nullable=True),
        *_timestamps(with_updated_at=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_icd_relationships")),
        _dataset_fk("icd_relationships"),
        _same_dataset_fk("icd_relationships", "source_node_id", "icd_nodes", ondelete="CASCADE"),
        _same_dataset_fk("icd_relationships", "target_node_id", "icd_nodes"),
        sa.CheckConstraint(
            "target_node_id IS NOT NULL OR target_code IS NOT NULL",
            name=op.f("ck_icd_relationships_has_target"),
        ),
        _enum_check("icd_relationships", "relationship_type", RELATIONSHIP_TYPES),
    )
    _indexes(
        "icd_relationships", "dataset_id", "source_node_id", "target_node_id", "relationship_type"
    )

    # --- icd_index_entries ----------------------------------------------------------------
    op.create_table(
        "icd_index_entries",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("dataset_id", sa.BigInteger(), nullable=False),
        sa.Column("parent_id", sa.BigInteger(), nullable=True),
        sa.Column("lead_term", sa.Text(), nullable=False),
        sa.Column("normalized_term", sa.Text(), nullable=False),
        sa.Column("modifier", sa.Text(), nullable=True),
        sa.Column("depth", sa.SmallInteger(), nullable=False),
        sa.Column("candidate_code", sa.String(32), nullable=True),
        sa.Column("target_node_id", sa.BigInteger(), nullable=True),
        sa.Column("cross_reference_type", sa.String(32), nullable=True),
        sa.Column("cross_reference_target", sa.Text(), nullable=True),
        sa.Column("source_page", sa.Integer(), nullable=False),
        *_timestamps(with_updated_at=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_icd_index_entries")),
        sa.UniqueConstraint("dataset_id", "id", name=op.f("uq_icd_index_entries_dataset_id_id")),
        _dataset_fk("icd_index_entries"),
        _same_dataset_fk("icd_index_entries", "parent_id", "icd_index_entries", ondelete="CASCADE"),
        _same_dataset_fk("icd_index_entries", "target_node_id", "icd_nodes"),
        sa.CheckConstraint("depth >= 0", name=op.f("ck_icd_index_entries_depth_non_negative")),
        sa.CheckConstraint(
            "parent_id IS NULL OR parent_id <> id",
            name=op.f("ck_icd_index_entries_not_own_parent"),
        ),
        sa.CheckConstraint(
            "source_page > 0", name=op.f("ck_icd_index_entries_source_page_positive")
        ),
        _enum_check("icd_index_entries", "cross_reference_type", CROSS_REFERENCE_TYPES),
    )
    _indexes("icd_index_entries", "dataset_id", "parent_id", "candidate_code", "target_node_id")
    op.create_index(
        "ix_icd_index_entries_dataset_id_normalized_term",
        "icd_index_entries",
        ["dataset_id", "normalized_term"],
    )

    # --- icd_source_refs ------------------------------------------------------------------
    op.create_table(
        "icd_source_refs",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("dataset_id", sa.BigInteger(), nullable=False),
        sa.Column("node_id", sa.BigInteger(), nullable=True),
        sa.Column("rule_id", sa.BigInteger(), nullable=True),
        sa.Column("index_entry_id", sa.BigInteger(), nullable=True),
        sa.Column("source_filename", sa.Text(), nullable=False),
        sa.Column("pdf_page", sa.Integer(), nullable=False),
        sa.Column("printed_page", sa.String(32), nullable=True),
        sa.Column("section", sa.Text(), nullable=True),
        sa.Column("raw_text", sa.Text(), nullable=True),
        *_timestamps(with_updated_at=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_icd_source_refs")),
        _dataset_fk("icd_source_refs"),
        _same_dataset_fk("icd_source_refs", "node_id", "icd_nodes", ondelete="CASCADE"),
        _same_dataset_fk("icd_source_refs", "rule_id", "icd_rules", ondelete="CASCADE"),
        _same_dataset_fk(
            "icd_source_refs", "index_entry_id", "icd_index_entries", ondelete="CASCADE"
        ),
        sa.CheckConstraint("pdf_page > 0", name=op.f("ck_icd_source_refs_pdf_page_positive")),
    )
    _indexes("icd_source_refs", "dataset_id", "node_id", "rule_id", "index_entry_id")

    # --- icd_search_documents -------------------------------------------------------------
    # `embedding` is dimensionless until an embedding model is chosen. A later migration will
    # ALTER it to vector(<dim>) and create an HNSW index (which needs a fixed dimension).
    op.create_table(
        "icd_search_documents",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("dataset_id", sa.BigInteger(), nullable=False),
        sa.Column("node_id", sa.BigInteger(), nullable=True),
        sa.Column("document_type", sa.String(64), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column(
            "metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("embedding", Vector(), nullable=True),
        sa.Column("embedding_model", sa.String(128), nullable=True),
        *_timestamps(with_updated_at=True),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_icd_search_documents")),
        _dataset_fk("icd_search_documents"),
        _same_dataset_fk("icd_search_documents", "node_id", "icd_nodes", ondelete="CASCADE"),
        sa.CheckConstraint(
            "embedding IS NULL OR embedding_model IS NOT NULL",
            name=op.f("ck_icd_search_documents_embedding_has_model"),
        ),
    )
    _indexes("icd_search_documents", "dataset_id", "node_id", "document_type")

    # --- icd_ingestion_runs ---------------------------------------------------------------
    op.create_table(
        "icd_ingestion_runs",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("dataset_id", sa.BigInteger(), nullable=False),
        sa.Column("run_id", sa.Uuid(), nullable=False),
        sa.Column("status", sa.String(32), server_default="pending", nullable=False),
        sa.Column("source_filename", sa.Text(), nullable=False),
        sa.Column("source_checksum", sa.String(128), nullable=True),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("records_seen", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("records_inserted", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("records_updated", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("records_failed", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("error_summary", sa.Text(), nullable=True),
        sa.Column(
            "metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        *_timestamps(with_updated_at=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_icd_ingestion_runs")),
        sa.UniqueConstraint("run_id", name=op.f("uq_icd_ingestion_runs_run_id")),
        _dataset_fk("icd_ingestion_runs"),
        sa.CheckConstraint(
            "records_seen >= 0 AND records_inserted >= 0 "
            "AND records_updated >= 0 AND records_failed >= 0",
            name=op.f("ck_icd_ingestion_runs_counters_non_negative"),
        ),
        sa.CheckConstraint(
            "completed_at IS NULL OR completed_at >= started_at",
            name=op.f("ck_icd_ingestion_runs_completed_after_start"),
        ),
        _enum_check("icd_ingestion_runs", "status", INGESTION_RUN_STATUSES),
    )
    _indexes("icd_ingestion_runs", "dataset_id", "status")

    # --- icd_ingestion_errors -------------------------------------------------------------
    op.create_table(
        "icd_ingestion_errors",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("ingestion_run_id", sa.BigInteger(), nullable=False),
        sa.Column("source_page", sa.Integer(), nullable=True),
        sa.Column("entity_type", sa.String(64), nullable=True),
        sa.Column("entity_identifier", sa.Text(), nullable=True),
        sa.Column("error_type", sa.String(64), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("raw_payload", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        *_timestamps(with_updated_at=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_icd_ingestion_errors")),
        sa.ForeignKeyConstraint(
            ["ingestion_run_id"],
            ["icd_ingestion_runs.id"],
            name=op.f("fk_icd_ingestion_errors_ingestion_run_id_icd_ingestion_runs"),
            ondelete="CASCADE",
        ),
    )
    _indexes("icd_ingestion_errors", "ingestion_run_id")


def downgrade() -> None:
    for table in (
        "icd_ingestion_errors",
        "icd_ingestion_runs",
        "icd_search_documents",
        "icd_source_refs",
        "icd_index_entries",
        "icd_relationships",
        "icd_rules",
        "icd_terms",
        "icd_nodes",
        "icd_datasets",
    ):
        op.drop_table(table)
    # The `vector` extension is intentionally left installed: it is database-level
    # infrastructure that other objects may depend on, and CREATE EXTENSION IF NOT EXISTS
    # in upgrade() makes re-applying this migration safe.
