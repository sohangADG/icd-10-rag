"""Batched writes for dataset import. Knows ORM tables, not source formats."""

from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import insert, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.constants import CLASSIFICATION_NODE_TYPES, IngestionRunStatus
from app.models import (
    IcdIndexEntry,
    IcdIngestionError,
    IcdIngestionRun,
    IcdNode,
    IcdRule,
    IcdSourceRef,
    IcdTerm,
)

BATCH_SIZE = 1000
_CLASSIFICATION_SQL = ", ".join(f"'{t.value}'" for t in CLASSIFICATION_NODE_TYPES)


def _chunks(
    rows: Sequence[dict[str, Any]], size: int = BATCH_SIZE
) -> Iterable[Sequence[dict[str, Any]]]:
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


class IngestionRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # --- content ------------------------------------------------------------------------------

    async def insert_nodes(self, rows: Sequence[dict[str, Any]]) -> list[int]:
        """Insert nodes; returns ids in the same order as `rows`."""
        ids: list[int] = []
        statement = insert(IcdNode).returning(IcdNode.id, sort_by_parameter_order=True)
        for chunk in _chunks(rows):
            result = await self._session.execute(statement, list(chunk))
            ids.extend(result.scalars().all())
        return ids

    async def _insert_many(self, model: type, rows: Sequence[dict[str, Any]]) -> int:
        for chunk in _chunks(rows):
            await self._session.execute(insert(model), list(chunk))
        return len(rows)

    async def insert_terms(self, rows: Sequence[dict[str, Any]]) -> int:
        return await self._insert_many(IcdTerm, rows)

    async def insert_rules(self, rows: Sequence[dict[str, Any]]) -> int:
        return await self._insert_many(IcdRule, rows)

    async def insert_index_entries(self, rows: Sequence[dict[str, Any]]) -> int:
        return await self._insert_many(IcdIndexEntry, rows)

    async def insert_source_refs(self, rows: Sequence[dict[str, Any]]) -> int:
        return await self._insert_many(IcdSourceRef, rows)

    async def resolve_rule_targets(self, dataset_id: int) -> int:
        """Link rules whose explicit target_code exists in the same dataset (set-based)."""
        result = await self._session.execute(
            text(
                f"""
                UPDATE icd_rules AS r SET target_node_id = n.id
                FROM icd_nodes AS n
                WHERE r.dataset_id = :dataset_id AND n.dataset_id = :dataset_id
                  AND r.target_code IS NOT NULL AND r.target_node_id IS NULL
                  AND n.node_type IN ({_CLASSIFICATION_SQL})
                  AND n.normalized_code = upper(regexp_replace(r.target_code, '[^A-Za-z0-9]', '',
                                                               'g'))
                """
            ),
            {"dataset_id": dataset_id},
        )
        return result.rowcount or 0

    async def resolve_index_targets(self, dataset_id: int) -> int:
        result = await self._session.execute(
            text(
                f"""
                UPDATE icd_index_entries AS e SET target_node_id = n.id
                FROM icd_nodes AS n
                WHERE e.dataset_id = :dataset_id AND n.dataset_id = :dataset_id
                  AND e.candidate_code IS NOT NULL AND e.target_node_id IS NULL
                  AND n.node_type IN ({_CLASSIFICATION_SQL})
                  AND n.normalized_code = upper(regexp_replace(e.candidate_code, '[^A-Za-z0-9]',
                                                               '', 'g'))
                """
            ),
            {"dataset_id": dataset_id},
        )
        return result.rowcount or 0

    # --- runs ---------------------------------------------------------------------------------

    async def create_run(
        self, dataset_id: int, source_filename: str, checksum: str | None, metadata: dict[str, Any]
    ) -> IcdIngestionRun:
        run = IcdIngestionRun(
            dataset_id=dataset_id,
            source_filename=source_filename,
            source_checksum=checksum,
            status=IngestionRunStatus.RUNNING,
            metadata_=metadata,
        )
        self._session.add(run)
        await self._session.flush()
        return run

    async def finish_run(
        self,
        run_id: int,
        status: IngestionRunStatus,
        *,
        records_seen: int = 0,
        records_inserted: int = 0,
        records_failed: int = 0,
        error_summary: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        values: dict[str, Any] = {
            "status": status,
            "completed_at": datetime.now(UTC),
            "records_seen": records_seen,
            "records_inserted": records_inserted,
            "records_failed": records_failed,
            "error_summary": error_summary,
        }
        if metadata is not None:
            values["metadata_"] = metadata
        await self._session.execute(
            update(IcdIngestionRun).where(IcdIngestionRun.id == run_id).values(**values)
        )

    async def add_issues(self, run_id: int, issues: Sequence[dict[str, Any]]) -> int:
        rows = [{"ingestion_run_id": run_id, **issue} for issue in issues]
        return await self._insert_many(IcdIngestionError, rows)

    async def latest_run(self, dataset_id: int) -> IcdIngestionRun | None:
        result = await self._session.execute(
            select(IcdIngestionRun)
            .where(IcdIngestionRun.dataset_id == dataset_id)
            .order_by(IcdIngestionRun.started_at.desc(), IcdIngestionRun.id.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()
