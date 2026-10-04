"""Transactional, idempotent import of a validated PipelineResult into PostgreSQL.

Transactions (each serialised per dataset identity by a transaction-scoped advisory lock):

T1 (short)  register/locate the dataset, detect duplicates, open an ingestion run.
            - same identity + same SHA-256 already READY  -> SKIPPED (idempotent no-op)
            - same SHA-256 registered under another identity -> DuplicateDatasetError
            - identity READY/ARCHIVED with a different SHA-256 -> DuplicateDatasetError
              (a new release must be a new version/revision, never an in-place overwrite)
            - identity PROCESSING/INDEXING with a recent RUNNING run -> ImportInProgressError
            - otherwise (new, PENDING, FAILED, VALIDATION_FAILED, stale PROCESSING) -> proceed
            Fatal validation errors stop here: dataset VALIDATION_FAILED, issues persisted, no
            content written.
T2 (long)   delete any leftover content, insert nodes/terms/rules/index entries/provenance in
            batches, resolve explicit cross-references, build search documents, mark READY.
            Any exception rolls the whole of T2 back: partial data is never visible.
T3          on failure only: dataset FAILED, run FAILED with an error summary.

Embeddings are generated after READY in their own transaction (see service.py) so a remote
provider outage cannot invalidate an otherwise complete import.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.constants import (
    INSTRUCTION_TO_RULE,
    DatasetStatus,
    IngestionRunStatus,
    NodeType,
    RuleType,
    TermType,
)
from app.core.exceptions import DuplicateDatasetError, ImportInProgressError
from app.core.text import normalize_text
from app.indexing.indexer import SearchIndexer
from app.ingestion.codes import clean_code
from app.ingestion.hierarchy import HierarchyNode
from app.ingestion.models import IndexTermKind, SourceProvenance
from app.ingestion.pipeline import PipelineResult
from app.models import IcdDataset
from app.repositories.dataset_repository import DatasetRepository
from app.repositories.ingestion_repository import IngestionRepository

logger = logging.getLogger(__name__)

STALE_RUN_AFTER = timedelta(hours=2)
MAX_PERSISTED_ISSUES = 5000

_TERM_KIND = {
    IndexTermKind.SYNONYM: TermType.SYNONYM,
    IndexTermKind.ABBREVIATION: TermType.ABBREVIATION,
}


@dataclass
class ImportOutcome:
    status: str  # imported | skipped | validation_failed | failed
    dataset_id: int | None
    dataset_status: DatasetStatus | None
    run_id: str | None = None
    counts: dict[str, int] = field(default_factory=dict)
    statistics: dict[str, Any] = field(default_factory=dict)
    message: str = ""
    duration_ms: int = 0


def _identity(result: PipelineResult) -> dict[str, Any]:
    meta = result.metadata
    assert meta is not None
    return {
        "system": meta.coding_system,
        "country": meta.country.upper(),
        "version": meta.version,
        "revision": meta.revision,
        "edition": meta.edition,
        "language": meta.language,
    }


def _identity_key(identity: dict[str, Any]) -> str:
    return "|".join(str(identity[k] or "") for k in sorted(identity))


def _dataset_values(result: PipelineResult) -> dict[str, Any]:
    meta = result.metadata
    assert meta is not None
    return {
        **_identity(result),
        "modification": meta.modification,
        "title": meta.title,
        "publisher": meta.publisher,
        "publication_year": meta.publication_year,
        "source_type": meta.source_type,
        "source_filename": meta.source_filename,
        "source_identifier": meta.source_identifier,
        "source_uri": meta.source_uri,
        "source_checksum": meta.source_sha256,
        "source_page_count": meta.source_page_count,
        "effective_from": meta.effective_from,
        "effective_to": meta.effective_to,
        "metadata_": {"licence": meta.licence, "extra": meta.extra},
    }


def _issue_rows(result: PipelineResult) -> list[dict[str, Any]]:
    rows = []
    for issue in result.issues[:MAX_PERSISTED_ISSUES]:
        locator = issue.locator or {}
        page = locator.get("page_start")
        rows.append(
            {
                "severity": issue.severity.value,
                "error_type": issue.code,
                "message": issue.message,
                "entity_type": "record" if issue.record_key else "source",
                "entity_identifier": issue.record_key or issue.record_code,
                "source_page": page if isinstance(page, int) and page > 0 else None,
                "locator": locator or None,
            }
        )
    return rows


def _page(provenance: SourceProvenance | None) -> int | None:
    return provenance.page_start if provenance and provenance.page_start else None


def _locator(provenance: SourceProvenance | None) -> dict[str, Any] | None:
    return provenance.to_locator() if provenance else None


class DatasetImporter:
    def __init__(self, session: AsyncSession, indexer: SearchIndexer | None = None) -> None:
        self._session = session
        self._datasets = DatasetRepository(session)
        self._writes = IngestionRepository(session)
        self._indexer = indexer or SearchIndexer(session)

    async def import_result(self, result: PipelineResult) -> ImportOutcome:
        started = time.perf_counter()
        if result.metadata is None:
            # Without identity there is nothing to register: report, persist nothing.
            return ImportOutcome(
                status="validation_failed",
                dataset_id=None,
                dataset_status=None,
                statistics=result.statistics(),
                message="; ".join(i.message for i in result.fatal_issues),
            )
        identity = _identity(result)
        key = _identity_key(identity)
        checksum = result.metadata.source_sha256

        # --- T1 -------------------------------------------------------------------------
        await self._datasets.lock_identity(key)
        dataset = await self._datasets.get_by_identity_values(identity)
        owner = await self._datasets.get_by_checksum(checksum) if checksum else None
        if owner is not None and (dataset is None or owner.id != dataset.id):
            owner_id = owner.id  # read before rollback expires the instance
            await self._session.rollback()
            raise DuplicateDatasetError(
                "This source file (same SHA-256) is already registered as another dataset.",
                details={"existing_dataset_id": owner_id},
            )
        if dataset is not None:
            outcome = await self._check_existing(dataset, checksum, result, started)
            if outcome is not None:
                return outcome
            for name, value in _dataset_values(result).items():
                setattr(dataset, name, value)
        else:
            dataset = IcdDataset(**_dataset_values(result), status=DatasetStatus.PENDING)
            self._session.add(dataset)
        await self._session.flush()
        dataset_id = dataset.id
        run = await self._writes.create_run(
            dataset_id,
            result.source_filename,
            checksum,
            {"statistics": result.statistics(), "status_history": ["processing"]},
        )
        run_pk, run_uuid = run.id, str(run.run_id)

        if not result.is_valid:
            await self._writes.add_issues(run_pk, _issue_rows(result))
            await self._datasets.set_status(dataset_id, DatasetStatus.VALIDATION_FAILED)
            await self._writes.finish_run(
                run_pk,
                IngestionRunStatus.VALIDATION_FAILED,
                records_seen=len(result.records),
                records_failed=len(result.fatal_issues),
                error_summary=f"{len(result.fatal_issues)} fatal validation error(s)",
                metadata={"statistics": result.statistics()},
            )
            await self._session.commit()
            logger.warning(
                "dataset validation failed; nothing imported",
                extra={"dataset_id": dataset_id, "fatal_issues": len(result.fatal_issues)},
            )
            return ImportOutcome(
                status="validation_failed",
                dataset_id=dataset_id,
                dataset_status=DatasetStatus.VALIDATION_FAILED,
                run_id=run_uuid,
                statistics=result.statistics(),
                message=f"{len(result.fatal_issues)} fatal validation error(s); see run issues.",
                duration_ms=int((time.perf_counter() - started) * 1000),
            )
        await self._datasets.set_status(dataset_id, DatasetStatus.PROCESSING)
        await self._session.commit()

        # --- T2 -------------------------------------------------------------------------
        try:
            await self._datasets.lock_identity(key)
            await self._datasets.delete_content(dataset_id)
            counts = await self._write_content(dataset_id, result)
            await self._datasets.set_status(dataset_id, DatasetStatus.INDEXING)
            dataset = await self._datasets.get(dataset_id)
            assert dataset is not None
            counts["search_documents"] = await self._indexer.build_documents(dataset)
            await self._writes.add_issues(run_pk, _issue_rows(result))
            await self._datasets.set_status(
                dataset_id, DatasetStatus.READY, imported_at=datetime.now(UTC)
            )
            await self._writes.finish_run(
                run_pk,
                IngestionRunStatus.COMPLETED,
                records_seen=len(result.records),
                records_inserted=counts["nodes"],
                metadata={
                    "statistics": result.statistics(),
                    "counts": counts,
                    "status_history": ["processing", "indexing", "ready"],
                },
            )
            await self._session.commit()
        except Exception as exc:
            await self._session.rollback()
            # --- T3 -----------------------------------------------------------------------
            await self._datasets.set_status(dataset_id, DatasetStatus.FAILED)
            await self._writes.finish_run(
                run_pk,
                IngestionRunStatus.FAILED,
                records_seen=len(result.records),
                error_summary=f"Import failed: {type(exc).__name__}",
            )
            await self._session.commit()
            logger.error(
                "dataset import failed; transaction rolled back",
                extra={"dataset_id": dataset_id, "error_type": type(exc).__name__},
            )
            raise

        duration = int((time.perf_counter() - started) * 1000)
        logger.info(
            "dataset imported",
            extra={"dataset_id": dataset_id, "duration_ms": duration, **counts},
        )
        return ImportOutcome(
            status="imported",
            dataset_id=dataset_id,
            dataset_status=DatasetStatus.READY,
            run_id=run_uuid,
            counts=counts,
            statistics=result.statistics(),
            duration_ms=duration,
        )

    async def _check_existing(
        self, dataset: IcdDataset, checksum: str | None, result: PipelineResult, started: float
    ) -> ImportOutcome | None:
        if dataset.status in (DatasetStatus.READY, DatasetStatus.ARCHIVED):
            if dataset.source_checksum == checksum:
                dataset_id, status = dataset.id, dataset.status
                run = await self._writes.create_run(
                    dataset_id, result.source_filename, checksum, {"reason": "duplicate import"}
                )
                await self._writes.finish_run(
                    run.id,
                    IngestionRunStatus.SKIPPED,
                    records_seen=len(result.records),
                    error_summary="Identical source already imported; nothing changed.",
                )
                await self._session.commit()
                logger.info("duplicate import skipped", extra={"dataset_id": dataset_id})
                return ImportOutcome(
                    status="skipped",
                    dataset_id=dataset_id,
                    dataset_status=status,
                    run_id=str(run.run_id),
                    statistics=result.statistics(),
                    message="Identical source already imported (same identity and SHA-256).",
                    duration_ms=int((time.perf_counter() - started) * 1000),
                )
            dataset_id = dataset.id
            await self._session.rollback()
            raise DuplicateDatasetError(
                "Dataset identity is already imported from a different source file. Register "
                "the new source as a new version/revision/edition instead of overwriting.",
                details={"existing_dataset_id": dataset_id},
            )
        if dataset.status in (DatasetStatus.PROCESSING, DatasetStatus.INDEXING):
            latest = await self._writes.latest_run(dataset.id)
            if (
                latest is not None
                and latest.status == IngestionRunStatus.RUNNING
                and datetime.now(UTC) - latest.started_at < STALE_RUN_AFTER
            ):
                dataset_id = dataset.id
                await self._session.rollback()
                raise ImportInProgressError(
                    "Another import of this dataset is in progress.",
                    details={"dataset_id": dataset_id},
                )
        return None

    # --- content ------------------------------------------------------------------------------

    async def _write_content(self, dataset_id: int, result: PipelineResult) -> dict[str, int]:
        assert result.hierarchy is not None and result.metadata is not None
        hierarchy = result.hierarchy
        language = result.metadata.language
        ids: dict[str, int] = {}

        ordered = hierarchy.ordered_for_insert()
        by_depth: dict[int, list[HierarchyNode]] = {}
        for node in ordered:
            by_depth.setdefault(node.depth, []).append(node)
        for depth in sorted(by_depth):
            nodes = by_depth[depth]
            rows = [self._node_row(dataset_id, node, ids) for node in nodes]
            new_ids = await self._writes.insert_nodes(rows)
            ids.update(
                {node.record.key: node_id for node, node_id in zip(nodes, new_ids, strict=True)}
            )

        terms: list[dict[str, Any]] = []
        rules: list[dict[str, Any]] = []
        index_entries: list[dict[str, Any]] = []
        source_refs: list[dict[str, Any]] = []
        for node in ordered:
            record, node_id = node.record, ids[node.record.key]
            for inclusion in record.inclusions:
                terms.append(
                    {
                        "dataset_id": dataset_id,
                        "node_id": node_id,
                        "term": inclusion.text,
                        "normalized_term": normalize_text(inclusion.text),
                        "term_type": TermType.INCLUSION,
                        "language": language,
                        "source_page": _page(inclusion.provenance),
                        "source_locator": _locator(inclusion.provenance),
                        "metadata_": inclusion.metadata,
                    }
                )
            for exclusion in record.exclusions:
                rules.append(
                    self._rule_row(
                        dataset_id,
                        node_id,
                        RuleType.EXCLUDE,
                        exclusion.text,
                        exclusion.target_code,
                        exclusion.referenced_codes,
                        exclusion.provenance,
                        exclusion_type=exclusion.exclusion_type,
                    )
                )
            for instruction in record.instructions:
                rules.append(
                    self._rule_row(
                        dataset_id,
                        node_id,
                        INSTRUCTION_TO_RULE[instruction.instruction_type],
                        instruction.text,
                        instruction.target_code,
                        instruction.referenced_codes,
                        instruction.provenance,
                    )
                )
            for term in record.index_terms:
                if term.kind in _TERM_KIND and not term.target_code:
                    terms.append(
                        {
                            "dataset_id": dataset_id,
                            "node_id": node_id,
                            "term": term.term,
                            "normalized_term": normalize_text(term.term),
                            "term_type": _TERM_KIND[term.kind],
                            "language": language,
                            "source_page": _page(term.provenance),
                            "source_locator": _locator(term.provenance),
                            "metadata_": {},
                        }
                    )
                    continue
                cross_type = "SEE" if term.see else "SEE_ALSO" if term.see_also else None
                index_entries.append(
                    {
                        "dataset_id": dataset_id,
                        "lead_term": term.term,
                        "normalized_term": normalize_text(term.term),
                        "modifier": term.modifier,
                        "depth": 0,
                        "candidate_code": clean_code(term.target_code)
                        if term.target_code
                        else record.code,
                        # Owner-targeted terms link directly; explicit targets resolve below.
                        "target_node_id": None if term.target_code else node_id,
                        "cross_reference_type": cross_type,
                        "cross_reference_target": term.see or term.see_also,
                        "source_page": _page(term.provenance),
                        "source_locator": _locator(term.provenance),
                    }
                )
            provenance = record.provenance
            if provenance is not None:
                source_refs.append(
                    {
                        "dataset_id": dataset_id,
                        "node_id": node_id,
                        "source_filename": provenance.source_filename,
                        "pdf_page": provenance.page_start,
                        "printed_page": provenance.printed_page,
                        "locator": provenance.to_locator(),
                    }
                )

        counts = {
            "nodes": len(ids),
            "terms": await self._writes.insert_terms(terms),
            "rules": await self._writes.insert_rules(rules),
            "index_entries": await self._writes.insert_index_entries(index_entries),
            "source_refs": await self._writes.insert_source_refs(source_refs),
        }
        counts["rule_targets_resolved"] = await self._writes.resolve_rule_targets(dataset_id)
        counts["index_targets_resolved"] = await self._writes.resolve_index_targets(dataset_id)
        return counts

    @staticmethod
    def _node_row(dataset_id: int, node: HierarchyNode, ids: dict[str, int]) -> dict[str, Any]:
        record = node.record
        provenance = record.provenance
        metadata = dict(record.metadata)
        if node.parent_inferred:
            metadata["parent_inferred"] = True
        level = record.level or NodeType.CODE
        return {
            "dataset_id": dataset_id,
            "parent_id": ids.get(node.parent_key) if node.parent_key else None,
            "chapter_id": ids.get(node.chapter_key) if node.chapter_key else None,
            "block_id": ids.get(node.block_key) if node.block_key else None,
            "category_id": ids.get(node.category_key) if node.category_key else None,
            "node_type": level,
            "code": record.code,
            "normalized_code": record.normalized_code,
            "code_from": record.range_start,
            "code_to": record.range_end,
            "title": record.title or "",
            "description": record.description,
            "depth": node.depth,
            "sort_order": record.sort_order,
            "is_selectable": node.is_selectable,
            "status": record.status,
            "source_page_start": provenance.page_start if provenance else None,
            "source_page_end": provenance.page_end if provenance else None,
            "source_locator": _locator(provenance),
            "raw_text": record.raw_text,
            "metadata_": metadata,
        }

    @staticmethod
    def _rule_row(
        dataset_id: int,
        node_id: int,
        rule_type: RuleType,
        rule_text: str,
        target_code: str | None,
        referenced: list[str],
        provenance: SourceProvenance | None,
        *,
        exclusion_type: str | None = None,
    ) -> dict[str, Any]:
        return {
            "dataset_id": dataset_id,
            "node_id": node_id,
            "rule_type": rule_type,
            "rule_text": rule_text,
            "exclusion_type": exclusion_type,
            "target_code": clean_code(target_code) if target_code else None,
            "source_page": _page(provenance),
            "source_locator": _locator(provenance),
            "metadata_": {"referenced_codes": referenced} if referenced else {},
        }
