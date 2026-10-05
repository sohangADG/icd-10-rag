from pathlib import Path

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.constants import DatasetStatus, IngestionRunStatus, NodeType, RuleType, TermType
from app.core.exceptions import DuplicateDatasetError
from app.indexing.indexer import SearchIndexer
from app.ingestion.adapters import SourceManifest
from app.ingestion.importer import DatasetImporter
from app.ingestion.pipeline import run_pipeline
from app.models import (
    IcdDataset,
    IcdIndexEntry,
    IcdIngestionError,
    IcdIngestionRun,
    IcdNode,
    IcdRule,
    IcdSearchDocument,
    IcdSourceRef,
    IcdTerm,
)
from app.synthetic.cli import build
from tests.helpers import write_source
from tests.integration.factories import make_dataset, make_node
from tests.integration.ingest import import_synthetic


async def _count(session: AsyncSession, model: type, dataset_id: int) -> int:
    return (
        await session.execute(
            select(func.count()).select_from(model).where(model.dataset_id == dataset_id)
        )
    ).scalar_one()


async def _node(session: AsyncSession, dataset_id: int, code: str) -> IcdNode:
    return (
        await session.execute(
            select(IcdNode).where(IcdNode.dataset_id == dataset_id, IcdNode.code == code)
        )
    ).scalar_one()


async def test_import_persists_a_ready_dataset(session: AsyncSession, tmp_path: Path) -> None:
    path, manifest = write_source(tmp_path, "json")
    outcome = await DatasetImporter(session).import_result(run_pipeline(path, manifest))

    assert outcome.status == "imported" and outcome.dataset_status == DatasetStatus.READY
    dataset = await session.get(IcdDataset, outcome.dataset_id)
    assert dataset is not None and dataset.status == DatasetStatus.READY
    assert dataset.system == "SYNTH-ICD" and dataset.version == "2024"
    assert dataset.source_checksum and len(dataset.source_checksum) == 64
    assert dataset.imported_at is not None and dataset.metadata_["licence"]["basis"] == "synthetic"
    assert outcome.counts == {
        "nodes": 70,
        "terms": 29,
        "rules": 17,
        "index_entries": 2,
        "source_refs": 70,
        "rule_targets_resolved": 13,
        "index_targets_resolved": 0,
        "search_documents": 70,
    }
    assert await _count(session, IcdNode, dataset.id) == 70


async def test_hierarchy_links_and_shortcuts(session: AsyncSession, tmp_path: Path) -> None:
    outcome = await import_synthetic(session, tmp_path, embed=False)
    code = await _node(session, outcome.dataset_id, "C00.20")
    sub = await _node(session, outcome.dataset_id, "C00.2")
    category = await _node(session, outcome.dataset_id, "C00")
    block = await _node(session, outcome.dataset_id, "C00-C09")
    chapter = await _node(session, outcome.dataset_id, "III")

    assert (code.parent_id, sub.parent_id, category.parent_id) == (sub.id, category.id, block.id)
    assert block.parent_id == chapter.id and chapter.parent_id is None
    assert (code.chapter_id, code.block_id, code.category_id) == (chapter.id, block.id, category.id)
    assert (code.node_type, code.depth, code.is_selectable) == (NodeType.CODE, 4, True)
    assert not category.is_selectable and code.normalized_code == "C0020"
    assert (block.code_from, block.code_to) == ("C00", "C09")


async def test_rules_terms_and_cross_references(session: AsyncSession, tmp_path: Path) -> None:
    outcome = await import_synthetic(session, tmp_path, embed=False)
    a00 = await _node(session, outcome.dataset_id, "A00")
    b15 = await _node(session, outcome.dataset_id, "B15")
    exclusion = (
        await session.execute(
            select(IcdRule).where(IcdRule.node_id == a00.id, IcdRule.rule_type == RuleType.EXCLUDE)
        )
    ).scalar_one()
    assert (exclusion.target_code, exclusion.target_node_id) == ("B15", b15.id)
    terms = (
        await session.execute(
            select(IcdTerm.term_type, IcdTerm.term).where(IcdTerm.node_id == a00.id)
        )
    ).all()
    assert set(terms) == {
        (TermType.INCLUSION, "bronchial passage infection"),
        (TermType.SYNONYM, "respiratory tract infection"),
    }
    entry = (
        await session.execute(select(IcdIndexEntry).where(IcdIndexEntry.target_node_id == a00.id))
    ).scalar_one()
    assert (entry.lead_term, entry.candidate_code) == ("chest infection", "A00")
    note = (
        await session.execute(
            select(IcdRule).where(IcdRule.node_id == b15.id, IcdRule.rule_type == RuleType.NOTE)
        )
    ).scalar_one()
    assert note.target_code is None  # no code in the text -> no fabricated target


async def test_pdf_import_keeps_page_provenance(session: AsyncSession, tmp_path: Path) -> None:
    outcome = await import_synthetic(session, tmp_path, "pdf", embed=False)
    dataset = await session.get(IcdDataset, outcome.dataset_id)
    assert dataset.source_type.value == "pdf" and dataset.source_page_count >= 5
    node = await _node(session, outcome.dataset_id, "B15")
    assert node.source_page_start >= 2 and node.raw_text.startswith("B15")
    assert node.source_locator["kind"] == "pdf"
    ref = (
        await session.execute(select(IcdSourceRef).where(IcdSourceRef.node_id == node.id))
    ).scalar_one()
    assert ref.pdf_page == node.source_page_start
    assert ref.printed_page == str(node.source_page_start - 1)
    no_page = (
        await session.execute(
            select(func.count())
            .select_from(IcdNode)
            .where(IcdNode.dataset_id == dataset.id, IcdNode.source_page_start.is_(None))
        )
    ).scalar_one()
    assert no_page == 0


async def test_duplicate_import_is_idempotent(session: AsyncSession, tmp_path: Path) -> None:
    first = await import_synthetic(session, tmp_path, embed=False)
    path, manifest = write_source(tmp_path, "json")
    second = await DatasetImporter(session).import_result(run_pipeline(path, manifest))

    assert second.status == "skipped" and second.dataset_id == first.dataset_id
    assert await _count(session, IcdNode, first.dataset_id) == 70
    statuses = (
        (
            await session.execute(
                select(IcdIngestionRun.status).where(IcdIngestionRun.dataset_id == first.dataset_id)
            )
        )
        .scalars()
        .all()
    )
    assert sorted(statuses) == sorted([IngestionRunStatus.COMPLETED, IngestionRunStatus.SKIPPED])


async def test_same_identity_with_a_different_file_is_refused(
    session: AsyncSession, tmp_path: Path
) -> None:
    await import_synthetic(session, tmp_path, "json", embed=False)
    path, manifest = write_source(tmp_path, "csv")  # same identity, different bytes
    with pytest.raises(DuplicateDatasetError, match="different source file"):
        await DatasetImporter(session).import_result(run_pipeline(path, manifest))


async def test_same_file_under_another_identity_is_refused(
    session: AsyncSession, tmp_path: Path
) -> None:
    await import_synthetic(session, tmp_path, "csv", embed=False)
    path, manifest = write_source(tmp_path, "csv")
    manifest.dataset = {**manifest.dataset, "version": "2024-copy"}
    with pytest.raises(DuplicateDatasetError, match="same SHA-256"):
        await DatasetImporter(session).import_result(run_pipeline(path, manifest))


async def test_validation_failure_persists_issues_but_no_content(
    session: AsyncSession, tmp_path: Path
) -> None:
    build(tmp_path)
    path = tmp_path / "synth_malformed.csv"
    manifest = SourceManifest.from_file(tmp_path / "synth_malformed.csv.manifest.json")
    outcome = await DatasetImporter(session).import_result(run_pipeline(path, manifest))

    assert outcome.status == "validation_failed"
    dataset = await session.get(IcdDataset, outcome.dataset_id)
    assert dataset.status == DatasetStatus.VALIDATION_FAILED
    assert await _count(session, IcdNode, dataset.id) == 0
    run = (
        await session.execute(
            select(IcdIngestionRun).where(IcdIngestionRun.dataset_id == dataset.id)
        )
    ).scalar_one()
    assert run.status == IngestionRunStatus.VALIDATION_FAILED
    issue_types = set(
        (
            await session.execute(
                select(IcdIngestionError.error_type).where(
                    IcdIngestionError.ingestion_run_id == run.id
                )
            )
        ).scalars()
    )
    assert {"DUPLICATE_CODE", "MALFORMED_CODE", "MISSING_TITLE", "HIERARCHY_CYCLE"} <= issue_types


async def test_failure_rolls_back_and_retry_succeeds(
    session: AsyncSession, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, manifest = write_source(tmp_path, "json")

    async def explode(self, dataset):  # noqa: ANN001, ANN202
        raise RuntimeError("indexing exploded")

    monkeypatch.setattr(SearchIndexer, "build_documents", explode)
    with pytest.raises(RuntimeError, match="indexing exploded"):
        await DatasetImporter(session).import_result(run_pipeline(path, manifest))

    dataset = (
        await session.execute(select(IcdDataset).where(IcdDataset.system == "SYNTH-ICD"))
    ).scalar_one()
    assert dataset.status == DatasetStatus.FAILED
    assert await _count(session, IcdNode, dataset.id) == 0  # nothing partial survived
    assert await _count(session, IcdRule, dataset.id) == 0
    run = (
        await session.execute(
            select(IcdIngestionRun).where(IcdIngestionRun.dataset_id == dataset.id)
        )
    ).scalar_one()
    assert run.status == IngestionRunStatus.FAILED and "RuntimeError" in run.error_summary

    monkeypatch.undo()
    retry = await DatasetImporter(session).import_result(run_pipeline(path, manifest))
    assert retry.status == "imported" and retry.dataset_id == dataset.id
    await session.refresh(dataset)
    assert dataset.status == DatasetStatus.READY
    assert await _count(session, IcdNode, dataset.id) == 70


async def test_versions_are_isolated(session: AsyncSession, tmp_path: Path) -> None:
    v2024 = await import_synthetic(session, tmp_path, "json", "2024", embed=False)
    v2025 = await import_synthetic(session, tmp_path, "json", "2025", embed=False)
    assert v2024.dataset_id != v2025.dataset_id

    a01_2024 = await _node(session, v2024.dataset_id, "A01")
    a01_2025 = await _node(session, v2025.dataset_id, "A01")
    assert a01_2024.id != a01_2025.id
    new_code = await session.execute(select(IcdNode.dataset_id).where(IcdNode.code == "A01.3"))
    assert new_code.scalars().all() == [v2025.dataset_id]
    retitled = await _node(session, v2025.dataset_id, "B00.9")
    assert retitled.title.endswith("organ involvement")


async def test_database_rejects_cross_dataset_hierarchy_shortcuts(session: AsyncSession) -> None:
    first = await make_dataset(session, version="A")
    second = await make_dataset(session, version="B")
    chapter = await make_node(session, first, node_type=NodeType.CHAPTER, code="I")
    with pytest.raises(IntegrityError, match="fk_icd_nodes_dataset_id_chapter_id_icd_nodes"):
        async with session.begin_nested():
            session.add(
                IcdNode(
                    dataset_id=second.id,
                    chapter_id=chapter.id,
                    node_type=NodeType.CATEGORY,
                    code="T00",
                    title="x",
                    depth=1,
                )
            )
            await session.flush()


async def test_search_documents_exclude_exclusions_from_the_lexical_index(
    session: AsyncSession, tmp_path: Path
) -> None:
    outcome = await import_synthetic(session, tmp_path, embed=False)
    a00 = await _node(session, outcome.dataset_id, "A00")
    document = (
        await session.execute(select(IcdSearchDocument).where(IcdSearchDocument.node_id == a00.id))
    ).scalar_one()
    assert "Excludes: airway infection in the newborn (B15)" in document.content
    assert document.metadata_["exclusions"] == ["airway infection in the newborn (B15)"]
    lexemes = (
        await session.execute(
            text("SELECT tsvector_to_array(search_vector) FROM icd_search_documents WHERE id = :i"),
            {"i": document.id},
        )
    ).scalar_one()
    assert "newborn" not in lexemes and "infect" in lexemes
    # Child documents inherit the category's source synonyms as context.
    child = await _node(session, outcome.dataset_id, "B01.0")
    child_doc = (
        await session.execute(
            select(IcdSearchDocument).where(IcdSearchDocument.node_id == child.id)
        )
    ).scalar_one()
    assert "heart failure" in child_doc.metadata_["inherited_terms"]
