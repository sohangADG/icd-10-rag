from datetime import UTC, datetime

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.constants import (
    IngestionRunStatus,
    NodeStatus,
    NodeType,
    RelationshipType,
    RuleType,
)
from app.models import (
    IcdIndexEntry,
    IcdIngestionError,
    IcdIngestionRun,
    IcdNode,
    IcdRelationship,
    IcdRule,
    IcdSearchDocument,
    IcdSourceRef,
)
from tests.integration.factories import make_dataset, make_node


async def _flush_rejected(session: AsyncSession, row: object, constraint: str) -> None:
    with pytest.raises(IntegrityError, match=constraint):
        async with session.begin_nested():
            session.add(row)
            await session.flush()


# --- icd_nodes -----------------------------------------------------------------------------


async def test_node_hierarchy_references_parent(session: AsyncSession) -> None:
    dataset = await make_dataset(session)
    chapter = await make_node(session, dataset, node_type=NodeType.CHAPTER, code="I", depth=0)
    block = await make_node(
        session,
        dataset,
        node_type=NodeType.BLOCK,
        code=None,
        depth=1,
        parent=chapter,
        code_from="T00",
        code_to="T09",
    )
    category = await make_node(session, dataset, code="T00", depth=2, parent=block)
    # Canadian specificity: codes are not limited to four characters.
    extension = await make_node(
        session,
        dataset,
        node_type=NodeType.CODE,
        code="T00.000",
        depth=3,
        parent=category,
        is_selectable=True,
        is_canadian_extension=True,
    )

    children = (
        (await session.execute(select(IcdNode.code).where(IcdNode.parent_id == category.id)))
        .scalars()
        .all()
    )

    assert block.parent_id == chapter.id
    assert category.parent_id == block.id
    assert children == ["T00.000"]
    assert extension.status == NodeStatus.ACTIVE


async def test_node_cannot_have_parent_from_another_dataset(session: AsyncSession) -> None:
    dataset_2022 = await make_dataset(session, version="2022")
    dataset_2026 = await make_dataset(session, version="2026")
    parent = await make_node(session, dataset_2022, node_type=NodeType.CHAPTER, code="I")

    orphan = IcdNode(
        dataset_id=dataset_2026.id,
        parent_id=parent.id,
        node_type=NodeType.CATEGORY,
        code="T00",
        title="Test",
        depth=1,
    )
    await _flush_rejected(session, orphan, "fk_icd_nodes_dataset_id_parent_id_icd_nodes")


async def test_same_code_allowed_across_different_datasets(session: AsyncSession) -> None:
    icd10ca = await make_dataset(session, system="ICD-10-CA", version="2022")
    icd10cm = await make_dataset(session, system="ICD-10-CM", country="US", version="2025")

    await make_node(session, icd10ca, code="T00")
    await make_node(session, icd10cm, code="T00")

    rows = (await session.execute(select(IcdNode).where(IcdNode.code == "T00"))).scalars().all()
    assert {row.dataset_id for row in rows} == {icd10ca.id, icd10cm.id}


async def test_code_is_unique_within_a_dataset(session: AsyncSession) -> None:
    dataset = await make_dataset(session)
    await make_node(session, dataset, code="T00")

    duplicate = IcdNode(
        dataset_id=dataset.id, node_type=NodeType.CATEGORY, code="T00", title="dup", depth=0
    )
    await _flush_rejected(session, duplicate, "uq_icd_nodes_dataset_classification_code")

    # The same code under a different classification node type is still a duplicate.
    as_code = IcdNode(
        dataset_id=dataset.id, node_type=NodeType.CODE, code="T00", title="dup", depth=0
    )
    await _flush_rejected(session, as_code, "uq_icd_nodes_dataset_classification_code")


async def test_disabled_codes_are_retained(session: AsyncSession) -> None:
    dataset = await make_dataset(session)
    node = await make_node(
        session, dataset, code="T01", status=NodeStatus.DISABLED, disabled_version="0000"
    )

    stored = await session.get(IcdNode, node.id)
    assert stored is not None and stored.status == NodeStatus.DISABLED


# --- icd_ingestion_runs ----------------------------------------------------------------------


async def test_ingestion_run_can_be_created_and_updated(session: AsyncSession) -> None:
    dataset = await make_dataset(session)
    run = IcdIngestionRun(dataset_id=dataset.id, source_filename="source.pdf")
    session.add(run)
    await session.flush()

    assert run.run_id is not None
    assert run.status == IngestionRunStatus.PENDING
    assert run.records_seen == 0

    run.status = IngestionRunStatus.RUNNING
    run.records_seen = 10
    run.records_inserted = 8
    run.records_failed = 2
    session.add(
        IcdIngestionError(
            ingestion_run_id=run.id,
            source_page=3,
            entity_type="node",
            error_type="parse_error",
            message="unparseable line",
            raw_payload={"line": 42},
        )
    )
    await session.flush()

    run.status = IngestionRunStatus.PARTIALLY_COMPLETED
    run.completed_at = datetime.now(UTC)
    await session.commit()
    await session.refresh(run)

    assert run.status == IngestionRunStatus.PARTIALLY_COMPLETED
    assert (run.records_seen, run.records_inserted, run.records_failed) == (10, 8, 2)
    errors = (
        (
            await session.execute(
                select(IcdIngestionError).where(IcdIngestionError.ingestion_run_id == run.id)
            )
        )
        .scalars()
        .all()
    )
    assert [error.raw_payload for error in errors] == [{"line": 42}]


async def test_ingestion_run_counters_cannot_be_negative(session: AsyncSession) -> None:
    dataset = await make_dataset(session)
    run = IcdIngestionRun(dataset_id=dataset.id, source_filename="source.pdf", records_failed=-1)
    await _flush_rejected(session, run, "ck_icd_ingestion_runs_counters_non_negative")


# --- icd_source_refs / rules / index / relationships ----------------------------------------


async def test_source_refs_link_to_node_rule_and_index_entry(session: AsyncSession) -> None:
    dataset = await make_dataset(session)
    node = await make_node(session, dataset, code="T00")
    rule = IcdRule(
        dataset_id=dataset.id,
        node_id=node.id,
        rule_type=RuleType.EXCLUDE,
        rule_text="test exclusion",
        target_code="T01",
        source_page=10,
    )
    entry = IcdIndexEntry(
        dataset_id=dataset.id,
        lead_term="Test",
        normalized_term="test",
        depth=0,
        candidate_code="T00",
        target_node_id=node.id,
        source_page=200,
    )
    session.add_all([rule, entry])
    await session.flush()

    refs = [
        IcdSourceRef(dataset_id=dataset.id, node_id=node.id, source_filename="t.pdf", pdf_page=10),
        IcdSourceRef(dataset_id=dataset.id, rule_id=rule.id, source_filename="t.pdf", pdf_page=10),
        IcdSourceRef(
            dataset_id=dataset.id,
            index_entry_id=entry.id,
            source_filename="t.pdf",
            pdf_page=200,
            printed_page="A-1",
            section="Alphabetical Index",
        ),
    ]
    session.add_all(refs)
    await session.flush()

    stored = (
        (await session.execute(select(IcdSourceRef).where(IcdSourceRef.dataset_id == dataset.id)))
        .scalars()
        .all()
    )
    assert {(r.node_id, r.rule_id, r.index_entry_id) for r in stored} == {
        (node.id, None, None),
        (None, rule.id, None),
        (None, None, entry.id),
    }


async def test_source_ref_cannot_point_into_another_dataset(session: AsyncSession) -> None:
    dataset_a = await make_dataset(session, version="A")
    dataset_b = await make_dataset(session, version="B")
    node_a = await make_node(session, dataset_a)

    ref = IcdSourceRef(
        dataset_id=dataset_b.id, node_id=node_a.id, source_filename="t.pdf", pdf_page=1
    )
    await _flush_rejected(session, ref, "fk_icd_source_refs_dataset_id_node_id_icd_nodes")


async def test_rules_may_attach_to_any_hierarchy_level_or_none(session: AsyncSession) -> None:
    dataset = await make_dataset(session)
    chapter = await make_node(session, dataset, node_type=NodeType.CHAPTER, code="I")
    session.add_all(
        [
            IcdRule(
                dataset_id=dataset.id,
                node_id=chapter.id,
                rule_type=RuleType.NOTE,
                rule_text="chapter note",
                source_page=1,
            ),
            IcdRule(
                dataset_id=dataset.id,
                rule_type=RuleType.INSTRUCTION,
                rule_text="general instruction",
                source_page=1,
            ),
        ]
    )
    await session.flush()


async def test_index_entries_are_hierarchical(session: AsyncSession) -> None:
    dataset = await make_dataset(session)
    lead = IcdIndexEntry(
        dataset_id=dataset.id, lead_term="Test", normalized_term="test", depth=0, source_page=5
    )
    session.add(lead)
    await session.flush()
    modifier = IcdIndexEntry(
        dataset_id=dataset.id,
        parent_id=lead.id,
        lead_term="Test",
        normalized_term="test",
        modifier="with modifier",
        depth=1,
        candidate_code="T00.0",
        source_page=5,
    )
    session.add(modifier)
    await session.flush()

    assert modifier.parent_id == lead.id


async def test_relationship_requires_a_target(session: AsyncSession) -> None:
    dataset = await make_dataset(session)
    node = await make_node(session, dataset)

    dangling = IcdRelationship(
        dataset_id=dataset.id, source_node_id=node.id, relationship_type=RelationshipType.EXCLUDES
    )
    await _flush_rejected(session, dangling, "ck_icd_relationships_has_target")


# --- icd_search_documents --------------------------------------------------------------------


async def test_search_document_without_embedding(session: AsyncSession) -> None:
    dataset = await make_dataset(session)
    document = IcdSearchDocument(
        dataset_id=dataset.id, document_type="test", content="test content", metadata_={"k": "v"}
    )
    session.add(document)
    await session.flush()
    await session.refresh(document)

    assert document.embedding is None
    assert document.metadata_ == {"k": "v"}


async def test_embedding_requires_model_name(session: AsyncSession) -> None:
    dataset = await make_dataset(session)
    unlabeled = IcdSearchDocument(
        dataset_id=dataset.id, document_type="test", content="x", embedding=[0.0, 1.0]
    )
    await _flush_rejected(session, unlabeled, "ck_icd_search_documents_embedding_has_model")
