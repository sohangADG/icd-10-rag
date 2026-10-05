"""Regression tests (PostgreSQL) for defects found in the pre-commit review."""

import logging
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.coding.suggestion_service import SuggestionService
from app.core.config import Settings, get_settings
from app.core.constants import RuleType
from app.indexing.embeddings import HashingEmbeddingProvider, get_embedding_provider
from app.indexing.indexer import SearchIndexer
from app.ingestion.pipeline import run_pipeline
from app.models import IcdDataset, IcdNode, IcdRule, IcdSearchDocument
from app.repositories.icd_repository import IcdRepository
from app.retrieval.hybrid import HybridRetriever
from app.schemas.icd import SuggestRequest
from tests.helpers import write_source
from tests.integration.ingest import import_synthetic


@pytest.fixture
async def dataset(session: AsyncSession, tmp_path: Path) -> IcdDataset:
    outcome = await import_synthetic(session, tmp_path)
    loaded = await session.get(IcdDataset, outcome.dataset_id)
    assert loaded is not None
    return loaded


async def _suggest(session: AsyncSession, note: str, **kwargs):  # noqa: ANN003, ANN202
    settings = get_settings()
    request = SuggestRequest(
        clinical_note=note, coding_system="SYNTH-ICD", version="2024", **kwargs
    )
    return await SuggestionService(session, settings, get_embedding_provider(settings)).suggest(
        request
    )


async def _node(session: AsyncSession, dataset: IcdDataset, code: str) -> IcdNode:
    return (
        await session.execute(
            select(IcdNode).where(IcdNode.dataset_id == dataset.id, IcdNode.code == code)
        )
    ).scalar_one()


def _all_codes(response) -> set[str]:  # noqa: ANN001
    return {s.code for s in response.suggestions} | {
        a.code for s in response.suggestions for a in s.alternatives if a.code
    }


# --- R1: records reached by fallback/descent pass the same rules ---


async def test_descended_child_with_its_own_exclusion_is_never_returned(
    session: AsyncSession, dataset: IcdDataset
) -> None:
    unspecified = await _node(session, dataset, "A01.9")
    session.add(
        IcdRule(
            dataset_id=dataset.id,
            node_id=unspecified.id,
            rule_type=RuleType.EXCLUDE,
            rule_text="lobar consolidation on chest imaging",
        )
    )
    await session.flush()

    response = await _suggest(session, "Lobar consolidation on chest imaging.")
    (suggestion,) = response.suggestions
    assert suggestion.code != "A01.9"
    assert "A01.9" not in _all_codes(response)  # not even as an alternative
    assert suggestion.code == "A01" and suggestion.confidence == "LOW"


async def test_descent_never_lands_on_a_history_code_for_a_current_condition(
    session: AsyncSession, dataset: IcdDataset
) -> None:
    response = await _suggest(session, "Tobacco exposure.")
    (suggestion,) = response.suggestions
    assert suggestion.code == "D00"  # no child is supported by "tobacco exposure"
    assert "D00.2" not in _all_codes(response)  # "Personal history of ..." is gated out


# --- specificity: less specific valid code or missing information, never invented detail ---


@pytest.mark.parametrize(
    ("note", "code", "missing"),
    [
        ("Wheezing airway disorder exacerbation.", "A02.9", "severity"),
        ("Diabetes mellitus.", "C00.9", "subtype"),
        ("Heart failure.", "B01.9", "acuity"),
        ("Lobar consolidation on chest imaging.", "A01.9", "laterality"),
        ("Left and right lung lobar consolidation.", "A01.9", "laterality (conflicting)"),
    ],
)
async def test_missing_detail_falls_back_to_the_unspecified_code(
    session: AsyncSession, dataset: IcdDataset, note: str, code: str, missing: str
) -> None:
    (suggestion,) = (await _suggest(session, note)).suggestions
    assert suggestion.code == code
    assert missing in suggestion.missing_information


async def test_missing_complication_status_stays_at_the_subcategory(
    session: AsyncSession, dataset: IcdDataset
) -> None:
    (suggestion,) = (await _suggest(session, "Type 2 diabetes.")).suggestions
    assert suggestion.code == "C00.2" and not suggestion.icd_reference["is_selectable"]
    assert suggestion.confidence == "LOW"
    assert any("complication" in m for m in suggestion.missing_information)


# --- R2: no clinical text in logs ---


async def test_rejections_do_not_log_clinical_text(
    session: AsyncSession, dataset: IcdDataset, caplog: pytest.LogCaptureFixture
) -> None:
    marker = "zyxquorb"
    caplog.set_level(logging.DEBUG)
    await _suggest(session, f"Type 2 diabetes with {marker} complication. Denies {marker} pain.")
    for record in caplog.records:
        assert marker not in record.getMessage()
        assert marker not in str(vars(record))


# --- R4: embeddings are identified by (model, dimension) ---


class _SharedNameProvider(HashingEmbeddingProvider):
    """Same model name at different dimensions (like OpenAI's `dimensions` parameter)."""

    def __init__(self, dimension: int) -> None:
        super().__init__(dimension)
        self.model = "shared-model"


async def test_changing_dimension_re_embeds_instead_of_mixing(
    session: AsyncSession, dataset: IcdDataset
) -> None:
    first = await SearchIndexer(session, _SharedNameProvider(32)).embed_documents(dataset)
    assert first.embedded == 70
    second = await SearchIndexer(session, _SharedNameProvider(64)).embed_documents(dataset)
    assert second.embedded == 70 and second.skipped_unchanged == 0
    dims = set(
        (
            await session.execute(
                select(IcdSearchDocument.embedding_dimension).where(
                    IcdSearchDocument.dataset_id == dataset.id
                )
            )
        ).scalars()
    )
    assert dims == {64}
    settings = get_settings()
    current = await HybridRetriever(session, settings, _SharedNameProvider(64)).retrieve(
        dataset, ["acute heart failure"], top_k=3
    )
    stale = await HybridRetriever(session, settings, _SharedNameProvider(32)).retrieve(
        dataset, ["acute heart failure"], top_k=3
    )
    assert current.semantic_available and not stale.semantic_available


# --- exact code matches dominate ---


async def test_exact_code_ranks_first_even_without_exact_weight(
    session: AsyncSession, dataset: IcdDataset
) -> None:
    settings = Settings(_env_file=None, retrieval_weight_exact=0.0)
    result = await HybridRetriever(session, settings).retrieve(dataset, ["A01.0"], top_k=5)
    assert result.candidates[0].node.code == "A01.0"


# --- R5: textual cross-references never become codes ---


async def test_every_rule_type_is_db_validated(session: AsyncSession, dataset: IcdDataset) -> None:
    records = IcdRepository(session)
    expected = {
        ("A00", RuleType.EXCLUDE): "B15",
        ("A01", RuleType.CODE_FIRST): "A00",
        ("A02", RuleType.USE_ADDITIONAL_CODE): "D00",
        ("C00", RuleType.CODE_ALSO): "D02",
        ("A10", RuleType.SEE_ALSO): "A00",
        ("C13", RuleType.SEE): "C12",
    }
    for (owner, rule_type), target in expected.items():
        node = await _node(session, dataset, owner)
        rule = (
            await session.execute(
                select(IcdRule).where(IcdRule.node_id == node.id, IcdRule.rule_type == rule_type)
            )
        ).scalar_one()
        target_node = await records.get_by_code(dataset.id, target)
        assert rule.target_code == target and rule.target_node_id == target_node.id

    d02 = await _node(session, dataset, "D02")
    dangling = (
        await session.execute(
            select(IcdRule).where(IcdRule.node_id == d02.id, IcdRule.rule_type == RuleType.SEE)
        )
    ).scalar_one()
    assert dangling.target_code == "Z99" and dangling.target_node_id is None
    assert await records.get_by_code(dataset.id, "Z99") is None


async def test_unresolved_reference_is_a_warning_and_never_suggested(
    session: AsyncSession, dataset: IcdDataset, tmp_path: Path
) -> None:
    result = run_pipeline(*write_source(tmp_path, "json"))
    unresolved = [i for i in result.issues if i.code == "UNRESOLVED_CROSS_REFERENCE"]
    assert [(i.record_code, i.severity.value) for i in unresolved] == [("D02", "WARNING")]

    response = await _suggest(session, "Long-term insulin use.")
    (suggestion,) = response.suggestions
    assert suggestion.code == "D02"
    see = next(c for c in suggestion.validation["rule_checks"] if c["rule_type"] == "SEE")
    assert see["target_code"] == "Z99" and see["target_exists"] is False
    assert see["target_record_id"] is None
    assert "Z99" not in _all_codes(response)
    found = await IcdRepository(session).get_by_codes(dataset.id, sorted(_all_codes(response)))
    assert set(found) == _all_codes(response)
