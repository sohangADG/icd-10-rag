from pathlib import Path

import pytest
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.coding.suggestion_service import SuggestionService
from app.core.config import get_settings
from app.core.constants import CLASSIFICATION_NODE_TYPES, DatasetStatus
from app.core.exceptions import (
    AmbiguousDatasetError,
    DatasetNotFound,
    DatasetNotReady,
    InvalidClinicalNoteError,
    LicenceRestrictionError,
    UnsupportedCodingSystem,
    UnsupportedDatasetVersion,
)
from app.evaluation.models import EvalCase
from app.evaluation.runner import EvaluationRunner
from app.indexing.embeddings import HashingEmbeddingProvider, get_embedding_provider
from app.indexing.indexer import SearchIndexer
from app.models import IcdDataset, IcdSearchDocument
from app.repositories.dataset_repository import DatasetRepository
from app.repositories.icd_repository import IcdRepository
from app.retrieval.hybrid import HybridRetriever, RetrievalFilters
from app.schemas.icd import SuggestRequest
from app.services.dataset_resolver import DatasetResolver
from app.synthetic.eval_cases import synthetic_cases
from tests.integration.ingest import import_synthetic


@pytest.fixture
async def dataset(session: AsyncSession, tmp_path: Path) -> IcdDataset:
    outcome = await import_synthetic(session, tmp_path)
    loaded = await session.get(IcdDataset, outcome.dataset_id)
    assert loaded is not None
    return loaded


def _retriever(session: AsyncSession, provider=None) -> HybridRetriever:  # noqa: ANN001
    return HybridRetriever(session, get_settings(), provider)


def _service(session: AsyncSession) -> SuggestionService:
    settings = get_settings()
    return SuggestionService(session, settings, get_embedding_provider(settings))


async def _suggest(session: AsyncSession, note: str, **kwargs):  # noqa: ANN003, ANN202
    request = SuggestRequest(
        clinical_note=note, coding_system="SYNTH-ICD", version="2024", **kwargs
    )
    return await _service(session).suggest(request)


# --- repositories ------------------------------------------------------------------------------


async def test_code_lookup_and_traversal(session: AsyncSession, dataset: IcdDataset) -> None:
    records = IcdRepository(session)
    by_code = await records.get_by_code(dataset.id, "c00.20")
    assert by_code is not None and by_code.code == "C00.20"
    assert (await records.get_by_code(dataset.id, "C0020")).id == by_code.id
    assert await records.get_by_code(dataset.id, "Z99.9") is None

    ancestors = await records.ancestors(dataset.id, by_code.id)
    assert [a.code for a in ancestors] == ["III", "C00-C09", "C00", "C00.2"]
    category = await records.get_by_code(dataset.id, "C00")
    assert [c.code for c in await records.children(dataset.id, category.id)] == [
        "C00.1",
        "C00.2",
        "C00.9",
    ]
    descendants = {n.code for n in await records.descendants(dataset.id, category.id)}
    assert descendants == {"C00.1", "C00.10", "C00.19", "C00.2", "C00.20", "C00.29", "C00.9"}
    many = await records.ancestors_of_many(dataset.id, [by_code.id, category.id])
    assert [a.code for a in many[category.id]] == ["III", "C00-C09"]


# --- retrieval ----------------------------------------------------------------------------------


async def test_exact_code_retrieval(session: AsyncSession, dataset: IcdDataset) -> None:
    result = await _retriever(session).retrieve(dataset, ["A01.0"], top_k=3)
    top = result.candidates[0]
    assert top.node.code == "A01.0" and top.scores.exact == 1.0


async def test_lexical_retrieval_via_source_synonym(
    session: AsyncSession, dataset: IcdDataset
) -> None:
    result = await _retriever(session).retrieve(dataset, ["hypertension"], top_k=3)
    top = result.candidates[0]
    assert top.node.code == "B00"
    # Absolute lexical scale: full query coverage (0.75) + rank density.
    assert top.scores.lexical >= 0.75 and top.scores.index_term == 1.0
    assert ("SYNONYM", "hypertension") in {(m.match_type, m.matched_text) for m in top.evidence}


async def test_fuzzy_retrieval_tolerates_misspelling(
    session: AsyncSession, dataset: IcdDataset
) -> None:
    result = await _retriever(session).retrieve(dataset, ["chronic airway infecton"], top_k=3)
    assert result.candidates[0].node.code == "A00.1"
    assert result.candidates[0].scores.fuzzy > 0.8


async def test_vector_retrieval_and_hnsw_index(session: AsyncSession, dataset: IcdDataset) -> None:
    provider = get_embedding_provider(get_settings())
    assert provider is not None
    documents = (
        await session.execute(
            select(IcdSearchDocument.embedding_model, IcdSearchDocument.embedding_dimension)
            .where(IcdSearchDocument.dataset_id == dataset.id)
            .distinct()
        )
    ).all()
    assert documents == [(provider.model, provider.dimension)]
    index = (
        (
            await session.execute(
                text(
                    "SELECT indexdef FROM pg_indexes WHERE tablename = 'icd_search_documents' "
                    "AND indexname LIKE 'ix_icd_search_documents_hnsw_%'"
                )
            )
        )
        .scalars()
        .all()
    )
    assert index and "hnsw" in index[0] and f"vector({provider.dimension})" in index[0]

    result = await _retriever(session, provider).retrieve(
        dataset, ["lobar consolidation of left lung"], top_k=5
    )
    assert result.semantic_available
    assert result.candidates[0].node.code == "A01.0"
    assert result.candidates[0].scores.semantic is not None
    assert result.candidates[0].scores.semantic > 0.5


async def test_semantic_is_unavailable_for_a_model_without_vectors(
    session: AsyncSession, dataset: IcdDataset
) -> None:
    other_model = HashingEmbeddingProvider(dimension=64)
    result = await _retriever(session, other_model).retrieve(dataset, ["airway"], top_k=3)
    assert not result.semantic_available
    assert all(c.scores.semantic is None for c in result.candidates)


async def test_hybrid_scores_are_exposed_and_bounded(
    session: AsyncSession, dataset: IcdDataset
) -> None:
    result = await _retriever(session, get_embedding_provider(get_settings())).retrieve(
        dataset,
        ["acute heart failure"],
        top_k=4,
        filters=RetrievalFilters(node_types=CLASSIFICATION_NODE_TYPES, selectable_only=True),
    )
    assert len(result.candidates) == 4
    assert result.candidates[0].node.code == "B01.0"
    for candidate in result.candidates:
        assert candidate.node.is_selectable and candidate.node.dataset_id == dataset.id
        assert set(candidate.scores.as_dict()) == {
            "exact",
            "lexical",
            "fuzzy",
            "semantic",
            "hierarchy",
            "index_term",
        }
        assert 0.0 <= candidate.hybrid_score <= 1.0
    scores = [c.hybrid_score for c in result.candidates]
    assert scores == sorted(scores, reverse=True)


async def test_retrieval_is_version_isolated(session: AsyncSession, tmp_path: Path) -> None:
    v2024 = await session.get(
        IcdDataset, (await import_synthetic(session, tmp_path, "json", "2024")).dataset_id
    )
    v2025 = await session.get(
        IcdDataset, (await import_synthetic(session, tmp_path, "json", "2025")).dataset_id
    )
    query = ["lobar consolidation of multiple lobes"]
    in_2025 = await _retriever(session).retrieve(v2025, query, top_k=1)
    in_2024 = await _retriever(session).retrieve(v2024, query, top_k=10)
    assert in_2025.candidates[0].node.code == "A01.3"
    assert "A01.3" not in {c.node.code for c in in_2024.candidates}
    assert all(c.node.dataset_id == v2024.id for c in in_2024.candidates)


async def test_embeddings_are_not_regenerated_for_unchanged_content(
    session: AsyncSession, dataset: IcdDataset
) -> None:
    indexer = SearchIndexer(session, get_embedding_provider(get_settings()))
    stats = await indexer.embed_documents(dataset)
    assert stats.embedded == 0 and stats.skipped_unchanged == 55


async def test_remote_embedding_requires_licence_permission(
    session: AsyncSession, dataset: IcdDataset
) -> None:
    class RemoteProvider(HashingEmbeddingProvider):
        remote = True

    dataset.metadata_ = {"licence": {"basis": "restricted"}, "extra": {}}
    await session.flush()
    with pytest.raises(LicenceRestrictionError):
        await SearchIndexer(session, RemoteProvider(32)).embed_documents(dataset)


# --- dataset resolution ---------------------------------------------------------------------------


async def test_dataset_resolution_errors(session: AsyncSession, dataset: IcdDataset) -> None:
    resolver = DatasetResolver(session)
    assert (await resolver.resolve(coding_system="synth-icd", version="2024")).id == dataset.id
    with pytest.raises(UnsupportedCodingSystem):
        await resolver.resolve(coding_system="ICD-99", version="2024")
    with pytest.raises(UnsupportedDatasetVersion):
        await resolver.resolve(coding_system="SYNTH-ICD", version="1999")
    with pytest.raises(DatasetNotFound):
        await resolver.resolve(dataset_id=999_999)
    with pytest.raises(DatasetNotFound):
        await resolver.resolve(dataset_id=dataset.id, version="2025")
    await DatasetRepository(session).set_status(dataset.id, DatasetStatus.INDEXING)
    await session.flush()
    session.expire_all()
    with pytest.raises(DatasetNotReady):
        await resolver.resolve(coding_system="SYNTH-ICD", version="2024")


async def test_ambiguous_resolution(session: AsyncSession, tmp_path: Path) -> None:
    await import_synthetic(session, tmp_path, "json", "2024", embed=False)
    await import_synthetic(session, tmp_path, "json", "2025", embed=False)
    with pytest.raises(AmbiguousDatasetError):
        await DatasetResolver(session).resolve(coding_system="SYNTH-ICD")


# --- suggestions --------------------------------------------------------------------------------


async def test_suggestion_structure_and_evidence(
    session: AsyncSession, dataset: IcdDataset
) -> None:
    response = await _suggest(session, "Assessment: Acute airway infection. Plan: fluids.")
    (suggestion,) = response.suggestions
    assert (suggestion.code, suggestion.confidence) == ("A00.0", "HIGH")
    assert suggestion.validation["db_verified"] and suggestion.validation["dataset_match"]
    assert suggestion.icd_reference["dataset_id"] == dataset.id
    assert [h["code"] for h in suggestion.icd_reference["hierarchy"]] == ["I", "A00-A09", "A00"]
    assert suggestion.evidence[0] == {
        "type": "clinical_text",
        "text": "Acute airway infection",
        "start": 12,
        "end": 34,
        "status_cues": [],
    }
    assert {"exact", "lexical", "fuzzy", "semantic", "hybrid", "rerank"} <= set(
        suggestion.retrieval_scores
    )
    assert suggestion.source_provenance.locator["kind"] == "json"
    assert response.dataset.version == "2024"


async def test_specificity_guard_never_guesses_laterality(
    session: AsyncSession, dataset: IcdDataset
) -> None:
    (suggestion,) = (await _suggest(session, "Lobar consolidation on chest imaging.")).suggestions
    assert suggestion.code == "A01.9"
    assert "laterality" in suggestion.missing_information
    specific = {a.code for a in suggestion.alternatives}
    assert {"A01.0", "A01.1"} <= specific
    (left,) = (await _suggest(session, "Lobar consolidation of the left lung.")).suggestions
    assert left.code == "A01.0" and not left.missing_information


async def test_unsupported_detail_falls_back_to_the_category(
    session: AsyncSession, dataset: IcdDataset
) -> None:
    (suggestion,) = (await _suggest(session, "Known HTN.")).suggestions
    assert suggestion.code == "B00"
    assert not suggestion.icd_reference["is_selectable"] and suggestion.confidence == "LOW"
    assert suggestion.missing_information
    assert "warning" in suggestion.validation


async def test_exclusion_redirects_to_the_excluded_to_code(
    session: AsyncSession, dataset: IcdDataset
) -> None:
    response = await _suggest(session, "Airway infection in a 5-day-old newborn.")
    (suggestion,) = response.suggestions
    assert suggestion.code == "B15"


async def test_negated_and_ruled_out_concepts_are_not_coded(
    session: AsyncSession, dataset: IcdDataset
) -> None:
    response = await _suggest(
        session, "Denies cough. No evidence of airway infection. Pneumonia was ruled out."
    )
    assert response.suggestions == []
    assert {u.concept_status for u in response.unmatched_concepts} == {"negated", "ruled_out"}


async def test_history_and_family_history_use_history_codes(
    session: AsyncSession, dataset: IcdDataset
) -> None:
    response = await _suggest(session, "Former smoker. Family history of diabetes.")
    assert [(s.code, s.concept_status) for s in response.suggestions] == [
        ("D00.2", "history"),
        ("D01", "family_history"),
    ]


async def test_uncertain_concepts_are_capped_or_disabled(
    session: AsyncSession, dataset: IcdDataset
) -> None:
    (uncertain,) = (
        await _suggest(session, "Possible lobar consolidation of the right lung.")
    ).suggestions
    assert uncertain.code == "A01.1" and uncertain.confidence != "HIGH"
    assert uncertain.validation["concept_status_policy"]
    disabled = await _suggest(
        session, "Possible lobar consolidation of the right lung.", include_uncertain=False
    )
    assert disabled.suggestions == [] and disabled.unmatched_concepts


async def test_final_database_validation_rejects_codes_not_in_the_dataset(
    session: AsyncSession, dataset: IcdDataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def vanished(self, dataset_id, code, *, classification_only=False):  # noqa: ANN001, ANN202
        return None

    monkeypatch.setattr(IcdRepository, "get_by_code", vanished)
    response = await _suggest(session, "Acute airway infection.")
    assert response.suggestions == []
    assert "database validation" in response.unmatched_concepts[0].reason


async def test_clinical_note_size_limit(session: AsyncSession, dataset: IcdDataset) -> None:
    with pytest.raises(InvalidClinicalNoteError):
        await _suggest(session, "x " * (get_settings().clinical_note_max_chars + 1))


async def test_every_returned_code_exists_in_the_dataset(
    session: AsyncSession, dataset: IcdDataset
) -> None:
    note = (
        "Type 2 diabetes with kidney complication. CKD stage 3. Acute on chronic CHF. "
        "Severe wheezing airway disorder. Current smoker. AKI."
    )
    response = await _suggest(session, note, top_k=5)
    codes = {s.code for s in response.suggestions} | {
        a.code for s in response.suggestions for a in s.alternatives if a.code
    }
    found = await IcdRepository(session).get_by_codes(dataset.id, sorted(codes))
    assert set(found) == codes
    assert [s.code for s in response.suggestions] == [
        "C00.20",
        "C12.3",
        "B01.2",
        "A02.2",
        "D00.0",
        "C13",
    ]


# --- evaluation framework -------------------------------------------------------------------------


async def test_evaluation_framework_on_synthetic_cases(
    session: AsyncSession, dataset: IcdDataset
) -> None:
    settings = get_settings()
    runner = EvaluationRunner(session, settings, get_embedding_provider(settings))
    report = await runner.run([EvalCase.model_validate(c) for c in synthetic_cases()])
    assert report["cases"] == len(synthetic_cases())
    assert report["final_selection"]["top1_accuracy"] == 1.0
    assert report["final_selection"]["negative_case_accuracy"] == 1.0
    assert report["safety"]["unsupported_code_rate"] == 0.0
    assert report["safety"]["rule_violation_rate"] == 0.0
    assert report["retrieval"]["recall@10"] == 1.0
    assert report["human_coder_agreement"] == 1.0
    assert "system behaviour only" in report["disclaimer"]
