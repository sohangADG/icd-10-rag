"""Deterministic semantic edge cases (PostgreSQL + pgvector, no model download).

KeywordProvider is an engineered embedding space: a record document whose text starts with a
given code, and a query containing a given phrase, map to the same one-hot direction (cosine
1.0). Everything else gets a hashing vector in disjoint dimensions (cosine 0 against one-hot
directions). This lets each test state exactly which record is semantically closest to a query
and check that the rest of the pipeline (exact codes, exclusions, specificity, evidence gate)
still decides correctly.
"""

from pathlib import Path
from typing import ClassVar

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.coding.suggestion_service import SuggestionService
from app.core.config import Settings
from app.core.constants import CLASSIFICATION_NODE_TYPES
from app.indexing.embeddings import HashingEmbeddingProvider
from app.indexing.indexer import SearchIndexer
from app.models import IcdDataset
from app.retrieval.hybrid import HybridRetriever, RetrievalFilters, SemanticStatus
from app.schemas.icd import SuggestRequest
from tests.integration.ingest import import_synthetic

DIMENSION = 64
KEYED = 10  # dimensions 0..9 are reserved for one-hot "concepts"


class KeywordProvider(HashingEmbeddingProvider):
    provider_name: ClassVar[str] = "keyword_test"
    default_similarity_floor: ClassVar[float] = 0.5
    meaning_based: ClassVar[bool] = True  # an engineered "meaning" space

    def __init__(self, rules: list[tuple[str, str, int]]) -> None:
        """rules: (document code prefix, query phrase, direction index)."""
        super().__init__(DIMENSION, "keyword-v1")
        self.model = "keyword-v1"
        self._rules = rules
        self._background = HashingEmbeddingProvider(DIMENSION - KEYED)

    def _vector(self, text: str) -> list[float]:
        lowered = text.lower()
        for code_prefix, phrase, index in self._rules:
            if text.startswith(f"{code_prefix} ") or phrase in lowered:
                vector = [0.0] * DIMENSION
                vector[index] = 1.0
                return vector
        return [0.0] * KEYED + self._background._vector(text)


class ExplodingProvider(HashingEmbeddingProvider):
    async def embed_queries(self, texts):  # noqa: ANN001, ANN202
        raise RuntimeError("CUDA out of memory")  # a raw library error, not an AppError


SETTINGS = Settings(_env_file=None)
FILTERS = RetrievalFilters(node_types=CLASSIFICATION_NODE_TYPES)


@pytest.fixture
async def dataset(session: AsyncSession, tmp_path: Path) -> IcdDataset:
    outcome = await import_synthetic(session, tmp_path, embed=False)
    loaded = await session.get(IcdDataset, outcome.dataset_id)
    assert loaded is not None
    return loaded


async def _embedded(session: AsyncSession, dataset: IcdDataset, rules) -> KeywordProvider:  # noqa: ANN001
    provider = KeywordProvider(rules)
    stats = await SearchIndexer(session, provider).embed_documents(dataset)
    assert stats.embedded == 70
    # Engineered one-hot vectors are pathological for an approximate HNSW graph (mostly
    # orthogonal points), and these tests check pipeline logic, not ANN recall: use an exact
    # scan. (The index is created inside the test transaction and rolled back with it.)
    await session.execute(text(f"DROP INDEX IF EXISTS {stats.vector_index}"))
    return provider


async def _suggest(session: AsyncSession, provider, note: str):  # noqa: ANN001, ANN202
    request = SuggestRequest(clinical_note=note, coding_system="SYNTH-ICD", version="2024")
    return await SuggestionService(session, SETTINGS, provider).suggest(request)


async def test_no_lexical_hit_but_strong_semantic_hit(
    session: AsyncSession, dataset: IcdDataset
) -> None:
    provider = await _embedded(session, dataset, [("B00", "zzvasculopathy", 0)])
    result = await HybridRetriever(session, SETTINGS, provider).retrieve(
        dataset, ["zzvasculopathy"], top_k=3, filters=FILTERS
    )
    top = result.candidates[0]
    assert result.semantic_status is SemanticStatus.OK
    assert top.node.code == "B00"
    assert (top.scores.lexical, top.scores.fuzzy, top.scores.semantic) == (0.0, 0.0, 1.0)


async def test_strong_lexical_hit_with_weak_semantic_score(
    session: AsyncSession, dataset: IcdDataset
) -> None:
    provider = await _embedded(session, dataset, [("B00", "zz-never-used", 0)])
    result = await HybridRetriever(session, SETTINGS, provider).retrieve(
        dataset, ["hypertension"], top_k=3, filters=FILTERS
    )
    top = result.candidates[0]
    assert top.node.code == "B00"
    assert top.scores.semantic == 0.0 and top.scores.index_term == 1.0


async def test_exact_code_query_wins_over_a_perfect_semantic_match(
    session: AsyncSession, dataset: IcdDataset
) -> None:
    provider = await _embedded(session, dataset, [("B00", "a01.0", 0)])
    result = await HybridRetriever(session, SETTINGS, provider).retrieve(
        dataset, ["A01.0"], top_k=20, filters=FILTERS
    )
    codes = [c.node.code for c in result.candidates]
    b00 = next(c for c in result.candidates if c.node.code == "B00")
    assert codes[0] == "A01.0" and b00.scores.semantic == 1.0


async def test_excluded_candidate_with_high_semantic_score_is_rejected(
    session: AsyncSession, dataset: IcdDataset
) -> None:
    # A00 excludes "airway infection in the newborn (B15)"; make A00 the closest vector.
    provider = await _embedded(session, dataset, [("A00", "newborn", 0)])
    response = await _suggest(session, provider, "Airway infection in a 5-day-old newborn.")
    assert [s.code for s in response.suggestions] == ["B15"]
    assert all(a.code != "A00" for s in response.suggestions for a in s.alternatives)


async def test_specificity_conflict_despite_strong_semantic_match(
    session: AsyncSession, dataset: IcdDataset
) -> None:
    # Make A01.0 (LEFT lung) the closest vector for a RIGHT-lung note.
    provider = await _embedded(session, dataset, [("A01.0", "right lung", 0)])
    response = await _suggest(session, provider, "Lobar consolidation of the right lung.")
    codes = [s.code for s in response.suggestions]
    assert codes == ["A01.1"]
    assert "A01.0" not in {a.code for s in response.suggestions for a in s.alternatives}


@pytest.mark.parametrize("note", ["Pain.", "Assessment: purple quokka syndrome.", "Zzqx."])
async def test_one_word_ambiguous_or_unknown_terms_abstain(
    session: AsyncSession, dataset: IcdDataset, note: str
) -> None:
    provider = await _embedded(session, dataset, [])
    response = await _suggest(session, provider, note)
    assert response.suggestions == []  # never a random top vector match
    assert response.unmatched_concepts


async def test_filler_only_note_yields_nothing(session: AsyncSession, dataset: IcdDataset) -> None:
    response = await _suggest(session, None, "Plan: follow up in two weeks.")
    assert response.clinical_concepts == [] and response.suggestions == []


async def test_raw_provider_exception_degrades_or_fails_by_mode(
    session: AsyncSession, dataset: IcdDataset
) -> None:
    from app.core.exceptions import RetrievalError

    await SearchIndexer(session, HashingEmbeddingProvider(256)).embed_documents(dataset)
    result = await HybridRetriever(session, SETTINGS, ExplodingProvider(256)).retrieve(
        dataset, ["hypertension"], top_k=3, filters=FILTERS
    )
    assert result.semantic_status is SemanticStatus.ERROR
    assert result.candidates[0].node.code == "B00"
    required = Settings(_env_file=None, semantic_retrieval_mode="required")
    with pytest.raises(RetrievalError):
        await HybridRetriever(session, required, ExplodingProvider(256)).retrieve(
            dataset, ["hypertension"], top_k=3
        )


# --- meaning-only evidence (Phase 3 clinical evaluation: symptom-11, ambiguous-03) ---


async def test_meaning_only_match_is_accepted_when_nothing_contradicts_it(
    session: AsyncSession, dataset: IcdDataset
) -> None:
    provider = await _embedded(session, dataset, [("C13", "zzrenal", 0)])
    response = await _suggest(session, provider, "Assessment: zzrenal.")
    assert [s.code for s in response.suggestions] == ["C13"]


async def test_meaning_only_match_is_rejected_when_the_notes_words_point_elsewhere(
    session: AsyncSession, dataset: IcdDataset
) -> None:
    # "kidney" is shared with C12/C13 (rejected: chronic/acute undocumented); a perfect
    # meaning-only match to an unrelated category must not win by default.
    provider = await _embedded(session, dataset, [("C00.9", "kidney disease", 0)])
    response = await _suggest(session, provider, "Kidney disease.")
    assert response.suggestions == []


async def test_a_symptom_is_never_coded_on_meaning_only_evidence(
    session: AsyncSession, dataset: IcdDataset
) -> None:
    provider = await _embedded(session, dataset, [("C13", "fatigue", 0)])
    response = await _suggest(session, provider, "Fatigue.")
    assert response.suggestions == []
    (unmatched,) = response.unmatched_concepts
    reasons = [r for c in unmatched.rejected_candidates for r in c["reasons"]]
    assert any("Symptom" in r for r in reasons)
