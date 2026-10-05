"""Embedding lifecycle and semantic retrieval against PostgreSQL + pgvector.

Uses deterministic providers (hashing, and a fake sentence-transformers model whose vectors are
hashing vectors) so the tests are offline and repeatable; real-model behaviour is covered by
test_semantic_model.py (marked semantic_model).
"""

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select, text, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import ProviderHandle, embedding_provider
from app.core.config import Settings, get_settings
from app.core.constants import CLASSIFICATION_NODE_TYPES
from app.core.database import get_session
from app.core.exceptions import EmbeddingProviderError, RetrievalError
from app.indexing.embeddings import HashingEmbeddingProvider, SentenceTransformerProvider
from app.indexing.indexer import SearchIndexer
from app.main import create_app
from app.models import IcdDataset, IcdNode, IcdSearchDocument
from app.retrieval.hybrid import HybridRetriever, RetrievalFilters, SemanticStatus
from tests.integration.ingest import import_synthetic


class _HashModel:
    """A 'sentence-transformers model' producing hashing vectors (deterministic)."""

    def __init__(self, dimension: int = 256) -> None:
        self._hash = HashingEmbeddingProvider(dimension)
        self._dimension = dimension

    def get_embedding_dimension(self) -> int:
        return self._dimension

    def encode(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        return [self._hash._vector(t) for t in texts]


def fake_st(model: str = "fake/semantic", dimension: int = 256) -> SentenceTransformerProvider:
    return SentenceTransformerProvider(model, loader=lambda name, device: _HashModel(dimension))


class FailingProvider(HashingEmbeddingProvider):
    """Fails on the n-th call to embed (to simulate a crash mid-run)."""

    def __init__(self, fail_on_call: int) -> None:
        super().__init__(256)
        self.calls = 0
        self.fail_on_call = fail_on_call

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise EmbeddingProviderError("provider exploded")
        return await super()._embed(texts)


class WrongDimensionProvider(HashingEmbeddingProvider):
    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * (self.dimension + 1) for _ in texts]  # bypasses the provider check


@pytest.fixture
async def dataset(session: AsyncSession, tmp_path: Path) -> IcdDataset:
    outcome = await import_synthetic(session, tmp_path, embed=False)
    loaded = await session.get(IcdDataset, outcome.dataset_id)
    assert loaded is not None
    return loaded


async def _spaces(session: AsyncSession, dataset: IcdDataset | int) -> set[tuple]:
    dataset_id = dataset if isinstance(dataset, int) else dataset.id
    rows = await session.execute(
        select(
            IcdSearchDocument.embedding_provider,
            IcdSearchDocument.embedding_model,
            IcdSearchDocument.embedding_dimension,
            IcdSearchDocument.embedding_normalized,
        ).where(IcdSearchDocument.dataset_id == dataset_id)
    )
    return set(rows.all())


# --- content hashing / skip / regeneration ---


async def test_unchanged_documents_are_skipped_and_changed_ones_regenerated(
    session: AsyncSession, dataset: IcdDataset
) -> None:
    indexer = SearchIndexer(session, fake_st())
    first = await indexer.embed_documents(dataset, batch_size=10)
    assert (first.documents, first.embedded, first.skipped_unchanged, first.batches) == (
        70,
        70,
        0,
        7,
    )
    assert await _spaces(session, dataset) == {
        ("sentence_transformers", "fake/semantic", 256, True)
    }

    again = await indexer.embed_documents(dataset)
    assert (again.embedded, again.skipped_unchanged) == (0, 70)

    node = (
        await session.execute(
            select(IcdNode).where(IcdNode.dataset_id == dataset.id, IcdNode.code == "A00")
        )
    ).scalar_one()
    node.title = "Airway infection (retitled)"
    await session.flush()
    await SearchIndexer(session).build_documents(dataset)
    changed = await indexer.embed_documents(dataset)
    # A00's own text changed, and so did the documents that show it in their hierarchy path.
    assert changed.embedded == 4 and changed.skipped_unchanged == 66

    forced = await indexer.embed_documents(dataset, force=True)
    assert forced.embedded == 70


async def test_model_change_regenerates_and_isolates(
    session: AsyncSession, dataset: IcdDataset
) -> None:
    await SearchIndexer(session, fake_st("fake/model-a")).embed_documents(dataset)
    switched = await SearchIndexer(session, fake_st("fake/model-b")).embed_documents(dataset)
    assert switched.embedded == 70 and switched.skipped_unchanged == 0
    assert await _spaces(session, dataset) == {("sentence_transformers", "fake/model-b", 256, True)}
    settings = get_settings()
    old = await HybridRetriever(session, settings, fake_st("fake/model-a")).retrieve(
        dataset, ["airway infection"], top_k=3
    )
    assert old.semantic_status is SemanticStatus.NOT_INDEXED  # old vectors are gone, not mixed


async def test_provider_change_is_isolated_even_with_same_model_name_and_dimension(
    session: AsyncSession, dataset: IcdDataset
) -> None:
    hashing = HashingEmbeddingProvider(256)
    await SearchIndexer(session, hashing).embed_documents(dataset)
    impostor = fake_st(hashing.model, 256)  # same model string + dimension, other provider
    settings = get_settings()
    result = await HybridRetriever(session, settings, impostor).retrieve(
        dataset, ["airway"], top_k=3
    )
    assert result.semantic_status is SemanticStatus.NOT_INDEXED
    regenerated = await SearchIndexer(session, impostor).embed_documents(dataset)
    assert regenerated.embedded == 70


# --- failures / validation ---


async def test_failed_batch_keeps_earlier_batches_and_retry_resumes(
    session: AsyncSession, dataset: IcdDataset
) -> None:
    failing = FailingProvider(fail_on_call=3)
    with pytest.raises(EmbeddingProviderError, match="provider exploded"):
        await SearchIndexer(session, failing).embed_documents(dataset, batch_size=10)
    stored = (
        await session.execute(
            select(IcdSearchDocument.id).where(
                IcdSearchDocument.dataset_id == dataset.id, IcdSearchDocument.embedding.is_not(None)
            )
        )
    ).all()
    assert len(stored) == 20  # two committed batches survive

    partial = await HybridRetriever(
        session, get_settings(), HashingEmbeddingProvider(256)
    ).retrieve(dataset, ["airway"], top_k=3)
    assert partial.semantic_status is SemanticStatus.INCOMPLETE_INDEX  # never half-used

    retry = await SearchIndexer(session, HashingEmbeddingProvider(256)).embed_documents(
        dataset, batch_size=10
    )
    assert (retry.embedded, retry.skipped_unchanged) == (50, 20)


async def test_malformed_dimensions_are_never_stored(
    session: AsyncSession, dataset: IcdDataset
) -> None:
    dataset_id = dataset.id  # the failed batch's rollback expires loaded ORM objects
    with pytest.raises(EmbeddingProviderError, match="Vector of length 257"):
        await SearchIndexer(session, WrongDimensionProvider(256)).embed_documents(dataset)
    assert await _spaces(session, dataset_id) == {(None, None, None, None)}
    document = (
        await session.execute(
            select(IcdSearchDocument).where(IcdSearchDocument.dataset_id == dataset_id).limit(1)
        )
    ).scalar_one()
    with pytest.raises(IntegrityError, match="embedding_dimension_matches"):
        async with session.begin_nested():
            await session.execute(
                update(IcdSearchDocument)
                .where(IcdSearchDocument.id == document.id)
                .values(
                    embedding=[0.1, 0.2, 0.3],
                    embedding_dimension=4,
                    embedding_model="m",
                    embedding_provider="p",
                    embedding_normalized=True,
                )
            )


async def test_empty_search_document_is_skipped_without_blocking_semantic_search(
    session: AsyncSession, dataset: IcdDataset
) -> None:
    await session.execute(
        update(IcdSearchDocument)
        .where(
            IcdSearchDocument.id
            == (
                select(IcdSearchDocument.id)
                .where(IcdSearchDocument.dataset_id == dataset.id)
                .order_by(IcdSearchDocument.id)
                .limit(1)
                .scalar_subquery()
            )
        )
        .values(semantic_text="   ")
    )
    stats = await SearchIndexer(session, HashingEmbeddingProvider(256)).embed_documents(dataset)
    assert (stats.embedded, stats.skipped_empty) == (69, 1)
    result = await HybridRetriever(session, get_settings(), HashingEmbeddingProvider(256)).retrieve(
        dataset, ["airway infection"], top_k=3
    )
    assert result.semantic_status is SemanticStatus.OK


# --- semantic text: positive evidence only ---


async def test_semantic_text_excludes_exclusions_and_cross_references(
    session: AsyncSession, dataset: IcdDataset
) -> None:
    async def semantic_text(code: str) -> str:
        return (
            await session.execute(
                select(IcdSearchDocument.semantic_text)
                .join(IcdNode, IcdNode.id == IcdSearchDocument.node_id)
                .where(IcdNode.dataset_id == dataset.id, IcdNode.code == code)
            )
        ).scalar_one()

    a00 = await semantic_text("A00")
    assert a00.startswith("A00 Airway infection") and "Includes: bronchial passage" in a00
    assert "newborn" not in a00 and "Excludes" not in a00  # exclusion kept out
    assert "Hierarchy: Chapter I" in a00 and "Synonyms: respiratory tract infection" in a00
    b01 = await semantic_text("B01")
    assert "Code first" not in b01 and "elevated blood pressure" not in b01  # names another code
    b15 = await semantic_text("B15")
    assert "Notes: Use only for conditions arising" in b15  # self-describing note kept
    b010 = await semantic_text("B01.0")
    assert "Parent terms: heart failure" in b010


# --- retrieval with spaces / status / isolation ---


async def test_semantic_search_is_filtered_by_dataset_and_space(
    session: AsyncSession, tmp_path: Path
) -> None:
    v2024 = await session.get(
        IcdDataset,
        (await import_synthetic(session, tmp_path, "json", "2024", embed=False)).dataset_id,
    )
    v2025 = await session.get(
        IcdDataset,
        (await import_synthetic(session, tmp_path, "json", "2025", embed=False)).dataset_id,
    )
    provider = fake_st()
    await SearchIndexer(session, provider).embed_documents(v2024)
    await SearchIndexer(session, HashingEmbeddingProvider(256)).embed_documents(v2025)
    settings = get_settings()
    in_2024 = await HybridRetriever(session, settings, provider).retrieve(
        v2024, ["lobar consolidation of multiple lobes"], top_k=10
    )
    assert in_2024.semantic_status is SemanticStatus.OK
    assert all(c.node.dataset_id == v2024.id for c in in_2024.candidates)
    assert "A01.3" not in {c.node.code for c in in_2024.candidates}
    # 2025 has vectors, but in another space: never compared with this provider's queries.
    in_2025 = await HybridRetriever(session, settings, provider).retrieve(
        v2025, ["lobar consolidation"], top_k=3
    )
    assert in_2025.semantic_status is SemanticStatus.NOT_INDEXED


async def test_l2_distance_uses_a_matching_index(
    session: AsyncSession, dataset: IcdDataset
) -> None:
    stats = await SearchIndexer(session, HashingEmbeddingProvider(256)).embed_documents(
        dataset, distance="l2"
    )
    definition = (
        await session.execute(
            text("SELECT indexdef FROM pg_indexes WHERE indexname = :n"), {"n": stats.vector_index}
        )
    ).scalar_one()
    assert "vector_l2_ops" in definition and "'hashing'::text" in definition
    settings = Settings(_env_file=None, embedding_distance="l2")
    result = await HybridRetriever(session, settings, HashingEmbeddingProvider(256)).retrieve(
        dataset, ["airway infection"], top_k=3
    )
    assert result.semantic_status is SemanticStatus.OK
    assert result.embedding_space["distance"] == "l2"


async def test_query_failure_degrades_or_fails_by_mode(
    session: AsyncSession, dataset: IcdDataset
) -> None:
    await SearchIndexer(session, HashingEmbeddingProvider(256)).embed_documents(dataset)

    class BrokenQueries(HashingEmbeddingProvider):
        async def embed_queries(self, texts):  # noqa: ANN001, ANN202
            raise EmbeddingProviderError("query embedding failed")

    optional = await HybridRetriever(session, get_settings(), BrokenQueries(256)).retrieve(
        dataset,
        ["hypertension"],
        top_k=3,
        filters=RetrievalFilters(node_types=CLASSIFICATION_NODE_TYPES),
    )
    assert optional.semantic_status is SemanticStatus.ERROR
    assert optional.candidates[0].node.code == "B00"  # exact/lexical signals still rank
    assert all(c.scores.semantic is None for c in optional.candidates)
    required = Settings(_env_file=None, semantic_retrieval_mode="required")
    with pytest.raises(RetrievalError, match="required"):
        await HybridRetriever(session, required, BrokenQueries(256)).retrieve(
            dataset, ["hypertension"], top_k=3
        )
    with pytest.raises(RetrievalError, match="provider_unavailable"):
        await HybridRetriever(
            session, required, None, provider_status=SemanticStatus.PROVIDER_UNAVAILABLE
        ).retrieve(dataset, ["hypertension"], top_k=3)


# --- API serialisation ---


def _no_vectors(value: Any) -> bool:
    """True when no list of >= 32 floats (an embedding) appears anywhere in the payload."""
    if isinstance(value, dict):
        return "embedding" not in value and all(_no_vectors(v) for v in value.values())
    if isinstance(value, list):
        if len(value) >= 32 and all(isinstance(v, float) for v in value):
            return False
        return all(_no_vectors(v) for v in value)
    return True


async def test_api_exposes_scores_and_status_but_never_vectors(
    session: AsyncSession, dataset: IcdDataset
) -> None:
    provider = fake_st()
    await SearchIndexer(session, provider).embed_documents(dataset)
    app = create_app()

    async def test_session() -> AsyncIterator[AsyncSession]:
        yield session

    app.dependency_overrides[get_session] = test_session
    app.dependency_overrides[embedding_provider] = lambda: ProviderHandle(
        provider, SemanticStatus.OK
    )
    params = {"coding_system": "SYNTH-ICD", "version": "2024"}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as http:
        search = (
            await http.get("/api/v1/icd/search", params={**params, "q": "heart failure"})
        ).json()
        suggest = (
            await http.post(
                "/api/v1/icd/suggest", json={"clinical_note": "Acute heart failure.", **params}
            )
        ).json()
        text_mode = (
            await http.get("/api/v1/icd/search", params={**params, "q": "x", "mode": "text"})
        ).json()
    assert search["semantic_status"] == "ok"
    assert search["embedding_space"]["model"] == "fake/semantic"
    assert search["embedding_space"]["dimension"] == 256
    hit = search["results"][0]
    assert {
        "exact",
        "lexical",
        "fuzzy",
        "semantic",
        "hierarchy",
        "index_term",
        "semantic_raw",
    } <= set(hit["scores"])
    scores = suggest["suggestions"][0]["retrieval_scores"]
    assert {
        "semantic",
        "semantic_raw",
        "hybrid",
        "rerank",
        "semantic_status",
        "embedding_space",
    } <= set(scores)
    assert scores["semantic_status"] == "ok"
    assert text_mode["semantic_status"] == "not_used"
    assert _no_vectors(search) and _no_vectors(suggest)


# --- minimum evidence gate ---


async def test_weak_evidence_is_reported_not_guessed(
    session: AsyncSession, dataset: IcdDataset
) -> None:
    from app.coding.suggestion_service import SuggestionService
    from app.schemas.icd import SuggestRequest

    settings = get_settings()
    request = SuggestRequest(
        clinical_note="Assessment: purple quokka syndrome.",
        coding_system="SYNTH-ICD",
        version="2024",
    )
    response = await SuggestionService(session, settings, None).suggest(request)
    assert response.suggestions == []
    (unmatched,) = response.unmatched_concepts
    assert unmatched.reason.startswith("No ") and unmatched.rejected_candidates
    # Every candidate is reported with the evidence gate among its reasons (the condition
    # gate may add that "syndrome" alone does not name its condition).
    assert all(
        any("Insufficient retrieval evidence" in r for r in c["reasons"])
        for c in unmatched.rejected_candidates
    )
