"""Real sentence-transformers model against PostgreSQL + pgvector (opt-in).

Run with RUN_SEMANTIC_MODEL_TESTS=1 (the default model is downloaded into HF_HOME once).
Uses the original synthetic paraphrase dataset: every query shares no word with its target.
"""

from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.constants import CLASSIFICATION_NODE_TYPES
from app.indexing.embeddings import EmbeddingProvider, get_embedding_provider
from app.indexing.indexer import SearchIndexer
from app.indexing.vector_space import VectorSpace
from app.ingestion.adapters import SourceManifest
from app.ingestion.importer import DatasetImporter
from app.ingestion.pipeline import run_pipeline
from app.models import IcdDataset
from app.repositories.icd_repository import IcdRepository
from app.repositories.search_repository import SearchRepository
from app.retrieval.hybrid import HybridRetriever, RetrievalFilters, SemanticStatus
from app.retrieval.scoring import calibrate_similarity
from app.synthetic.paraphrase import DATASET_PARAPHRASE, PARAPHRASE_PAIRS, paraphrase_chapters
from app.synthetic.renderers import write_json

pytestmark = pytest.mark.semantic_model

MODEL = "BAAI/bge-small-en-v1.5"


def _settings(**overrides: object) -> Settings:
    return Settings(
        _env_file=None,
        embedding_provider="sentence_transformers",
        embedding_model=MODEL,
        embedding_dimension=None,
        **overrides,
    )


@pytest.fixture(scope="module")
def real_provider() -> EmbeddingProvider:
    provider = get_embedding_provider(_settings())
    assert provider is not None
    return provider


@pytest.fixture
async def paraphrase_dataset(
    session: AsyncSession, tmp_path: Path, real_provider: EmbeddingProvider
) -> IcdDataset:
    path = tmp_path / "paraphrase.json"
    manifest = write_json(path, DATASET_PARAPHRASE, paraphrase_chapters())
    outcome = await DatasetImporter(session).import_result(
        run_pipeline(path, SourceManifest.model_validate(manifest))
    )
    dataset = await session.get(IcdDataset, outcome.dataset_id)
    assert dataset is not None
    stats = await SearchIndexer(session, real_provider).embed_documents(dataset, batch_size=8)
    assert (stats.embedded, stats.failed, stats.dimension) == (17, 0, 384)
    return dataset


def test_real_provider_detects_its_dimension(real_provider: EmbeddingProvider) -> None:
    assert real_provider.provider_name == "sentence_transformers"
    assert real_provider.model == MODEL
    assert real_provider.dimension == 384  # read from the model, not configured


async def test_real_vectors_are_normalised(real_provider: EmbeddingProvider) -> None:
    vectors = await real_provider.embed(["high blood pressure", "broken thigh bone"])
    assert all(len(v) == 384 for v in vectors)
    assert all(abs(sum(x * x for x in v) - 1.0) < 1e-4 for v in vectors)


async def test_semantic_search_beyond_word_overlap(
    session: AsyncSession, paraphrase_dataset: IcdDataset, real_provider: EmbeddingProvider
) -> None:
    """query -> embed -> pgvector similarity (dataset + space filter) -> ranked DB records."""
    search, records = SearchRepository(session), IcdRepository(session)
    space = VectorSpace.of(real_provider)
    hits_at_3 = 0
    for code, _title, query in PARAPHRASE_PAIRS:
        vector = await real_provider.embed_query(query)
        similarities = await search.semantic(paraphrase_dataset.id, space, vector, limit=10)
        nodes = await records.get_many(paraphrase_dataset.id, list(similarities))
        ranked = [
            {
                "semantic_score": calibrate_similarity(similarities[node_id], 0.6),
                "similarity": round(similarities[node_id], 4),
                "record_id": node_id,
                "code": nodes[node_id].code,
                "dataset_id": nodes[node_id].dataset_id,
                "model": space.model,
            }
            for node_id in sorted(similarities, key=lambda n: -similarities[n])
            if nodes[node_id].node_type in CLASSIFICATION_NODE_TYPES
        ]
        assert all(hit["dataset_id"] == paraphrase_dataset.id for hit in ranked)
        hits_at_3 += code in [hit["code"] for hit in ranked[:3]]
    assert hits_at_3 / len(PARAPHRASE_PAIRS) >= 0.85


async def test_hybrid_with_real_semantic_finds_paraphrases(
    session: AsyncSession, paraphrase_dataset: IcdDataset, real_provider: EmbeddingProvider
) -> None:
    settings = _settings()
    filters = RetrievalFilters(node_types=CLASSIFICATION_NODE_TYPES)
    lexical_top1 = semantic_top1 = 0
    for code, _title, query in PARAPHRASE_PAIRS:
        lexical = await HybridRetriever(session, settings, None).retrieve(
            paraphrase_dataset, [query], top_k=3, filters=filters
        )
        hybrid = await HybridRetriever(session, settings, real_provider).retrieve(
            paraphrase_dataset, [query], top_k=3, filters=filters
        )
        assert hybrid.semantic_status is SemanticStatus.OK
        assert hybrid.embedding_space["model"] == MODEL
        lexical_top1 += bool(lexical.candidates) and lexical.candidates[0].node.code == code
        semantic_top1 += hybrid.candidates[0].node.code == code
    assert semantic_top1 >= 0.85 * len(PARAPHRASE_PAIRS)
    assert semantic_top1 > lexical_top1 + 0.5 * len(PARAPHRASE_PAIRS)


async def test_near_tie_semantic_match_is_not_suggested(
    session: AsyncSession, paraphrase_dataset: IcdDataset, real_provider: EmbeddingProvider
) -> None:
    """bge-small scores 'insomnia' ~equally against two records (cosine 0.648 vs 0.643): the
    weak, near-floor evidence must not become a confident wrong code."""
    from app.coding.suggestion_service import SuggestionService
    from app.schemas.icd import SuggestRequest

    request = SuggestRequest(
        clinical_note="Assessment: insomnia.",
        coding_system="SYNTH-ICD",
        version="paraphrase-1",
    )
    response = await SuggestionService(session, _settings(), real_provider).suggest(request)
    assert "P11" not in [s.code for s in response.suggestions]
