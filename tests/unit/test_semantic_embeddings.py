"""Deterministic tests of the semantic embedding layer (no model download, no network)."""

import math
from typing import Any

import pytest

from app.api.deps import load_provider
from app.coding.specificity import stems
from app.core.config import Settings
from app.core.exceptions import EmbeddingProviderError
from app.indexing import embeddings
from app.indexing.embeddings import (
    HashingEmbeddingProvider,
    QueryEmbeddingCache,
    SentenceTransformerProvider,
    get_embedding_provider,
    similarity_floor,
)
from app.indexing.vector_space import VectorSpace
from app.ingestion.pipeline import run_pipeline
from app.retrieval.hybrid import SemanticStatus
from app.retrieval.scoring import (
    calibrate_similarity,
    calibrate_trigram,
    hybrid_score,
    lexical_score,
)
from app.synthetic.paraphrase import PARAPHRASE_PAIRS
from tests.helpers import write_source


class FakeModel:
    """Stands in for a SentenceTransformer: deterministic, records how it was called."""

    def __init__(self, dimension: int | None = 8, output_dimension: int | None = None) -> None:
        self._dimension = dimension
        self._output = output_dimension or dimension or 8
        self.calls: list[dict[str, Any]] = []

    def get_embedding_dimension(self) -> int | None:
        return self._dimension

    def encode(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        self.calls.append({"texts": list(texts), **kwargs})
        return [[float(len(t) + i + 1) for i in range(self._output)] for t in texts]


def provider(model: FakeModel, **kwargs: Any) -> SentenceTransformerProvider:
    return SentenceTransformerProvider("fake/model", loader=lambda name, device: model, **kwargs)


# --- provider initialisation / dimension ---------------------------------------------------------


def test_real_provider_reports_identity_and_detects_dimension() -> None:
    p = provider(FakeModel(dimension=12), device="cpu")
    assert (p.provider_name, p.model, p.dimension, p.normalized) == (
        "sentence_transformers",
        "fake/model",
        12,
        True,
    )
    assert p.space == "sentence_transformers|fake/model|12|norm"


def test_dimension_is_measured_when_the_model_does_not_declare_it() -> None:
    assert provider(FakeModel(dimension=None, output_dimension=6)).dimension == 6


def test_configured_dimension_must_match_the_model() -> None:
    with pytest.raises(EmbeddingProviderError, match="produces 12-dimensional"):
        provider(FakeModel(dimension=12), expected_dimension=384)


async def test_batch_encoding_uses_the_configured_batch_size() -> None:
    model = FakeModel(dimension=4)
    vectors = await provider(model, batch_size=3).embed([f"text {i}" for i in range(7)])
    assert len(vectors) == 7
    assert [c["batch_size"] for c in model.calls] == [3]
    assert model.calls[0]["normalize_embeddings"] is False  # normalised once, uniformly


@pytest.mark.parametrize("normalize", [True, False])
async def test_normalisation_is_explicit_and_uniform(normalize: bool) -> None:
    p = provider(FakeModel(dimension=4), normalize=normalize)
    docs = await p.embed(["alpha", "beta gamma"])
    queries = await p.embed_queries(["delta"])
    norms = {round(math.sqrt(sum(v * v for v in vec)), 6) for vec in docs + queries}
    assert (norms == {1.0}) is normalize


async def test_query_prefix_applies_to_queries_only() -> None:
    model = FakeModel(dimension=4)
    p = provider(model, query_prefix="query: ")
    await p.embed(["document"])
    await p.embed_queries(["question"])
    assert model.calls[0]["texts"] == ["document"]
    assert model.calls[1]["texts"] == ["query: question"]


async def test_wrong_vector_length_is_never_truncated_or_padded() -> None:
    p = provider(FakeModel(dimension=4))
    p._model = FakeModel(dimension=4, output_dimension=5)  # model starts misbehaving
    with pytest.raises(EmbeddingProviderError, match="never truncated or padded"):
        await p.embed(["x"])


async def test_non_finite_vectors_are_rejected() -> None:
    class NaNModel(FakeModel):
        def encode(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
            return [[float("nan")] * 4 for _ in texts]

    with pytest.raises(EmbeddingProviderError, match="non-finite"):
        await provider(NaNModel(dimension=4)).embed(["x"])


async def test_empty_input_needs_no_model_call() -> None:
    model = FakeModel(dimension=4)
    assert await provider(model).embed([]) == []
    assert model.calls == []


# --- query cache ----------------------------------------------------------------------------------


async def test_query_cache_hits_and_never_stores_raw_text() -> None:
    model = FakeModel(dimension=4)
    p = provider(model)
    p.enable_query_cache(2)
    first = await p.embed_query("Acute Airway  Infection")
    second = await p.embed_query("acute airway infection")  # same after normalisation
    assert first == second and len(model.calls) == 1
    cache = p.query_cache
    assert cache is not None and (cache.hits, cache.misses) == (1, 1)
    assert all("airway" not in key for key in cache._items)  # keys are SHA-256 hashes


def test_query_cache_is_bounded() -> None:
    cache = QueryEmbeddingCache(2)
    for i in range(5):
        cache.put(QueryEmbeddingCache.key("s", f"q{i}"), [float(i)])
    assert len(cache) == 2
    assert cache.get(QueryEmbeddingCache.key("s", "q0")) is None
    assert cache.get(QueryEmbeddingCache.key("s", "q4")) == [4.0]
    assert QueryEmbeddingCache.key("space-a", "q") != QueryEmbeddingCache.key("space-b", "q")


# --- factory / configuration ----------------------------------------------------------------------


def test_provider_swapping_is_configuration_only() -> None:
    fake = FakeModel(dimension=16)
    real = get_embedding_provider(
        Settings(
            _env_file=None, embedding_provider="sentence-transformers", embedding_dimension=None
        ),
        loader=lambda name, device: fake,
    )
    assert real is not None and real.provider_name == "sentence_transformers"
    assert real.dimension == 16 and real.query_cache is not None
    hashing = get_embedding_provider(Settings(_env_file=None, embedding_provider="hashing"))
    assert isinstance(hashing, HashingEmbeddingProvider) and hashing.model == "hashing-v1-256"
    assert get_embedding_provider(Settings(_env_file=None, embedding_provider="none")) is None


def test_openai_requires_an_explicit_dimension() -> None:
    with pytest.raises(EmbeddingProviderError, match="EMBEDDING_DIMENSION"):
        get_embedding_provider(
            Settings(_env_file=None, embedding_provider="openai", embedding_dimension=None)
        )


def test_failed_provider_load_is_reported_not_raised(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(model: str, device: str) -> Any:
        raise EmbeddingProviderError("model download failed")

    monkeypatch.setattr(embeddings, "_load_sentence_transformer", broken)
    settings = Settings(_env_file=None, embedding_provider="sentence_transformers")
    handle = load_provider(settings)
    assert handle.provider is None and handle.status is SemanticStatus.PROVIDER_UNAVAILABLE
    assert load_provider(settings) is handle  # loaded once per settings object
    disabled = load_provider(Settings(_env_file=None, embedding_provider="none"))
    assert disabled.status is SemanticStatus.DISABLED


def test_similarity_floor_defaults_and_override() -> None:
    p = provider(FakeModel(dimension=4))
    assert similarity_floor(Settings(_env_file=None), p) == 0.6
    assert similarity_floor(Settings(_env_file=None, embedding_similarity_floor=0.2), p) == 0.2
    assert similarity_floor(Settings(_env_file=None), HashingEmbeddingProvider()) == 0.0


# --- vector space: index/operator consistency, isolation ------------------------------------------


@pytest.mark.parametrize(
    ("distance", "operator", "opclass"),
    [
        ("cosine", "<=>", "vector_cosine_ops"),
        ("l2", "<->", "vector_l2_ops"),
        ("inner_product", "<#>", "vector_ip_ops"),
    ],
)
def test_index_operator_class_always_matches_the_query_operator(
    distance: str, operator: str, opclass: str
) -> None:
    space = VectorSpace("sentence_transformers", "m", 384, True, distance)  # type: ignore[arg-type]
    assert opclass in space.create_index_sql()
    assert operator in space.distance_sql() and operator in space.similarity_sql()
    for other_op in {"<=>", "<->", "<#>"} - {operator}:
        assert other_op not in space.distance_sql()


def test_spaces_are_isolated_by_provider_model_dimension_and_normalisation() -> None:
    base = VectorSpace("sentence_transformers", "m", 384, True)
    variants = [
        VectorSpace("hashing", "m", 384, True),
        VectorSpace("sentence_transformers", "m2", 384, True),
        VectorSpace("sentence_transformers", "m", 256, True),
        VectorSpace("sentence_transformers", "m", 384, False),
        VectorSpace("sentence_transformers", "m", 384, True, "l2"),
    ]
    assert len({v.index_name for v in [base, *variants]}) == 6
    predicate = base.predicate_sql()
    for column in (
        "embedding_provider",
        "embedding_model",
        "embedding_dimension",
        "embedding_normalized",
    ):
        assert column in predicate
    assert predicate in base.create_index_sql()  # same predicate for index and queries


def test_space_literals_are_escaped_and_dimensions_validated() -> None:
    space = VectorSpace("p", "evil'model", 8, True)
    assert "'evil''model'" in space.predicate_sql()
    with pytest.raises(ValueError):
        VectorSpace("p", "m", 0, True)
    with pytest.raises(ValueError):
        VectorSpace("p", "m", 8, True, "manhattan")  # type: ignore[arg-type]


# --- score normalisation ---


def test_lexical_score_is_absolute() -> None:
    assert lexical_score(0.5, 1.0) == 0.875
    assert lexical_score(0.0, 0.5) == 0.375
    assert lexical_score(2.0, -1.0) == 0.25  # inputs clamped


def test_semantic_and_trigram_calibration() -> None:
    assert calibrate_similarity(0.6, 0.6) == 0.0
    assert calibrate_similarity(0.8, 0.6) == 0.5
    assert calibrate_similarity(1.0, 0.6) == 1.0
    assert calibrate_similarity(0.2, 0.6) == 0.0
    assert calibrate_trigram(0.125, 0.3) == 0.0  # trigram noise scores nothing
    assert calibrate_trigram(1.0, 0.3) == 1.0
    with pytest.raises(ValueError):
        calibrate_similarity(0.5, 1.0)


def test_no_component_dominates_by_raw_scale() -> None:
    weights = {
        "exact": 0.3,
        "lexical": 0.25,
        "fuzzy": 0.2,
        "semantic": 0.15,
        "hierarchy": 0.05,
        "index_term": 0.05,
    }
    # A raw cosine of 0.62 (unrelated, typical for BGE) must not beat a real lexical match.
    noise = {"semantic": calibrate_similarity(0.62, 0.6), "fuzzy": calibrate_trigram(0.2, 0.3)}
    lexical_match = {"lexical": lexical_score(0.3, 1.0), "semantic": 0.0}
    assert hybrid_score(noise, weights) < hybrid_score(lexical_match, weights)
    # A strong semantic match outranks trigram noise.
    strong = {"semantic": calibrate_similarity(0.8, 0.6), "fuzzy": 0.0}
    assert hybrid_score(strong, weights) > hybrid_score(
        {"fuzzy": calibrate_trigram(0.25, 0.3), "semantic": 0.0}, weights
    )
    assert hybrid_score({"semantic": None, "lexical": 1.0}, weights) == pytest.approx(
        0.25 / 0.85, abs=1e-4
    )


# --- paraphrase dataset ---------------------------------------------------------------------------


def test_paraphrase_queries_share_no_word_with_their_target() -> None:
    for code, title, query in PARAPHRASE_PAIRS:
        assert not (stems(title) & stems(query)), (code, title, query)


def test_paraphrase_dataset_is_valid(tmp_path) -> None:  # noqa: ANN001
    from app.ingestion.adapters import SourceManifest
    from app.synthetic.paraphrase import DATASET_PARAPHRASE, paraphrase_chapters
    from app.synthetic.renderers import write_json

    path = tmp_path / "p.json"
    manifest = write_json(path, DATASET_PARAPHRASE, paraphrase_chapters())
    result = run_pipeline(path, SourceManifest.model_validate(manifest))
    assert result.is_valid and result.statistics()["categories"] == len(PARAPHRASE_PAIRS)
    assert write_source  # helpers stay importable for the integration suite
