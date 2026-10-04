import json
import logging
from collections.abc import Iterator

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

import app.main as main_module
from app.core.config import Settings, get_settings
from app.core.exceptions import EmbeddingProviderError
from app.core.logging import JsonFormatter, RequestIdFilter, request_id_var
from app.evaluation import metrics
from app.evaluation.models import load_cases
from app.indexing.embeddings import (
    HashingEmbeddingProvider,
    OpenAIEmbeddingProvider,
    get_embedding_provider,
)

# --- embeddings ----------------------------------------------------------------------------------


async def test_hashing_provider_is_deterministic_and_normalised() -> None:
    provider = HashingEmbeddingProvider(dimension=64)
    first, again, other = await provider.embed(["airway infection", "airway infection", "kidney"])
    assert first == again and first != other
    assert len(first) == 64 and provider.model == "hashing-v1-64"
    assert sum(v * v for v in first) == pytest.approx(1.0)
    assert not provider.remote


async def test_hashing_provider_captures_subword_similarity() -> None:
    provider = HashingEmbeddingProvider(dimension=256)
    query, typo, unrelated = await provider.embed(
        ["chronic airway infection", "chronic airway infecton", "stage 3 kidney impairment"]
    )

    def cosine(a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b, strict=True))

    assert cosine(query, typo) > cosine(query, unrelated)


async def test_openai_provider_success_and_errors_never_leak_the_key() -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers["Authorization"]
        body = json.loads(request.content)
        data = [{"index": i, "embedding": [1.0, 0.0, 0.0]} for i in range(len(body["input"]))]
        return httpx.Response(200, json={"data": list(reversed(data))})

    provider = OpenAIEmbeddingProvider(
        model="m",
        dimension=3,
        api_key="sk-secret",
        base_url="https://example.test/v1",
        transport=httpx.MockTransport(handler),
    )
    vectors = await provider.embed(["a", "b"])
    assert vectors == [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]] and seen["auth"] == "Bearer sk-secret"
    assert provider.remote

    failing = OpenAIEmbeddingProvider(
        model="m",
        dimension=3,
        api_key="sk-secret",
        base_url="https://example.test/v1",
        transport=httpx.MockTransport(lambda r: httpx.Response(500, json={})),
    )
    with pytest.raises(EmbeddingProviderError) as error:
        await failing.embed(["a"])
    assert "sk-secret" not in str(error.value) and "500" in str(error.value)

    wrong_dimension = OpenAIEmbeddingProvider(
        model="m",
        dimension=4,
        api_key="k",
        base_url="https://example.test/v1",
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(EmbeddingProviderError, match="dimension"):
        await wrong_dimension.embed(["a"])


def test_provider_factory_reads_configuration() -> None:
    assert get_embedding_provider(Settings(_env_file=None, embedding_provider="none")) is None
    hashing = get_embedding_provider(Settings(_env_file=None, embedding_dimension=32))
    assert isinstance(hashing, HashingEmbeddingProvider) and hashing.dimension == 32
    with pytest.raises(EmbeddingProviderError, match="EMBEDDING_API_KEY"):
        get_embedding_provider(Settings(_env_file=None, embedding_provider="openai"))
    remote = get_embedding_provider(
        Settings(_env_file=None, embedding_provider="openai", embedding_api_key=SecretStr("k"))
    )
    assert remote is not None and remote.remote


# --- evaluation metrics ---------------------------------------------------------------------------


def test_ranking_metrics() -> None:
    ranks = [1, 2, None, 4]
    assert metrics.rank_of("B", ["A", "B"]) == 2 and metrics.rank_of("C", ["A"]) is None
    assert metrics.recall_at_k(ranks, 1) == 0.25
    assert metrics.recall_at_k(ranks, 3) == 0.5
    assert metrics.mean_reciprocal_rank(ranks) == pytest.approx((1 + 0.5 + 0.25) / 4)
    assert metrics.recall_at_k([], 3) == 0.0 and metrics.rate(1, 0) == 0.0


def test_hierarchy_and_agreement_metrics() -> None:
    ancestors = {"A01.0": {"A01", "A00-A09"}, "A01": {"A00-A09"}}
    assert metrics.hierarchy_related("A01", "A01.0", ancestors)
    assert metrics.hierarchy_related("A01.0", "A01", ancestors)
    assert not metrics.hierarchy_related("A02", "A01.0", ancestors)
    assert metrics.agreement([("A", "A"), ("B", "C"), ("D", None)]) == 0.5


def test_case_loader_jsonl_and_csv(tmp_path) -> None:  # noqa: ANN001
    jsonl = tmp_path / "c.jsonl"
    jsonl.write_text(
        '{"clinical_note": "x", "expected_code": "A00", "expected_dataset": "S", '
        '"expected_version": "1"}\n\n'
    )
    csv_path = tmp_path / "c.csv"
    csv_path.write_text(
        "clinical_note,expected_code,expected_dataset,expected_version,notes\nx,,S,1,neg\n"
    )
    (case,) = load_cases(jsonl)
    assert case.case_id == "line-1" and case.expected_code == "A00"
    (negative,) = load_cases(csv_path)
    assert negative.expected_code is None and negative.notes == "neg"


# --- API (no database) ---------------------------------------------------------------------------


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    async def skip_probe() -> None:
        return None

    monkeypatch.setattr(main_module, "verify_database", skip_probe)
    with TestClient(main_module.create_app()) as test_client:
        yield test_client


def test_invalid_suggest_requests_are_rejected_without_echoing_the_note(client: TestClient) -> None:
    secret_note = "Patient John Doe, MRN 12345, has acute airway infection"
    cases = [
        {"clinical_note": "", "coding_system": "SYNTH-ICD", "version": "2024"},
        {"clinical_note": "   ", "coding_system": "SYNTH-ICD", "version": "2024"},
        {"clinical_note": "\n\t ", "coding_system": "SYNTH-ICD", "version": "2024"},
        {"clinical_note": secret_note, "coding_system": "SYNTH-ICD"},
        {"clinical_note": secret_note, "coding_system": "S", "version": "1", "top_k": 0},
        {"clinical_note": secret_note, "coding_system": "S", "version": "1", "extra": 1},
    ]
    for body in cases:
        response = client.post("/api/v1/icd/suggest", json=body)
        assert response.status_code == 422
        payload = response.json()
        assert payload["error"]["code"] == "REQUEST_VALIDATION_ERROR"
        assert "John Doe" not in response.text and payload["error"]["request_id"]


def test_request_id_is_propagated_and_sanitised(client: TestClient) -> None:
    response = client.get("/health", headers={"X-Request-ID": "abc-123"})
    assert response.headers["x-request-id"] == "abc-123"
    generated = client.get("/health", headers={"X-Request-ID": "bad id with spaces\n"})
    assert generated.headers["x-request-id"] != "bad id with spaces\n"
    assert len(generated.headers["x-request-id"]) == 32


def test_admin_endpoints_are_disabled_without_a_token(client: TestClient) -> None:
    response = client.post("/api/v1/admin/datasets/1/archive")
    assert response.status_code == 404 and response.json()["error"]["code"] == "ADMIN_DISABLED"


def test_admin_endpoints_require_the_configured_token(client: TestClient) -> None:
    client.app.dependency_overrides[get_settings] = lambda: Settings(
        _env_file=None, admin_api_token=SecretStr("s3cret")
    )
    assert client.post("/api/v1/admin/datasets/1/archive").status_code == 401
    wrong = client.post("/api/v1/admin/datasets/1/archive", headers={"X-Admin-Token": "nope"})
    assert wrong.status_code == 401 and wrong.json()["error"]["code"] == "UNAUTHORIZED"


def test_search_limits_are_bounded(client: TestClient) -> None:
    assert client.get("/api/v1/icd/search", params={"q": "x", "limit": 1000}).status_code == 422
    assert client.get("/api/v1/icd/search", params={"q": ""}).status_code == 422
    assert client.get("/api/v1/icd/search", params={"q": "x" * 201}).status_code == 422


# --- config / logging ----------------------------------------------------------------------------


def test_clinical_text_logging_is_forbidden_in_production() -> None:
    with pytest.raises(ValueError, match="LOG_CLINICAL_TEXT"):
        Settings(_env_file=None, app_env="production", log_clinical_text=True)
    assert Settings(_env_file=None, app_env="local", log_clinical_text=True).log_clinical_text


def test_secrets_are_not_rendered() -> None:
    settings = Settings(
        _env_file=None,
        embedding_api_key=SecretStr("sk-zq81"),
        admin_api_token=SecretStr("adm-77xq"),
    )
    assert "sk-zq81" not in repr(settings) and "adm-77xq" not in repr(settings)
    assert set(settings.retrieval_weights()) == {
        "exact",
        "lexical",
        "fuzzy",
        "semantic",
        "hierarchy",
        "index_term",
    }


def test_request_id_filter_adds_correlation_id() -> None:
    record = logging.makeLogRecord({"name": "t", "levelname": "INFO", "msg": "m"})
    token = request_id_var.set("rid-1")
    try:
        RequestIdFilter().filter(record)
    finally:
        request_id_var.reset(token)
    assert json.loads(JsonFormatter().format(record))["request_id"] == "rid-1"
