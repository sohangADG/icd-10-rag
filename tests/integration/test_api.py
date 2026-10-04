"""HTTP API against the real PostgreSQL test database (ASGI transport, session override)."""

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import SecretStr
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.core.constants import DatasetStatus
from app.core.database import get_session
from app.main import create_app
from app.repositories.dataset_repository import DatasetRepository
from tests.integration.ingest import import_synthetic

DATASET = {"coding_system": "SYNTH-ICD", "version": "2024"}


@pytest.fixture
async def client(session: AsyncSession, tmp_path: Path) -> AsyncIterator[AsyncClient]:
    outcome = await import_synthetic(session, tmp_path)
    app = create_app()

    async def test_session() -> AsyncIterator[AsyncSession]:
        yield session

    app.dependency_overrides[get_session] = test_session
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as http:
        http.dataset_id = outcome.dataset_id  # type: ignore[attr-defined]
        http.app = app  # type: ignore[attr-defined]
        yield http


async def test_dataset_endpoints(client: AsyncClient) -> None:
    listing = (
        await client.get("/api/v1/icd/datasets", params={"coding_system": "synth-icd"})
    ).json()
    assert [d["version"] for d in listing["datasets"]] == ["2024"]
    detail = (await client.get(f"/api/v1/icd/datasets/{client.dataset_id}")).json()
    assert detail["status"] == "ready" and detail["record_counts"]["SUBCATEGORY"] == 27
    assert detail["licence"]["basis"] == "synthetic"
    assert detail["latest_ingestion"]["status"] == "completed"
    missing = await client.get("/api/v1/icd/datasets/999999")
    assert missing.status_code == 404 and missing.json()["error"]["code"] == "DATASET_NOT_FOUND"


async def test_search_modes(client: AsyncClient) -> None:
    exact = (
        await client.get("/api/v1/icd/search", params={**DATASET, "q": "a01.0", "mode": "exact"})
    ).json()
    assert [r["code"] for r in exact["results"]] == ["A01.0"]
    text_mode = (
        await client.get(
            "/api/v1/icd/search", params={**DATASET, "q": "hypertension", "mode": "text"}
        )
    ).json()
    assert text_mode["results"][0]["code"] == "B00" and not text_mode["semantic_available"]
    hybrid = (
        await client.get(
            "/api/v1/icd/search", params={**DATASET, "q": "acute heart failure", "limit": 3}
        )
    ).json()
    assert hybrid["semantic_available"] and hybrid["results"][0]["code"] == "B01.0"
    assert len(hybrid["results"]) == 3
    assert hybrid["results"][0]["hierarchy"][-1]["code"] == "B01"
    level = (
        await client.get(
            "/api/v1/icd/search", params={**DATASET, "q": "airway", "level": "CHAPTER"}
        )
    ).json()
    assert {r["level"] for r in level["results"]} == {"CHAPTER"}


async def test_code_endpoints(client: AsyncClient) -> None:
    record = (await client.get("/api/v1/icd/codes/A00", params=DATASET)).json()
    assert record["title"] == "Airway infection" and record["level"] == "CATEGORY"
    assert record["inclusions"] == ["bronchial passage infection"]
    assert record["exclusions"][0]["target_code"] == "B15"
    assert record["exclusions"][0]["target_record_id"] is not None
    assert record["synonyms"] == ["respiratory tract infection"]
    assert record["index_terms"] == ["chest infection"]
    children = (await client.get("/api/v1/icd/codes/A00/children", params=DATASET)).json()
    assert [r["code"] for r in children["records"]] == ["A00.0", "A00.1", "A00.9"]
    ancestors = (await client.get("/api/v1/icd/codes/C00.20/ancestors", params=DATASET)).json()
    assert [r["code"] for r in ancestors["records"]] == ["III", "C00-C09", "C00", "C00.2"]
    by_id = await client.get("/api/v1/icd/codes/A00.0", params={"dataset_id": client.dataset_id})
    assert by_id.status_code == 200
    missing = await client.get("/api/v1/icd/codes/Z99.9", params=DATASET)
    assert missing.status_code == 404 and missing.json()["error"]["code"] == "INVALID_ICD_CODE"


async def test_suggest_endpoint(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/icd/suggest",
        json={
            "clinical_note": "Diagnosis: lobar consolidation of the left lung. Denies cough.",
            **DATASET,
            "top_k": 3,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert [s["code"] for s in body["suggestions"]] == ["A01.0"]
    suggestion = body["suggestions"][0]
    assert set(suggestion) == {
        "clinical_concept",
        "concept_status",
        "concept_type",
        "code",
        "title",
        "record_id",
        "evidence",
        "icd_reference",
        "alternatives",
        "missing_information",
        "confidence",
        "retrieval_scores",
        "validation",
        "source_provenance",
    }
    assert body["unmatched_concepts"][0]["concept_status"] == "negated"
    assert "not diagnostic certainty" in body["confidence_note"]
    assert response.headers["x-request-id"]


@pytest.mark.parametrize(
    ("payload", "status", "code"),
    [
        ({"coding_system": "ICD-99", "version": "1"}, 404, "UNSUPPORTED_CODING_SYSTEM"),
        ({"coding_system": "SYNTH-ICD", "version": "1999"}, 404, "UNSUPPORTED_DATASET_VERSION"),
        ({**DATASET, "dataset_id": 999999}, 404, "DATASET_NOT_FOUND"),
    ],
)
async def test_suggest_dataset_errors(
    client: AsyncClient, payload: dict, status: int, code: str
) -> None:
    response = await client.post("/api/v1/icd/suggest", json={"clinical_note": "AKI", **payload})
    assert response.status_code == status and response.json()["error"]["code"] == code


async def test_dataset_not_ready_is_409(client: AsyncClient, session: AsyncSession) -> None:
    await DatasetRepository(session).set_status(client.dataset_id, DatasetStatus.PROCESSING)
    await session.flush()
    session.expire_all()
    response = await client.post("/api/v1/icd/suggest", json={"clinical_note": "AKI", **DATASET})
    assert response.status_code == 409 and response.json()["error"]["code"] == "DATASET_NOT_READY"


async def test_admin_reindex_and_archive_with_token(client: AsyncClient) -> None:
    client.app.dependency_overrides[get_settings] = lambda: Settings(
        _env_file=None, admin_api_token=SecretStr("adm-token-1")
    )
    headers = {"X-Admin-Token": "adm-token-1"}
    reindex = await client.post(
        f"/api/v1/admin/datasets/{client.dataset_id}/reindex", headers=headers
    )
    assert reindex.status_code == 200 and reindex.json()["documents"] == 55
    archived = await client.post(
        f"/api/v1/admin/datasets/{client.dataset_id}/archive", headers=headers
    )
    assert archived.json()["status"] == "archived"
    client.app.dependency_overrides.pop(get_settings)
    gone = await client.get("/api/v1/icd/search", params={**DATASET, "q": "airway"})
    assert gone.status_code == 409  # archived datasets are no longer served
