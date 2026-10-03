from collections.abc import AsyncIterator, Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

import app.main as main_module
from app.core.database import get_session


class _UnreachableSession:
    async def execute(self, *args: object, **kwargs: object) -> None:
        raise OperationalError("SELECT 1", {}, ConnectionRefusedError("refused"))


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """App with its full lifespan, but without contacting a real database."""

    async def skip_database_probe() -> None:
        return None

    monkeypatch.setattr(main_module, "verify_database", skip_database_probe)
    with TestClient(main_module.create_app()) as test_client:
        yield test_client


def test_application_starts_and_health_returns_ok(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "icd-rag-service"}


def test_db_health_reports_unavailable_database(client: TestClient) -> None:
    async def unreachable_session() -> AsyncIterator[_UnreachableSession]:
        yield _UnreachableSession()

    client.app.dependency_overrides[get_session] = unreachable_session

    response = client.get("/health/db")

    assert response.status_code == 503
    assert response.json() == {"status": "error", "database": "unavailable", "pgvector": False}


def test_suggestion_api_is_not_exposed(client: TestClient) -> None:
    # The OpenAPI document is the public route contract; it does not depend on the internal
    # route classes FastAPI uses (which vary between versions).
    paths = set(client.get("/openapi.json").json()["paths"])

    assert paths == {"/health", "/health/db"}
    assert client.post("/api/v1/icd/suggest", json={}).status_code == 404
