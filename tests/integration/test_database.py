from collections.abc import AsyncIterator

from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_session
from app.main import create_app
from app.repositories.health_repository import HealthRepository


async def test_database_connectivity(session: AsyncSession) -> None:
    assert (await session.execute(text("SELECT 1"))).scalar_one() == 1
    server_version = (await session.execute(text("SHOW server_version_num"))).scalar_one()
    # NULLS NOT DISTINCT (dataset identity constraint) requires PostgreSQL 15+.
    assert int(server_version) >= 150000


async def test_pgvector_extension_is_installed(session: AsyncSession) -> None:
    assert await HealthRepository(session).pgvector_version() is not None
    # The vector type and its distance operator are usable (no embeddings are stored).
    distance = (
        await session.execute(text("SELECT '[0,0]'::vector <-> '[3,4]'::vector"))
    ).scalar_one()
    assert distance == 5.0


async def test_db_health_endpoint_against_real_database(session: AsyncSession) -> None:
    app = create_app()

    async def test_session() -> AsyncIterator[AsyncSession]:
        yield session

    app.dependency_overrides[get_session] = test_session
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/health/db")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "database": "connected", "pgvector": True}
