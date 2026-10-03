from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


class HealthRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def ping(self) -> None:
        await self._session.execute(text("SELECT 1"))

    async def pgvector_version(self) -> str | None:
        """Installed version of the `vector` extension in this database, or None if absent."""
        result = await self._session.execute(
            text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
        )
        return result.scalar_one_or_none()
