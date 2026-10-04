"""Import synthetic sources into the (rolled-back) test transaction."""

from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.indexing.embeddings import get_embedding_provider
from app.indexing.indexer import SearchIndexer
from app.ingestion.importer import DatasetImporter, ImportOutcome
from app.ingestion.pipeline import run_pipeline
from app.models import IcdDataset
from tests.helpers import write_source


async def import_synthetic(
    session: AsyncSession,
    directory: Path,
    fmt: str = "json",
    version: str = "2024",
    *,
    embed: bool = True,
) -> ImportOutcome:
    path, manifest = write_source(directory, fmt, version)
    outcome = await DatasetImporter(session).import_result(run_pipeline(path, manifest))
    if embed and outcome.status == "imported":
        dataset = await session.get(IcdDataset, outcome.dataset_id)
        assert dataset is not None
        await SearchIndexer(session, get_embedding_provider(get_settings())).embed_documents(
            dataset
        )
        await session.commit()
    return outcome
