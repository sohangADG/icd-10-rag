"""Orchestration of the full ingestion pipeline:

Source -> Adapter -> Normalized Records -> Structural Validation -> Hierarchy -> Rule Validation
       -> Database Import (transactional) -> Index Generation -> READY -> (optional) Embeddings
"""

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.constants import DatasetStatus
from app.core.exceptions import DatasetNotFound, DatasetNotReady
from app.indexing.embeddings import EmbeddingProvider
from app.indexing.indexer import IndexStats, SearchIndexer
from app.ingestion.adapters import SourceManifest
from app.ingestion.importer import DatasetImporter, ImportOutcome
from app.ingestion.pipeline import PipelineResult, run_pipeline
from app.repositories.dataset_repository import DatasetRepository

logger = logging.getLogger(__name__)


class IngestionService:
    def __init__(
        self, session: AsyncSession, embedding_provider: EmbeddingProvider | None = None
    ) -> None:
        self._session = session
        self._provider = embedding_provider

    @staticmethod
    def validate(path: Path, manifest: SourceManifest | None = None) -> PipelineResult:
        """Dry run: parse + validate, no database access."""
        return run_pipeline(path, manifest)

    async def ingest(
        self,
        path: Path,
        manifest: SourceManifest | None = None,
        *,
        embed: bool = False,
    ) -> tuple[PipelineResult, ImportOutcome, IndexStats | None]:
        result = run_pipeline(path, manifest)
        outcome = await DatasetImporter(self._session, SearchIndexer(self._session)).import_result(
            result
        )
        index_stats = None
        if embed and outcome.status == "imported" and outcome.dataset_id is not None:
            index_stats = await self.embed(outcome.dataset_id)
        return result, outcome, index_stats

    async def embed(
        self, dataset_id: int, *, force: bool = False, batch_size: int | None = None
    ) -> IndexStats:
        """(Re)generate embeddings of a READY dataset: only documents whose semantic text or
        embedding space changed, or all of them with `force`. Batches are committed one by one
        (restart-safe); the dataset itself is never re-imported."""
        dataset = await DatasetRepository(self._session).get(dataset_id)
        if dataset is None:
            raise DatasetNotFound(f"Dataset {dataset_id} does not exist")
        if dataset.status != DatasetStatus.READY:
            raise DatasetNotReady(f"Dataset {dataset_id} is {dataset.status.value}, not ready")
        settings = get_settings()
        indexer = SearchIndexer(self._session, self._provider)
        return await indexer.embed_documents(
            dataset,
            batch_size=batch_size or settings.embedding_batch_size,
            force=force,
            distance=settings.embedding_distance,
        )

    async def reindex(
        self,
        dataset_id: int,
        *,
        embed: bool = True,
        force: bool = False,
        batch_size: int | None = None,
    ) -> dict[str, Any]:
        """Rebuild search documents (and embeddings) of a READY dataset from relational data."""
        repository = DatasetRepository(self._session)
        dataset = await repository.get(dataset_id)
        if dataset is None:
            raise DatasetNotFound(f"Dataset {dataset_id} does not exist")
        if dataset.status != DatasetStatus.READY:
            raise DatasetNotReady(f"Dataset {dataset_id} is {dataset.status.value}, not ready")
        try:
            documents = await SearchIndexer(self._session).build_documents(dataset)
            await self._session.commit()
        except Exception:
            await self._session.rollback()
            raise
        stats = (
            await self.embed(dataset_id, force=force, batch_size=batch_size)
            if embed and self._provider
            else None
        )
        return {"documents": documents, "embeddings": asdict(stats) if stats else None}
