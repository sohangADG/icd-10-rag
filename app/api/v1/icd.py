"""Public ICD API: datasets, search, code lookup, hierarchy traversal and suggestions.

Every endpoint is scoped to exactly one dataset (resolved from dataset_id or
coding_system + version [+ country, language]). Results are bounded (limit <= MAX_TOP_K).
"""

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import embedding_provider, rate_limit_hook
from app.coding.suggestion_service import SuggestionService
from app.core.config import Settings, get_settings
from app.core.constants import DatasetStatus, NodeType
from app.core.database import get_session
from app.core.exceptions import DatasetNotFound, InvalidICDCodeError
from app.indexing.embeddings import EmbeddingProvider
from app.models import IcdDataset, IcdNode
from app.repositories.dataset_repository import DatasetRepository
from app.repositories.icd_repository import IcdRepository
from app.repositories.ingestion_repository import IngestionRepository
from app.retrieval.hybrid import HybridRetriever, RetrievalFilters
from app.schemas.icd import (
    DatasetDetail,
    DatasetList,
    ICDRecordOut,
    MatchedTerm,
    RecordListResponse,
    SearchHit,
    SearchResponse,
    SuggestRequest,
    SuggestResponse,
)
from app.services.dataset_resolver import DatasetResolver
from app.services.presenters import dataset_summary, hierarchy_item, record_out

router = APIRouter(prefix="/api/v1/icd", tags=["icd"], dependencies=[Depends(rate_limit_hook)])

SessionDep = Annotated[AsyncSession, Depends(get_session)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
ProviderDep = Annotated[EmbeddingProvider | None, Depends(embedding_provider)]


class DatasetSelector:
    """Query parameters selecting one dataset."""

    def __init__(
        self,
        coding_system: Annotated[str | None, Query(max_length=64)] = None,
        version: Annotated[str | None, Query(max_length=32)] = None,
        country: Annotated[str | None, Query(max_length=8)] = None,
        language: Annotated[str | None, Query(max_length=16)] = None,
        dataset_id: Annotated[int | None, Query(ge=1)] = None,
    ) -> None:
        self.coding_system = coding_system
        self.version = version
        self.country = country
        self.language = language
        self.dataset_id = dataset_id

    async def resolve(self, session: AsyncSession) -> IcdDataset:
        return await DatasetResolver(session).resolve(
            dataset_id=self.dataset_id,
            coding_system=self.coding_system,
            version=self.version,
            country=self.country,
            language=self.language,
        )


SelectorDep = Annotated[DatasetSelector, Depends()]


# --- datasets ---------------------------------------------------------------------------------


@router.get("/datasets", response_model=DatasetList)
async def list_datasets(
    session: SessionDep,
    coding_system: Annotated[str | None, Query(max_length=64)] = None,
    version: Annotated[str | None, Query(max_length=32)] = None,
    country: Annotated[str | None, Query(max_length=8)] = None,
    language: Annotated[str | None, Query(max_length=16)] = None,
    status: DatasetStatus | None = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> DatasetList:
    rows = await DatasetRepository(session).list_datasets(
        system=coding_system,
        version=version,
        country=country,
        language=language,
        statuses=[status] if status else None,
        limit=limit,
        offset=offset,
    )
    return DatasetList(datasets=[dataset_summary(d) for d in rows])


@router.get("/datasets/{dataset_id}", response_model=DatasetDetail)
async def get_dataset(dataset_id: int, session: SessionDep) -> DatasetDetail:
    dataset = await DatasetRepository(session).get(dataset_id)
    if dataset is None:
        raise DatasetNotFound(f"Dataset {dataset_id} does not exist")
    run = await IngestionRepository(session).latest_run(dataset_id)
    return DatasetDetail(
        **dataset_summary(dataset).model_dump(),
        record_counts=await IcdRepository(session).count_by_type(dataset_id),
        licence=(dataset.metadata_ or {}).get("licence", {}),
        latest_ingestion={
            "run_id": str(run.run_id),
            "status": run.status.value,
            "started_at": run.started_at,
            "completed_at": run.completed_at,
            "records_seen": run.records_seen,
            "records_inserted": run.records_inserted,
            "error_summary": run.error_summary,
        }
        if run
        else None,
    )


# --- search -----------------------------------------------------------------------------------


@router.get("/search", response_model=SearchResponse)
async def search(
    session: SessionDep,
    settings: SettingsDep,
    provider: ProviderDep,
    selector: SelectorDep,
    q: Annotated[str, Query(min_length=1, max_length=200)],
    mode: Literal["exact", "text", "hybrid"] = "hybrid",
    limit: Annotated[int, Query(ge=1, le=100)] = 10,
    level: NodeType | None = None,
    selectable_only: bool = False,
) -> SearchResponse:
    dataset = await selector.resolve(session)
    limit = min(limit, settings.max_top_k)
    records = IcdRepository(session)
    if mode == "exact":
        node = await records.get_by_code(dataset.id, q)
        ancestors = await records.ancestors(dataset.id, node.id) if node else []
        results = (
            [
                SearchHit(
                    record_id=node.id,
                    code=node.code,
                    title=node.title,
                    level=node.node_type.value,
                    is_selectable=node.is_selectable,
                    hybrid_score=1.0,
                    scores={"exact": 1.0},
                    matched_terms=[MatchedTerm(text=node.code or "", match_type="code", score=1.0)],
                    hierarchy=[hierarchy_item(a) for a in ancestors],
                )
            ]
            if node and (not selectable_only or node.is_selectable)
            else []
        )
        return SearchResponse(
            dataset=dataset_summary(dataset),
            query=q,
            mode=mode,
            semantic_available=False,
            results=results,
        )
    retriever = HybridRetriever(session, settings, provider if mode == "hybrid" else None)
    result = await retriever.retrieve(
        dataset,
        [q],
        top_k=limit,
        filters=RetrievalFilters(
            node_types=[level] if level else None, selectable_only=selectable_only
        ),
    )
    return SearchResponse(
        dataset=dataset_summary(dataset),
        query=q,
        mode=mode,
        semantic_available=result.semantic_available,
        results=[
            SearchHit(
                record_id=c.node.id,
                code=c.node.code,
                title=c.node.title,
                level=c.node.node_type.value,
                is_selectable=c.node.is_selectable,
                hybrid_score=c.hybrid_score,
                scores=c.scores.as_dict(),
                matched_terms=[
                    MatchedTerm(
                        text=m.matched_text, match_type=m.match_type, score=round(m.score, 4)
                    )
                    for m in c.evidence
                ],
                hierarchy=[hierarchy_item(a) for a in c.ancestors],
            )
            for c in result.candidates
        ],
    )


# --- codes ------------------------------------------------------------------------------------


async def _lookup(
    session: AsyncSession, selector: DatasetSelector, code: str
) -> tuple[IcdDataset, IcdNode]:
    dataset = await selector.resolve(session)
    node = await IcdRepository(session).get_by_code(dataset.id, code)
    if node is None:
        raise InvalidICDCodeError(
            f"Code {code!r} does not exist in {dataset.system} {dataset.version}",
            details={"dataset_id": dataset.id, "code": code},
        )
    return dataset, node


@router.get("/codes/{code}", response_model=ICDRecordOut)
async def get_code(code: str, session: SessionDep, selector: SelectorDep) -> ICDRecordOut:
    dataset, node = await _lookup(session, selector, code)
    records = IcdRepository(session)
    ancestors = await records.ancestors(dataset.id, node.id)
    details = await records.details_of_many(dataset.id, [node.id])
    return record_out(node, ancestors, details[node.id])


async def _records_out(
    session: AsyncSession, dataset: IcdDataset, nodes: list[IcdNode]
) -> list[ICDRecordOut]:
    records = IcdRepository(session)
    ids = [n.id for n in nodes]
    ancestors = await records.ancestors_of_many(dataset.id, ids)
    details = await records.details_of_many(dataset.id, ids)
    return [record_out(n, ancestors.get(n.id, []), details.get(n.id)) for n in nodes]


@router.get("/codes/{code}/children", response_model=RecordListResponse)
async def get_children(code: str, session: SessionDep, selector: SelectorDep) -> RecordListResponse:
    dataset, node = await _lookup(session, selector, code)
    children = await IcdRepository(session).children(dataset.id, node.id)
    return RecordListResponse(
        dataset=dataset_summary(dataset),
        record=hierarchy_item(node),
        records=await _records_out(session, dataset, children),
    )


@router.get("/codes/{code}/ancestors", response_model=RecordListResponse)
async def get_ancestors(
    code: str, session: SessionDep, selector: SelectorDep
) -> RecordListResponse:
    dataset, node = await _lookup(session, selector, code)
    ancestors = await IcdRepository(session).ancestors(dataset.id, node.id)
    return RecordListResponse(
        dataset=dataset_summary(dataset),
        record=hierarchy_item(node),
        records=await _records_out(session, dataset, ancestors),
    )


# --- suggestions --------------------------------------------------------------------------------


@router.post("/suggest", response_model=SuggestResponse)
async def suggest(
    request: SuggestRequest,
    session: SessionDep,
    settings: SettingsDep,
    provider: ProviderDep,
) -> SuggestResponse:
    """Evidence-backed suggestions. The clinical note is processed in memory only: it is not
    stored and not logged."""
    return await SuggestionService(session, settings, provider).suggest(request)
