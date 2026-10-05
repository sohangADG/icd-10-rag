"""Request/response models for the ICD search, lookup and suggestion APIs."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.clinical.models import ClinicalConcept

CONFIDENCE_NOTE = (
    "Confidence expresses how strongly the documentation and the classification data support "
    "the suggested code (retrieval + rule evidence). It is not diagnostic certainty. Every "
    "suggestion must be reviewed by a qualified coder."
)


class DatasetSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    coding_system: str
    modification: str | None = None
    country: str
    version: str
    revision: str | None = None
    edition: str | None = None
    language: str
    title: str | None = None
    publisher: str
    status: str
    source_type: str | None = None
    source_filename: str | None = None
    source_sha256: str | None = None
    source_page_count: int | None = None
    imported_at: datetime | None = None


class DatasetDetail(DatasetSummary):
    record_counts: dict[str, int] = Field(default_factory=dict)
    licence: dict[str, Any] = Field(default_factory=dict)
    latest_ingestion: dict[str, Any] | None = None


class DatasetList(BaseModel):
    datasets: list[DatasetSummary]


class HierarchyItem(BaseModel):
    record_id: int
    code: str | None
    title: str
    level: str


class SourceProvenanceOut(BaseModel):
    source_filename: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    printed_page: str | None = None
    locator: dict[str, Any] | None = None


class RuleOut(BaseModel):
    type: str
    text: str
    target_code: str | None = None
    target_record_id: int | None = None
    exclusion_type: str | None = None


class ICDRecordOut(BaseModel):
    record_id: int
    dataset_id: int
    code: str | None
    normalized_code: str | None
    title: str
    description: str | None = None
    level: str
    is_selectable: bool
    status: str
    parent_record_id: int | None = None
    hierarchy: list[HierarchyItem] = Field(default_factory=list)
    inclusions: list[str] = Field(default_factory=list)
    exclusions: list[RuleOut] = Field(default_factory=list)
    instructions: list[RuleOut] = Field(default_factory=list)
    synonyms: list[str] = Field(default_factory=list)
    index_terms: list[str] = Field(default_factory=list)
    provenance: SourceProvenanceOut | None = None


class RecordListResponse(BaseModel):
    dataset: DatasetSummary
    record: HierarchyItem
    records: list[ICDRecordOut]


class MatchedTerm(BaseModel):
    text: str
    match_type: str
    score: float


class SearchHit(BaseModel):
    record_id: int
    code: str | None
    title: str
    level: str
    is_selectable: bool
    hybrid_score: float
    scores: dict[str, float | None]
    matched_terms: list[MatchedTerm] = Field(default_factory=list)
    hierarchy: list[HierarchyItem] = Field(default_factory=list)


class SearchResponse(BaseModel):
    dataset: DatasetSummary
    query: str
    mode: Literal["exact", "text", "hybrid"]
    semantic_available: bool
    # ok | disabled | provider_unavailable | not_indexed | incomplete_index | error |
    # not_used (exact/text modes)
    semantic_status: str
    # Provider/model/dimension/distance of the semantic component (never vectors).
    embedding_space: dict[str, Any] | None = None
    results: list[SearchHit]


class SuggestRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    clinical_note: str = Field(min_length=1, max_length=200_000)
    coding_system: str = Field(min_length=1, max_length=64)
    version: str = Field(min_length=1, max_length=32)
    country: str | None = Field(default=None, max_length=8)
    language: str | None = Field(default=None, max_length=16)
    dataset_id: int | None = Field(default=None, ge=1)
    top_k: int = Field(default=5, ge=1, le=100)
    include_uncertain: bool | None = None

    @field_validator("clinical_note")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("clinical_note must not be blank")
        return value


class Alternative(BaseModel):
    record_id: int
    code: str | None
    title: str
    reason: str
    rerank_score: float | None = None
    is_selectable: bool | None = None


class Suggestion(BaseModel):
    clinical_concept: str
    concept_status: str
    concept_type: str
    code: str
    title: str
    record_id: int
    evidence: list[dict[str, Any]]
    icd_reference: dict[str, Any]
    alternatives: list[Alternative]
    missing_information: list[str]
    confidence: Literal["HIGH", "MEDIUM", "LOW"]
    retrieval_scores: dict[str, Any]
    validation: dict[str, Any]
    source_provenance: SourceProvenanceOut | None


class UnmatchedConcept(BaseModel):
    clinical_concept: str
    concept_status: str
    reason: str
    rejected_candidates: list[dict[str, Any]] = Field(default_factory=list)


class SuggestResponse(BaseModel):
    dataset: DatasetSummary
    clinical_concepts: list[ClinicalConcept]
    suggestions: list[Suggestion]
    unmatched_concepts: list[UnmatchedConcept]
    confidence_note: str = CONFIDENCE_NOTE
    duration_ms: int
