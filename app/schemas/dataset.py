from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.core.constants import DatasetStatus


class DatasetCreate(BaseModel):
    """Identity + metadata needed to register an ICD dataset. No classification content."""

    model_config = ConfigDict(str_strip_whitespace=True)

    system: str = Field(min_length=1, max_length=64)
    country: str = Field(min_length=2, max_length=8)
    version: str = Field(min_length=1, max_length=32)
    revision: str | None = Field(default=None, max_length=32)
    edition: str | None = Field(default=None, max_length=64)
    publisher: str = Field(min_length=1, max_length=255)
    publication_year: int | None = Field(default=None, gt=1900)
    language: str = Field(min_length=2, max_length=16)
    source_filename: str | None = None
    source_checksum: str | None = Field(default=None, max_length=128)
    effective_from: date | None = None
    effective_to: date | None = None

    @model_validator(mode="after")
    def _check_effective_range(self) -> "DatasetCreate":
        if self.effective_from and self.effective_to and self.effective_to < self.effective_from:
            raise ValueError("effective_to must not be before effective_from")
        return self


class DatasetRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    system: str
    country: str
    version: str
    revision: str | None
    edition: str | None
    publisher: str
    publication_year: int | None
    language: str
    source_filename: str | None
    source_checksum: str | None
    status: DatasetStatus
    effective_from: date | None
    effective_to: date | None
    created_at: datetime
    updated_at: datetime


class DatasetRegistration(BaseModel):
    dataset: DatasetRead
    created: bool
