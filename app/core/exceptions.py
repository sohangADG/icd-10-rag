"""Application exceptions. The API layer maps each to an HTTP status (app/api/errors.py)."""

from typing import Any


class AppError(Exception):
    """Base class: `code` is a stable machine-readable identifier returned to clients."""

    code = "APP_ERROR"
    status_code = 500

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class DatasetNotFound(AppError):
    code = "DATASET_NOT_FOUND"
    status_code = 404


class DatasetNotReady(AppError):
    code = "DATASET_NOT_READY"
    status_code = 409


class AmbiguousDatasetError(AppError):
    code = "AMBIGUOUS_DATASET"
    status_code = 409


class UnsupportedCodingSystem(AppError):
    code = "UNSUPPORTED_CODING_SYSTEM"
    status_code = 404


class UnsupportedDatasetVersion(AppError):
    code = "UNSUPPORTED_DATASET_VERSION"
    status_code = 404


class IngestionValidationError(AppError):
    code = "INGESTION_VALIDATION_ERROR"
    status_code = 422


class DuplicateDatasetError(AppError):
    code = "DUPLICATE_DATASET"
    status_code = 409


class ImportInProgressError(AppError):
    code = "IMPORT_IN_PROGRESS"
    status_code = 409


class InvalidClinicalNoteError(AppError):
    code = "INVALID_CLINICAL_NOTE"
    status_code = 422


class InvalidICDCodeError(AppError):
    code = "INVALID_ICD_CODE"
    status_code = 404


class RetrievalError(AppError):
    code = "RETRIEVAL_ERROR"
    status_code = 503


class EmbeddingProviderError(AppError):
    code = "EMBEDDING_PROVIDER_ERROR"
    status_code = 503


class LicenceRestrictionError(AppError):
    """An operation would send or expose dataset content in a way the dataset's recorded
    licence metadata does not allow (e.g. a remote embedding provider)."""

    code = "LICENCE_RESTRICTION"
    status_code = 403
