from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import URL, make_url


class Settings(BaseSettings):
    """Environment-driven configuration. Values come from env vars or a local .env file."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "icd-rag-service"
    app_env: Literal["local", "test", "staging", "production"] = "local"
    app_host: str = "127.0.0.1"
    app_port: int = Field(default=8000, ge=1, le=65535)

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_format: Literal["json", "console"] = "json"

    database_host: str = "localhost"
    database_port: int = Field(default=5432, ge=1, le=65535)
    database_name: str = "icd_rag"
    database_user: str = "icd_rag"
    database_password: SecretStr = SecretStr("")
    # Optional full URL; when set it takes precedence over the individual DATABASE_* parts.
    database_url: SecretStr | None = None

    database_pool_size: int = Field(default=5, ge=1)
    database_max_overflow: int = Field(default=10, ge=0)
    database_connect_timeout_seconds: float = Field(default=5.0, gt=0)

    # --- embeddings (provider-independent; see docs/retrieval.md) ---------------------------
    # none | hashing (local, deterministic) | openai (OpenAI-compatible HTTP API) |
    # sentence-transformers (local model, optional extra)
    embedding_provider: Literal["none", "hashing", "openai", "sentence-transformers"] = "hashing"
    embedding_model: str = "hashing-v1"
    embedding_dimension: int = Field(default=256, ge=8, le=4096)
    embedding_api_base_url: str = "https://api.openai.com/v1"
    embedding_api_key: SecretStr | None = None
    embedding_batch_size: int = Field(default=64, ge=1, le=2048)
    embedding_timeout_seconds: float = Field(default=30.0, gt=0)

    # --- retrieval / suggestion ---------------------------------------------------------------
    retrieval_weight_exact: float = Field(default=0.30, ge=0)
    retrieval_weight_lexical: float = Field(default=0.25, ge=0)
    retrieval_weight_fuzzy: float = Field(default=0.20, ge=0)
    retrieval_weight_semantic: float = Field(default=0.15, ge=0)
    retrieval_weight_hierarchy: float = Field(default=0.05, ge=0)
    retrieval_weight_index_term: float = Field(default=0.05, ge=0)
    retrieval_candidates_per_method: int = Field(default=30, ge=1, le=500)
    retrieval_fuzzy_threshold: float = Field(default=0.3, gt=0, le=1)
    max_top_k: int = Field(default=20, ge=1, le=100)
    clinical_note_max_chars: int = Field(default=20_000, ge=100, le=200_000)
    suggest_uncertain_concepts: bool = True

    # --- security -----------------------------------------------------------------------------
    # Admin endpoints are disabled unless a token is configured.
    admin_api_token: SecretStr | None = None
    # Never log clinical note text unless explicitly enabled (and never in production).
    log_clinical_text: bool = False

    @model_validator(mode="after")
    def _no_clinical_text_logging_in_production(self) -> "Settings":
        if self.app_env == "production" and self.log_clinical_text:
            raise ValueError("LOG_CLINICAL_TEXT must not be enabled in production")
        return self

    @model_validator(mode="after")
    def _weights_not_all_zero(self) -> "Settings":
        if sum(self.retrieval_weights().values()) <= 0:
            raise ValueError("At least one RETRIEVAL_WEIGHT_* must be greater than zero")
        return self

    def retrieval_weights(self) -> dict[str, float]:
        return {
            "exact": self.retrieval_weight_exact,
            "lexical": self.retrieval_weight_lexical,
            "fuzzy": self.retrieval_weight_fuzzy,
            "semantic": self.retrieval_weight_semantic,
            "hierarchy": self.retrieval_weight_hierarchy,
            "index_term": self.retrieval_weight_index_term,
        }

    def sqlalchemy_url(self, database_name: str | None = None) -> URL:
        """Async (asyncpg) SQLAlchemy URL. Never log the result: it carries the password."""
        if self.database_url is not None:
            url = make_url(self.database_url.get_secret_value()).set(
                drivername="postgresql+asyncpg"
            )
        else:
            url = URL.create(
                drivername="postgresql+asyncpg",
                username=self.database_user,
                password=self.database_password.get_secret_value() or None,
                host=self.database_host,
                port=self.database_port,
                database=self.database_name,
            )
        return url.set(database=database_name) if database_name else url


@lru_cache
def get_settings() -> Settings:
    return Settings()
