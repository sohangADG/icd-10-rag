from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
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
