"""Shared FastAPI dependencies."""

import hmac
import logging
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Header

from app.core.config import Settings, get_settings
from app.core.exceptions import AppError, EmbeddingProviderError
from app.indexing.embeddings import EmbeddingProvider, get_embedding_provider
from app.retrieval.hybrid import SemanticStatus

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProviderHandle:
    """The process-wide embedding provider, or why there is none."""

    provider: EmbeddingProvider | None
    status: SemanticStatus


_handles: dict[int, tuple[Settings, ProviderHandle]] = {}


def load_provider(settings: Settings) -> ProviderHandle:
    """Load (once per settings object) the configured provider. A provider that cannot be
    loaded (missing model, no network on first download...) is reported, not raised: requests
    then degrade or fail according to SEMANTIC_RETRIEVAL_MODE."""
    cached = _handles.get(id(settings))
    if cached is not None and cached[0] is settings:
        return cached[1]
    if settings.embedding_provider == "none":
        handle = ProviderHandle(None, SemanticStatus.DISABLED)
    else:
        try:
            handle = ProviderHandle(get_embedding_provider(settings), SemanticStatus.OK)
        except EmbeddingProviderError as exc:
            logger.error(
                "embedding provider could not be loaded",
                extra={"embedding_provider": settings.embedding_provider, "error": exc.message},
            )
            handle = ProviderHandle(None, SemanticStatus.PROVIDER_UNAVAILABLE)
    _handles[id(settings)] = (settings, handle)
    return handle


def embedding_provider(settings: Annotated[Settings, Depends(get_settings)]) -> ProviderHandle:
    """Process-wide provider (models are expensive to load)."""
    return load_provider(settings)


class AdminDisabled(AppError):
    code = "ADMIN_DISABLED"
    status_code = 404


class AdminUnauthorized(AppError):
    code = "UNAUTHORIZED"
    status_code = 401


def require_admin(
    settings: Annotated[Settings, Depends(get_settings)],
    x_admin_token: Annotated[str | None, Header()] = None,
) -> None:
    """Admin endpoints exist only when ADMIN_API_TOKEN is configured; the token is compared in
    constant time."""
    if settings.admin_api_token is None or not settings.admin_api_token.get_secret_value():
        raise AdminDisabled("Not found")
    expected = settings.admin_api_token.get_secret_value().encode()
    if not x_admin_token or not hmac.compare_digest(x_admin_token.encode(), expected):
        raise AdminUnauthorized("Missing or invalid admin token")


async def rate_limit_hook() -> None:
    """Extension point for rate limiting (e.g. a Redis token bucket or an API gateway).

    Intentionally a no-op: deployments enforce limits at the gateway. Keeping it as a dependency
    on every public ICD route means a limiter can be added without touching the routes.
    """
