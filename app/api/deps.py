"""Shared FastAPI dependencies."""

import hmac
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, Header

from app.core.config import Settings, get_settings
from app.core.exceptions import AppError
from app.indexing.embeddings import EmbeddingProvider, get_embedding_provider


@lru_cache
def _provider(settings_id: int) -> EmbeddingProvider | None:  # noqa: ARG001 - cache key
    return get_embedding_provider(get_settings())


def embedding_provider(
    settings: Annotated[Settings, Depends(get_settings)],
) -> EmbeddingProvider | None:
    """Process-wide provider instance (models may be expensive to load)."""
    return _provider(id(settings))


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
