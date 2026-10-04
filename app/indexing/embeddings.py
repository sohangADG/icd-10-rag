"""Provider-independent embedding interface and implementations.

* HashingEmbeddingProvider — local, deterministic, dependency-free feature hashing of words,
  word bigrams and character trigrams. No network, no model download. It captures lexical and
  sub-word similarity only; it is the default so the system works offline and in tests.
* OpenAIEmbeddingProvider — any OpenAI-compatible /embeddings HTTP endpoint (OpenAI, Azure
  OpenAI-compatible gateways, local servers such as vLLM/Ollama-compatible proxies).
  Sends dataset text to a remote service: only allowed for datasets whose licence metadata
  permits remote processing (enforced by the indexer).
* SentenceTransformerProvider — local transformer model (optional `local-embeddings` extra).

Configured through EMBEDDING_* environment variables; API keys are never hard-coded or logged.
"""

import asyncio
import hashlib
import math
from abc import ABC, abstractmethod
from typing import Any

import httpx

from app.core.config import Settings
from app.core.exceptions import EmbeddingProviderError
from app.core.text import normalize_text


class EmbeddingProvider(ABC):
    #: Stored with every vector; vectors from different models are never compared.
    model: str
    dimension: int
    #: True when text leaves this process (network call to a third party).
    remote: bool = False

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts; returns one L2-normalised vector of `dimension` floats per text."""

    async def embed_query(self, text: str) -> list[float]:
        return (await self.embed([text]))[0]


def _l2_normalise(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    return [v / norm for v in vector] if norm else vector


class HashingEmbeddingProvider(EmbeddingProvider):
    remote = False

    def __init__(self, dimension: int = 256, model: str = "hashing-v1") -> None:
        self.dimension = dimension
        self.model = f"{model}-{dimension}" if not model.endswith(f"-{dimension}") else model

    def _features(self, text: str) -> list[tuple[str, float]]:
        words = normalize_text(text).split()
        features: list[tuple[str, float]] = [(f"w:{w}", 1.0) for w in words]
        features += [(f"b:{a}_{b}", 0.7) for a, b in zip(words, words[1:], strict=False)]
        for word in words:
            padded = f"#{word}#"
            features += [(f"c:{padded[i : i + 3]}", 0.3) for i in range(len(padded) - 2)]
        return features

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for feature, weight in self._features(text):
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            sign = 1.0 if value & 1 else -1.0
            vector[(value >> 1) % self.dimension] += sign * weight
        return _l2_normalise(vector)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]


class OpenAIEmbeddingProvider(EmbeddingProvider):
    remote = True

    def __init__(
        self,
        *,
        model: str,
        dimension: int,
        api_key: str,
        base_url: str,
        timeout: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not api_key:
            raise EmbeddingProviderError("EMBEDDING_API_KEY is not configured")
        self.model = model
        self.dimension = dimension
        self._api_key = api_key
        self._url = base_url.rstrip("/") + "/embeddings"
        self._timeout = timeout
        self._transport = transport

    async def embed(self, texts: list[str]) -> list[list[float]]:
        payload: dict[str, Any] = {"model": self.model, "input": texts}
        if self.dimension:
            payload["dimensions"] = self.dimension
        try:
            async with httpx.AsyncClient(
                timeout=self._timeout, transport=self._transport
            ) as client:
                response = await client.post(
                    self._url,
                    json=payload,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                )
        except httpx.HTTPError as exc:
            # Never include request details: headers carry the API key.
            raise EmbeddingProviderError(
                f"Embedding request failed ({type(exc).__name__})"
            ) from None
        if response.status_code != 200:
            raise EmbeddingProviderError(
                f"Embedding provider returned HTTP {response.status_code}",
                details={"status_code": response.status_code},
            )
        try:
            data = sorted(response.json()["data"], key=lambda item: item["index"])
            vectors = [list(map(float, item["embedding"])) for item in data]
        except (KeyError, TypeError, ValueError) as exc:
            raise EmbeddingProviderError("Malformed embedding response") from exc
        if len(vectors) != len(texts) or any(len(v) != self.dimension for v in vectors):
            raise EmbeddingProviderError(
                "Embedding response does not match the configured dimension/batch size"
            )
        return [_l2_normalise(v) for v in vectors]


class SentenceTransformerProvider(EmbeddingProvider):
    remote = False

    def __init__(self, model: str, dimension: int) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
        except ImportError as exc:
            raise EmbeddingProviderError(
                "sentence-transformers is not installed (pip install '.[local-embeddings]')"
            ) from exc
        self._model = SentenceTransformer(model)
        self.model = model
        self.dimension = dimension
        actual = self._model.get_sentence_embedding_dimension()
        if actual != dimension:
            raise EmbeddingProviderError(
                f"Model {model} produces {actual}-dimensional vectors, not {dimension}"
            )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = await asyncio.to_thread(
            self._model.encode, texts, normalize_embeddings=True, convert_to_numpy=True
        )
        return [list(map(float, v)) for v in vectors]


def get_embedding_provider(settings: Settings) -> EmbeddingProvider | None:
    """The configured provider, or None when semantic retrieval is disabled."""
    match settings.embedding_provider:
        case "none":
            return None
        case "hashing":
            return HashingEmbeddingProvider(settings.embedding_dimension, settings.embedding_model)
        case "openai":
            return OpenAIEmbeddingProvider(
                model=settings.embedding_model,
                dimension=settings.embedding_dimension,
                api_key=settings.embedding_api_key.get_secret_value()
                if settings.embedding_api_key
                else "",
                base_url=settings.embedding_api_base_url,
                timeout=settings.embedding_timeout_seconds,
            )
        case "sentence-transformers":
            return SentenceTransformerProvider(
                settings.embedding_model, settings.embedding_dimension
            )
    raise EmbeddingProviderError(f"Unknown embedding provider {settings.embedding_provider!r}")
