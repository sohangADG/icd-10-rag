"""Provider-independent embedding interface and implementations.

* SentenceTransformerProvider: local semantic model (default; any sentence-transformers
  compatible model, CPU or GPU). The vector dimension is read from the model, never trusted
  from configuration.
* OpenAIEmbeddingProvider: any OpenAI-compatible /embeddings HTTP endpoint. Sends dataset text
  to a remote service, so it is only allowed for datasets whose licence metadata permits remote
  processing (enforced by the indexer).
* HashingEmbeddingProvider: deterministic feature hashing of words, word bigrams and character
  trigrams. No model, no network. It captures lexical/sub-word similarity only and exists for
  deterministic tests and offline development.

Contract for every provider:
* `provider_name`, `model`, `dimension`, `normalized` identify an embedding space. Vectors are
  stored with all four and only ever compared within the same space.
* Every returned vector has exactly `dimension` floats. Anything else raises
  EmbeddingProviderError; vectors are never truncated or padded.
* `normalized=True` means *every* document and query vector is L2-normalised.
* Documents go through `embed()`. Queries go through `embed_queries()`, which applies the
  optional query prefix (asymmetric models) and a bounded in-process cache keyed by a hash of
  the query text (the text itself is never kept).

Configured through EMBEDDING_* environment variables; API keys are never hard-coded or logged.
"""

import asyncio
import hashlib
import logging
import math
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Callable, Sequence
from typing import Any, ClassVar

import httpx

from app.core.config import Settings
from app.core.exceptions import EmbeddingProviderError
from app.core.text import normalize_text

logger = logging.getLogger(__name__)


def l2_normalise(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    return [v / norm for v in vector] if norm else vector


class QueryEmbeddingCache:
    """Bounded LRU of query vectors keyed by SHA-256 of (space, normalised query)."""

    def __init__(self, max_size: int) -> None:
        self.max_size = max_size
        self._items: OrderedDict[str, list[float]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(space: str, text: str) -> str:
        return hashlib.sha256(f"{space}\x00{normalize_text(text)}".encode()).hexdigest()

    def get(self, key: str) -> list[float] | None:
        vector = self._items.get(key)
        if vector is None:
            self.misses += 1
            return None
        self._items.move_to_end(key)
        self.hits += 1
        return vector

    def put(self, key: str, vector: list[float]) -> None:
        if self.max_size <= 0:
            return
        self._items[key] = vector
        self._items.move_to_end(key)
        while len(self._items) > self.max_size:
            self._items.popitem(last=False)

    def __len__(self) -> int:
        return len(self._items)


class EmbeddingProvider(ABC):
    provider_name: ClassVar[str]
    #: Similarity at/below which the semantic score is 0 unless configured otherwise.
    default_similarity_floor: ClassVar[float] = 0.0
    #: True when text leaves this process (network call to a third party).
    remote: ClassVar[bool] = False
    #: True when similarity reflects meaning (a language model). A purely lexical provider is
    #: never accepted as the only evidence that a concept and a code name the same condition.
    meaning_based: ClassVar[bool] = True

    model: str
    dimension: int
    normalized: bool = True
    query_prefix: str = ""
    _query_cache: QueryEmbeddingCache | None = None

    @property
    def space(self) -> str:
        """Identity of the embedding space: vectors are only comparable within one space."""
        state = "norm" if self.normalized else "raw"
        return f"{self.provider_name}|{self.model}|{self.dimension}|{state}"

    def enable_query_cache(self, max_size: int) -> None:
        self._query_cache = QueryEmbeddingCache(max_size) if max_size > 0 else None

    @property
    def query_cache(self) -> QueryEmbeddingCache | None:
        return self._query_cache

    @abstractmethod
    async def _embed(self, texts: list[str]) -> list[list[float]]:
        """Raw vectors for `texts`, in order."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Document embeddings: validated length, normalised when `normalized` is set."""
        if not texts:
            return []
        vectors = await self._embed(texts)
        return self._checked(vectors, len(texts))

    async def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        """Query embeddings: query prefix applied; cached by query hash."""
        results: list[list[float] | None] = [None] * len(texts)
        missing: list[int] = []
        cache = self._query_cache
        for index, text in enumerate(texts):
            cached = (
                cache.get(QueryEmbeddingCache.key(self.space, text)) if cache is not None else None
            )
            if cached is None:
                missing.append(index)
            else:
                results[index] = cached
        if missing:
            vectors = await self.embed([f"{self.query_prefix}{texts[i]}" for i in missing])
            for index, vector in zip(missing, vectors, strict=True):
                results[index] = vector
                if cache is not None:
                    cache.put(QueryEmbeddingCache.key(self.space, texts[index]), vector)
        return [vector for vector in results if vector is not None]

    async def embed_query(self, text: str) -> list[float]:
        return (await self.embed_queries([text]))[0]

    def _checked(self, vectors: list[list[float]], expected_count: int) -> list[list[float]]:
        if len(vectors) != expected_count:
            raise EmbeddingProviderError(
                f"{self.provider_name} returned {len(vectors)} vectors for {expected_count} texts"
            )
        checked: list[list[float]] = []
        for vector in vectors:
            if len(vector) != self.dimension:
                raise EmbeddingProviderError(
                    f"{self.provider_name}/{self.model} produced a {len(vector)}-dimensional "
                    f"vector; expected {self.dimension}. Vectors are never truncated or padded."
                )
            if not all(math.isfinite(v) for v in vector):
                raise EmbeddingProviderError(f"{self.provider_name} produced a non-finite vector")
            checked.append(l2_normalise(vector) if self.normalized else vector)
        return checked


class HashingEmbeddingProvider(EmbeddingProvider):
    """Deterministic lexical hashing. For tests and offline development only."""

    provider_name: ClassVar[str] = "hashing"
    meaning_based: ClassVar[bool] = False

    def __init__(
        self, dimension: int = 256, model: str = "hashing-v1", *, normalize: bool = True
    ) -> None:
        self.dimension = dimension
        self.model = f"{model}-{dimension}" if not model.endswith(f"-{dimension}") else model
        self.normalized = normalize

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
        return vector

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]


class OpenAIEmbeddingProvider(EmbeddingProvider):
    provider_name: ClassVar[str] = "openai"
    remote: ClassVar[bool] = True

    def __init__(
        self,
        *,
        model: str,
        dimension: int,
        api_key: str,
        base_url: str,
        timeout: float = 30.0,
        normalize: bool = True,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not api_key:
            raise EmbeddingProviderError("EMBEDDING_API_KEY is not configured")
        self.model = model
        self.dimension = dimension
        self.normalized = normalize
        self._api_key = api_key
        self._url = base_url.rstrip("/") + "/embeddings"
        self._timeout = timeout
        self._transport = transport

    async def _embed(self, texts: list[str]) -> list[list[float]]:
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
            return [list(map(float, item["embedding"])) for item in data]
        except (KeyError, TypeError, ValueError) as exc:
            raise EmbeddingProviderError("Malformed embedding response") from exc


ModelLoader = Callable[[str, str], Any]


def _load_sentence_transformer(model: str, device: str) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - dependency is installed by default
        raise EmbeddingProviderError(
            "sentence-transformers is not installed (pip install sentence-transformers)"
        ) from exc
    try:
        # Downloaded once into the Hugging Face cache (HF_HOME), then loaded locally.
        return SentenceTransformer(model, device=device)
    except Exception as exc:  # noqa: BLE001 - library raises many error types
        raise EmbeddingProviderError(
            f"Could not load embedding model {model!r} ({type(exc).__name__})"
        ) from exc


class SentenceTransformerProvider(EmbeddingProvider):
    """Local semantic embeddings with any sentence-transformers compatible model."""

    provider_name: ClassVar[str] = "sentence_transformers"
    # Calibrated for the default model (BAAI/bge-small-en-v1.5) on the synthetic suites
    # (2026-10-04): unrelated record documents score median 0.565 / p75 0.611 cosine against a
    # query, correct targets median 0.725 / min 0.643. 0.6 zeroes most unrelated documents and
    # keeps every target positive. Model-specific: set EMBEDDING_SIMILARITY_FLOOR when
    # switching models (docs/retrieval.md).
    default_similarity_floor: ClassVar[float] = 0.6

    def __init__(
        self,
        model: str,
        *,
        device: str = "cpu",
        batch_size: int = 32,
        normalize: bool = True,
        expected_dimension: int | None = None,
        query_prefix: str = "",
        loader: ModelLoader | None = None,
    ) -> None:
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.normalized = normalize
        self.query_prefix = query_prefix
        self._model = (loader or _load_sentence_transformer)(model, device)
        getter = getattr(self._model, "get_embedding_dimension", None) or getattr(
            self._model, "get_sentence_embedding_dimension", None
        )
        actual = getter() if getter else None
        if not actual:
            # Some models do not declare it: measure instead of trusting configuration.
            actual = len(self._model.encode(["dimension probe"], convert_to_numpy=True)[0])
        if expected_dimension is not None and expected_dimension != actual:
            raise EmbeddingProviderError(
                f"Model {model} produces {actual}-dimensional vectors, but EMBEDDING_DIMENSION "
                f"is {expected_dimension}"
            )
        self.dimension = int(actual)

    def _encode(self, texts: list[str]) -> list[list[float]]:
        # Normalisation is applied once, uniformly, in EmbeddingProvider._checked.
        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        return [[float(v) for v in vector] for vector in vectors]

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self._encode, texts)


def get_embedding_provider(
    settings: Settings, *, loader: ModelLoader | None = None
) -> EmbeddingProvider | None:
    """The configured provider (or None when semantic retrieval is disabled)."""
    provider: EmbeddingProvider | None
    match settings.embedding_provider:
        case "none":
            return None
        case "hashing":
            model = settings.embedding_model
            provider = HashingEmbeddingProvider(
                settings.embedding_dimension or 256,
                model if model.startswith("hashing") else "hashing-v1",
                normalize=settings.embedding_normalize,
            )
        case "openai":
            if settings.embedding_dimension is None:
                raise EmbeddingProviderError("EMBEDDING_DIMENSION is required for openai")
            provider = OpenAIEmbeddingProvider(
                model=settings.embedding_model,
                dimension=settings.embedding_dimension,
                api_key=settings.embedding_api_key.get_secret_value()
                if settings.embedding_api_key
                else "",
                base_url=settings.embedding_api_base_url,
                timeout=settings.embedding_timeout_seconds,
                normalize=settings.embedding_normalize,
            )
        case "sentence_transformers":
            provider = SentenceTransformerProvider(
                settings.embedding_model,
                device=settings.embedding_device,
                batch_size=settings.embedding_batch_size,
                normalize=settings.embedding_normalize,
                expected_dimension=settings.embedding_dimension,
                query_prefix=settings.embedding_query_prefix,
                loader=loader,
            )
        case _:
            raise EmbeddingProviderError(
                f"Unknown embedding provider {settings.embedding_provider!r}"
            )
    if settings.embedding_query_prefix and provider.provider_name != "sentence_transformers":
        provider.query_prefix = settings.embedding_query_prefix
    provider.enable_query_cache(settings.embedding_query_cache_size)
    logger.info(
        "embedding provider ready",
        extra={
            "embedding_provider": provider.provider_name,
            "embedding_model": provider.model,
            "dimension": provider.dimension,
            "normalized": provider.normalized,
        },
    )
    return provider


def similarity_floor(settings: Settings, provider: EmbeddingProvider) -> float:
    if settings.embedding_similarity_floor is not None:
        return settings.embedding_similarity_floor
    return provider.default_similarity_floor
