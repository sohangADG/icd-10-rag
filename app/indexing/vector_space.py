"""One embedding space = (provider, model, dimension, normalisation) + the distance used to
search it. The HNSW index and every vector query are generated from the same VectorSpace, so
the index operator class always matches the query operator (cosine index with `<=>`, L2 index
with `<->`, inner-product index with `<#>`), and rows of another space can never be compared.

Identifiers and literals are interpolated (partial-index predicates must be literal for the
planner to use them); every value is validated or escaped here.
"""

import hashlib
import re
from dataclasses import dataclass
from typing import Literal

from app.indexing.embeddings import EmbeddingProvider
from app.models.search_document import HNSW_INDEX_PREFIX

Distance = Literal["cosine", "l2", "inner_product"]

# distance -> (operator class, operator, similarity expression in terms of the distance `d`)
_DISTANCES: dict[str, tuple[str, str, str]] = {
    "cosine": ("vector_cosine_ops", "<=>", "1 - ({d})"),  # cosine similarity in [-1, 1]
    "l2": ("vector_l2_ops", "<->", "1 / (1 + ({d}))"),  # (0, 1]
    "inner_product": ("vector_ip_ops", "<#>", "-({d})"),  # <#> is the negative inner product
}
HNSW_MAX_DIMENSION = 2000  # pgvector limit for HNSW on `vector`


def _literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


@dataclass(frozen=True)
class VectorSpace:
    provider: str
    model: str
    dimension: int
    normalized: bool
    distance: Distance = "cosine"

    def __post_init__(self) -> None:
        if self.distance not in _DISTANCES:
            raise ValueError(f"Unsupported distance {self.distance!r}")
        if not 1 <= int(self.dimension) <= 16000:
            raise ValueError(f"Invalid vector dimension {self.dimension}")

    @classmethod
    def of(cls, provider: EmbeddingProvider, distance: Distance = "cosine") -> "VectorSpace":
        return cls(
            provider.provider_name,
            provider.model,
            provider.dimension,
            provider.normalized,
            distance,
        )

    @property
    def key(self) -> str:
        """Stored-vector identity (the distance does not change the vectors themselves)."""
        state = "norm" if self.normalized else "raw"
        return f"{self.provider}|{self.model}|{self.dimension}|{state}"

    @property
    def operator(self) -> str:
        return _DISTANCES[self.distance][1]

    @property
    def operator_class(self) -> str:
        return _DISTANCES[self.distance][0]

    def cast(self, expression: str) -> str:
        return f"({expression})::vector({int(self.dimension)})"

    def distance_sql(self, column: str = "embedding", query: str = "CAST(:qv AS vector)") -> str:
        return f"{self.cast(column)} {self.operator} {self.cast(query)}"

    def similarity_sql(self, column: str = "embedding", query: str = "CAST(:qv AS vector)") -> str:
        return _DISTANCES[self.distance][2].format(d=self.distance_sql(column, query))

    def predicate_sql(self) -> str:
        """Rows belonging to this space. Shared verbatim by the index and the queries."""
        return (
            f"embedding_provider = {_literal(self.provider)} "
            f"AND embedding_model = {_literal(self.model)} "
            f"AND embedding_dimension = {int(self.dimension)} "
            f"AND embedding_normalized = {'true' if self.normalized else 'false'}"
        )

    @property
    def index_name(self) -> str:
        digest = hashlib.sha1(f"{self.key}|{self.distance}".encode()).hexdigest()[:12]
        name = f"{HNSW_INDEX_PREFIX}{digest}"
        if not re.fullmatch(r"[a-z0-9_]+", name):  # defensive: identifiers are interpolated
            raise ValueError("invalid index name")
        return name

    @property
    def indexable(self) -> bool:
        return self.dimension <= HNSW_MAX_DIMENSION

    def create_index_sql(self) -> str:
        return (
            f"CREATE INDEX IF NOT EXISTS {self.index_name} ON icd_search_documents "
            f"USING hnsw (({self.cast('embedding')}) {self.operator_class}) "
            f"WHERE {self.predicate_sql()}"
        )
