"""Candidate-generation queries. Each returns per-node scores in [0, 1] for one dataset.

Every query is bounded by LIMIT and scoped by dataset_id; the optional `node_ids` argument
scores a known candidate pool instead of searching (used to complete component scores).
"""

from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.constants import SEARCH_DOCUMENT_TYPE_RECORD
from app.core.exceptions import RetrievalError
from app.indexing.vector_space import VectorSpace
from app.ingestion.codes import clean_code, normalize_code
from app.retrieval.scoring import lexical_score


@dataclass(frozen=True)
class TermMatch:
    node_id: int
    score: float
    matched_text: str
    match_type: str  # title | INCLUSION | SYNONYM | ABBREVIATION | INDEX_TERM | code


def _vector_literal(vector: Sequence[float]) -> str:
    return "[" + ",".join(f"{v:.7g}" for v in vector) + "]"


class SearchRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def exact_code(self, dataset_id: int, query: str) -> list[TermMatch]:
        cleaned = clean_code(query)
        rows = await self._session.execute(
            text(
                """
                SELECT id, code FROM icd_nodes
                WHERE dataset_id = :dataset_id
                  AND (code = :code OR normalized_code = :normalized OR normalized_code = :code)
                ORDER BY depth DESC LIMIT 5
                """
            ),
            {"dataset_id": dataset_id, "code": cleaned, "normalized": normalize_code(cleaned)},
        )
        return [TermMatch(row.id, 1.0, row.code, "code") for row in rows]

    async def lexical(
        self,
        dataset_id: int,
        ts_config: str,
        query: str,
        *,
        limit: int,
        node_ids: Sequence[int] | None = None,
    ) -> dict[int, float]:
        """Absolute full-text score (app.retrieval.scoring.lexical_score): coverage of the
        query's lexemes by the record's high-weight (A/B) lexemes plus ts_rank_cd density.
        Independent of which other documents matched, so scores are comparable everywhere."""
        rows = await self._session.execute(
            text(
                f"""
                WITH q AS (
                    SELECT array_agg(DISTINCT lexeme) AS lexemes
                    FROM unnest(to_tsvector(CAST(:cfg AS regconfig), :query))
                ), tq AS (
                    SELECT lexemes,
                           to_tsquery('simple', array_to_string(
                               ARRAY(SELECT quote_literal(l) FROM unnest(lexemes) AS l), ' | '
                           )) AS tsq
                    FROM q WHERE lexemes IS NOT NULL
                )
                SELECT d.node_id,
                       ts_rank_cd(d.search_vector, tq.tsq, 32) AS rank,
                       (SELECT count(*) FROM unnest(
                            tsvector_to_array(ts_filter(d.search_vector, '{{a,b}}'))) AS l
                        WHERE l = ANY(tq.lexemes))::float
                       / greatest(cardinality(tq.lexemes), 1) AS coverage
                FROM icd_search_documents d, tq
                WHERE d.dataset_id = :dataset_id
                  AND d.document_type = '{SEARCH_DOCUMENT_TYPE_RECORD}'
                  AND d.search_vector @@ tq.tsq
                  {"AND d.node_id = ANY(:node_ids)" if node_ids is not None else ""}
                ORDER BY coverage DESC, rank DESC
                LIMIT :limit
                """
            ),
            {
                "cfg": ts_config,
                "query": query,
                "dataset_id": dataset_id,
                "limit": limit,
                **({"node_ids": list(node_ids)} if node_ids is not None else {}),
            },
        )
        return {row.node_id: lexical_score(row.rank, row.coverage) for row in rows}

    async def _set_trigram_thresholds(self, threshold: float) -> None:
        await self._session.execute(
            text(
                "SELECT set_config('pg_trgm.similarity_threshold', :t, true), "
                "set_config('pg_trgm.word_similarity_threshold', :t, true)"
            ),
            {"t": str(threshold)},
        )

    async def fuzzy_titles(
        self,
        dataset_id: int,
        query: str,
        *,
        limit: int,
        threshold: float,
        node_ids: Sequence[int] | None = None,
    ) -> list[TermMatch]:
        """Trigram similarity against official titles (tolerates misspellings/word order)."""
        await self._set_trigram_thresholds(threshold)
        candidate_filter = (
            "n.id = ANY(:node_ids)" if node_ids is not None else "(n.title % :q OR n.title %> :q)"
        )
        rows = await self._session.execute(
            text(
                f"""
                SELECT n.id, n.title,
                       greatest(similarity(n.title, :q),
                                word_similarity(:q, n.title),
                                word_similarity(n.title, :q)) AS score
                FROM icd_nodes n
                WHERE n.dataset_id = :dataset_id AND {candidate_filter}
                ORDER BY score DESC LIMIT :limit
                """
            ),
            {
                "dataset_id": dataset_id,
                "q": query.lower(),
                "limit": limit,
                **({"node_ids": list(node_ids)} if node_ids is not None else {}),
            },
        )
        return [TermMatch(row.id, float(row.score), row.title, "title") for row in rows]

    async def source_terms(
        self,
        dataset_id: int,
        normalized_query: str,
        *,
        limit: int,
        threshold: float,
        node_ids: Sequence[int] | None = None,
    ) -> list[TermMatch]:
        """Matches against source-provided terms: inclusions, synonyms, abbreviations and
        index terms (exact normalized equality scores 1.0; otherwise trigram similarity)."""
        await self._set_trigram_thresholds(threshold)
        term_filter = (
            "t.node_id = ANY(:node_ids)"
            if node_ids is not None
            else "(t.normalized_term % :q OR t.normalized_term %> :q OR t.normalized_term = :q)"
        )
        entry_filter = (
            "e.target_node_id = ANY(:node_ids)"
            if node_ids is not None
            else "(e.normalized_term % :q OR e.normalized_term %> :q OR e.normalized_term = :q)"
        )
        rows = await self._session.execute(
            text(
                f"""
                SELECT node_id, matched, match_type, score FROM (
                    SELECT t.node_id, t.term AS matched, t.term_type AS match_type,
                           CASE WHEN t.normalized_term = :q THEN 1.0 ELSE
                           greatest(similarity(t.normalized_term, :q),
                                    word_similarity(:q, t.normalized_term),
                                    word_similarity(t.normalized_term, :q)) END AS score
                    FROM icd_terms t
                    WHERE t.dataset_id = :dataset_id AND {term_filter}
                    UNION ALL
                    SELECT e.target_node_id, e.lead_term, 'INDEX_TERM',
                           CASE WHEN e.normalized_term = :q THEN 1.0 ELSE
                           greatest(similarity(e.normalized_term, :q),
                                    word_similarity(:q, e.normalized_term),
                                    word_similarity(e.normalized_term, :q)) END
                    FROM icd_index_entries e
                    WHERE e.dataset_id = :dataset_id AND e.target_node_id IS NOT NULL
                      AND {entry_filter}
                ) m
                ORDER BY score DESC LIMIT :limit
                """
            ),
            {
                "dataset_id": dataset_id,
                "q": normalized_query,
                "limit": limit,
                **({"node_ids": list(node_ids)} if node_ids is not None else {}),
            },
        )
        return [
            TermMatch(row.node_id, float(row.score), row.matched, str(row.match_type))
            for row in rows
        ]

    async def space_coverage(self, dataset_id: int, space: VectorSpace) -> tuple[int, int]:
        """(record documents embedded in `space`, total record documents) for a dataset."""
        row = (
            await self._session.execute(
                text(
                    f"""
                    SELECT count(*) FILTER (WHERE embedding IS NOT NULL AND {space.predicate_sql()})
                               AS embedded,
                           count(*) AS total
                    FROM icd_search_documents
                    WHERE dataset_id = :dataset_id
                      AND document_type = '{SEARCH_DOCUMENT_TYPE_RECORD}'
                      -- Documents without semantic text are never embedded: not part of coverage.
                      AND coalesce(btrim(semantic_text), '') <> ''
                    """
                ),
                {"dataset_id": dataset_id},
            )
        ).one()
        return int(row.embedded), int(row.total)

    async def semantic(
        self,
        dataset_id: int,
        space: VectorSpace,
        query_vector: Sequence[float],
        *,
        limit: int,
        node_ids: Sequence[int] | None = None,
    ) -> dict[int, float]:
        """Raw vector similarity within one embedding space (HNSW index of that space).

        Filters by dataset and by the full space predicate (provider, model, dimension,
        normalisation), so vectors of different models are never compared. The ORDER BY uses the
        same operator as the space's index.
        """
        if len(query_vector) != space.dimension:
            raise RetrievalError(
                f"Query vector has {len(query_vector)} dimensions, expected {space.dimension}"
            )
        # pgvector >= 0.8: keep scanning the HNSW graph until enough rows pass the filters, and
        # widen the candidate list (default 40) for better recall on these bounded queries.
        await self._session.execute(
            text(
                "SELECT set_config('hnsw.iterative_scan', 'relaxed_order', true), "
                "set_config('hnsw.ef_search', :ef, true)"
            ),
            {"ef": str(max(100, limit))},
        )
        rows = await self._session.execute(
            text(
                f"""
                SELECT node_id, {space.similarity_sql()} AS similarity
                FROM icd_search_documents
                WHERE dataset_id = :dataset_id
                  AND document_type = '{SEARCH_DOCUMENT_TYPE_RECORD}'
                  AND embedding IS NOT NULL AND {space.predicate_sql()}
                  {"AND node_id = ANY(:node_ids)" if node_ids is not None else ""}
                ORDER BY {space.distance_sql()}
                LIMIT :limit
                """
            ),
            {
                "dataset_id": dataset_id,
                "qv": _vector_literal(query_vector),
                "limit": limit,
                **({"node_ids": list(node_ids)} if node_ids is not None else {}),
            },
        )
        return {row.node_id: float(row.similarity) for row in rows}
