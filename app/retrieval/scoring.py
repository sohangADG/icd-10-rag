"""Explicit score normalisation for hybrid retrieval.

Every component is mapped onto an *absolute* 0..1 scale that does not depend on which other
documents happened to be retrieved, so scores are comparable across queries, across candidate
generation and completion queries, and across components:

* exact: code equality, 1 or 0.
* lexical: 0.75 * coverage of the query lexemes by the record's A/B lexemes
  + 0.25 * ts_rank_cd density (normalisation 32, i.e. rank / (rank + 1)).
* fuzzy and index_term: (similarity - threshold) / (1 - threshold) over pg_trgm similarity,
  i.e. 0 at or below RETRIEVAL_FUZZY_THRESHOLD (trigram noise), 1.0 for exact term equality.
* hierarchy: share of query words found in the ancestor titles.
* semantic: (similarity - floor) / (1 - floor), clamped to 0..1, where similarity is the
  vector similarity of the space's distance (cosine for normalised vectors).

The semantic floor removes a model's similarity baseline (unrelated texts score ~0.5-0.6
cosine with BGE-style models); without it, semantic similarity would add a large constant to
every candidate and dominate by scale alone.
"""

from collections.abc import Mapping

COMPONENTS = ("exact", "lexical", "fuzzy", "semantic", "hierarchy", "index_term")
LEXICAL_COVERAGE_WEIGHT = 0.75


def clamp01(value: float) -> float:
    return 0.0 if value <= 0 else 1.0 if value >= 1 else value


def lexical_score(rank_density: float, coverage: float) -> float:
    """Absolute lexical score from ts_rank_cd normalisation 32 (rank/(rank+1)) and coverage."""
    rank_density, coverage = clamp01(rank_density), clamp01(coverage)
    return round(
        LEXICAL_COVERAGE_WEIGHT * coverage + (1 - LEXICAL_COVERAGE_WEIGHT) * rank_density, 4
    )


def calibrate_similarity(similarity: float, floor: float) -> float:
    """Map a raw vector similarity onto 0..1 above the model's baseline `floor`."""
    if floor >= 1:
        raise ValueError("similarity floor must be < 1")
    return round(clamp01((similarity - floor) / (1 - floor)), 4)


def calibrate_trigram(similarity: float, threshold: float) -> float:
    """Map pg_trgm similarity onto 0..1 above its noise threshold.

    Short queries share a few trigrams with almost any title (0.1-0.3 similarity); those must
    score 0, exactly like semantic similarity at or below the model's floor. 1.0 stays 1.0.
    """
    return calibrate_similarity(similarity, threshold)


def hybrid_score(scores: Mapping[str, float | None], weights: Mapping[str, float]) -> float:
    """Weighted mean over *available* components (None = unavailable, its weight dropped)."""
    total, weight_sum = 0.0, 0.0
    for name in COMPONENTS:
        value = scores.get(name, 0.0)  # absent signal = 0; None = unavailable (dropped)
        if value is None:
            continue
        weight = weights.get(name, 0.0)
        total += weight * clamp01(value)
        weight_sum += weight
    return round(total / weight_sum, 4) if weight_sum else 0.0
