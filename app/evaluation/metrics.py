"""Pure evaluation metrics (no I/O), so they can be unit-tested exactly."""

from collections.abc import Sequence


def rank_of(expected: str, ranked: Sequence[str | None]) -> int | None:
    """1-based rank of the expected code, or None if absent."""
    for index, code in enumerate(ranked, start=1):
        if code == expected:
            return index
    return None


def recall_at_k(ranks: Sequence[int | None], k: int) -> float:
    if not ranks:
        return 0.0
    return sum(1 for r in ranks if r is not None and r <= k) / len(ranks)


def mean_reciprocal_rank(ranks: Sequence[int | None]) -> float:
    if not ranks:
        return 0.0
    return sum(1.0 / r for r in ranks if r is not None) / len(ranks)


def rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def hierarchy_related(predicted: str, expected: str, ancestors_of: dict[str, set[str]]) -> bool:
    """True when the prediction is the expected code, one of its ancestors or descendants
    (i.e. the right branch at a different specificity)."""
    if predicted == expected:
        return True
    return expected in ancestors_of.get(predicted, set()) or predicted in ancestors_of.get(
        expected, set()
    )


def agreement(pairs: Sequence[tuple[str | None, str | None]]) -> float:
    """Share of cases where the system's top code equals the human coder's code."""
    usable = [(system, human) for system, human in pairs if human is not None]
    return rate(sum(1 for system, human in usable if system == human), len(usable))
