from __future__ import annotations

from typing import Iterable


def hit_rate_at_k(
    retrieved_pages: Iterable[int],
    expected_pages: Iterable[int],
    k: int,
) -> float:
    """
    Hit Rate@K

    Returns 1.0 if at least one expected page appears
    within the top-k retrieved pages, otherwise 0.0.
    """

    retrieved = list(retrieved_pages)[:k]
    expected = set(expected_pages)

    if not expected:
        return 0.0

    return float(
        any(page in expected for page in retrieved)
    )


def recall_at_k(
    retrieved_pages: Iterable[int],
    expected_pages: Iterable[int],
    k: int,
) -> float:
    """
    Recall@K

    Measures what fraction of expected pages were found
    within the top-k retrieved results.
    """

    retrieved = set(list(retrieved_pages)[:k])
    expected = set(expected_pages)

    if not expected:
        return 0.0

    relevant_found = retrieved.intersection(expected)

    return len(relevant_found) / len(expected)


def reciprocal_rank(
    retrieved_pages: Iterable[int],
    expected_pages: Iterable[int],
) -> float:
    """
    Reciprocal Rank

    Returns:
        1 / rank_of_first_relevant_result

    Example:
        relevant result at rank 1 -> 1.0
        relevant result at rank 2 -> 0.5
        relevant result at rank 4 -> 0.25
    """

    expected = set(expected_pages)

    if not expected:
        return 0.0

    for rank, page in enumerate(
        retrieved_pages,
        start=1,
    ):
        if page in expected:
            return 1.0 / rank

    return 0.0


def mean_reciprocal_rank(
    reciprocal_ranks: Iterable[float],
) -> float:
    """
    Mean Reciprocal Rank across multiple evaluation cases.
    """

    values = list(reciprocal_ranks)

    if not values:
        return 0.0

    return sum(values) / len(values)


def average_metric(
    values: Iterable[float],
) -> float:
    """
    Generic helper for averaging metric values.
    """

    values = list(values)

    if not values:
        return 0.0

    return sum(values) / len(values)