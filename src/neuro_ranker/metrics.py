from __future__ import annotations

import math
from collections.abc import Sequence


def reciprocal_rank_at_k(relevances: Sequence[int], k: int) -> float:
    for rank, relevance in enumerate(relevances[:k], start=1):
        if relevance > 0:
            return 1.0 / rank
    return 0.0


def dcg_at_k(relevances: Sequence[int], k: int) -> float:
    return sum(
        (2**relevance - 1) / math.log2(rank + 1)
        for rank, relevance in enumerate(relevances[:k], start=1)
    )


def ndcg_at_k(relevances: Sequence[int], k: int) -> float:
    ideal = sorted(relevances, reverse=True)
    denominator = dcg_at_k(ideal, k)
    return 0.0 if denominator == 0 else dcg_at_k(relevances, k) / denominator


def recall_at_k(relevances: Sequence[int], k: int) -> float:
    total_relevant = sum(relevance > 0 for relevance in relevances)
    if total_relevant == 0:
        return 0.0
    return sum(relevance > 0 for relevance in relevances[:k]) / total_relevant


def aggregate_ranking_metrics(rankings: Sequence[Sequence[int]], k: int = 10) -> dict[str, float]:
    if not rankings:
        raise ValueError("at least one ranking is required")
    count = len(rankings)
    return {
        f"mrr@{k}": sum(reciprocal_rank_at_k(row, k) for row in rankings) / count,
        f"ndcg@{k}": sum(ndcg_at_k(row, k) for row in rankings) / count,
        f"recall@{k}": sum(recall_at_k(row, k) for row in rankings) / count,
    }
