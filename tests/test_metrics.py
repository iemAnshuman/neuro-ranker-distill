import pytest

from neuro_ranker.metrics import (
    aggregate_ranking_metrics,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)


def test_reciprocal_rank_uses_first_relevant_result():
    assert reciprocal_rank_at_k([0, 0, 1, 1], 10) == pytest.approx(1 / 3)
    assert reciprocal_rank_at_k([0, 0, 1], 2) == 0.0


def test_ndcg_and_recall_have_expected_bounds():
    assert ndcg_at_k([1, 0, 0], 3) == 1.0
    assert 0.0 < ndcg_at_k([0, 1, 0], 3) < 1.0
    assert recall_at_k([1, 0, 1], 2) == 0.5


def test_aggregate_rejects_empty_input():
    with pytest.raises(ValueError):
        aggregate_ranking_metrics([], 10)
