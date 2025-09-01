from typing import List
import numpy as np

# Inputs: ranked list of (pid, score, label)

def dcg_at_k(labels: List[int], k: int) -> float:
	labels = list(np.asarray(labels)[:k])
	gains = (np.power(2, labels) - 1)
	discounts = 1.0 / np.log2(np.arange(2, len(labels) + 2))
	return float(np.sum(gains * discounts))


def ndcg_at_k(labels: List[int], k: int) -> float:
	ideal = sorted(labels, reverse=True)
	denom = dcg_at_k(ideal, k)
	if denom == 0:
		return 0.0
	return dcg_at_k(labels, k) / denom


def mrr_at_k(labels: List[int], k: int) -> float:
	for i, y in enumerate(labels[:k]):
		if y > 0:
			return 1.0 / (i + 1)
	return 0.0


def recall_at_k(labels: List[int], k: int) -> float:
	pos = sum(1 for y in labels if y > 0)
	if pos == 0:
		return 0.0
	got = sum(1 for y in labels[:k] if y > 0)
	return got / pos