import random
from typing import List

# Simple position‑bias click simulator

def simulate_clicks(labels: List[int], position_propensity: List[float], eta: float=1.0) -> List[int]:
    clicks = []
    for i, y in enumerate(labels):
        p = position_propensity[min(i, len(position_propensity)-1)]
        # click prob increases with relevance
        base = 0.05 + 0.4 * y
        prob = max(0.0, min(1.0, eta * p * base))
        clicks.append(1 if random.random() < prob else 0)
    return clicks