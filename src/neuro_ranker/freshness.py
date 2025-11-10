import math, time
from typing import List, Optional

# Exponential decay prior: newer = higher


def freshness_boost(
    timestamps: List[Optional[int]], now: Optional[int] = None, half_life_days: int = 14
):
    now = now or int(time.time())
    out = []
    hl = half_life_days * 86400
    for ts in timestamps:
        if ts is None:
            out.append(0.0)
        else:
            dt = max(0, now - ts)
            boost = 2 ** (-(dt / hl))  # 1.0 when ts=now, 0.5 at half‑life
            out.append(boost)
    return out
