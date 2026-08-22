from __future__ import annotations

import time


def freshness_boost(
    timestamps: list[int | None],
    now: int | None = None,
    half_life_days: int = 14,
) -> list[float]:
    reference = int(time.time()) if now is None else now
    half_life_seconds = half_life_days * 86400
    return [
        0.0 if timestamp is None else 2 ** (-(max(0, reference - timestamp) / half_life_seconds))
        for timestamp in timestamps
    ]
