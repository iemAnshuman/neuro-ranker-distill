from __future__ import annotations

import argparse
import asyncio
import time

import httpx
import numpy as np


def build_payload(batch_size: int) -> dict:
    passage = (
        "Neural ranking models score candidate passages for a query. Knowledge "
        "distillation transfers ranking behavior from a larger teacher to a smaller student."
    )
    return {
        "query": "how does neural ranking distillation work?",
        "texts": [passage] * batch_size,
        "use_freshness": False,
        "top_k": min(10, batch_size),
    }


async def worker(
    client,
    endpoint: str,
    deadline: float,
    payload: dict,
    results: list[float],
) -> None:
    while time.perf_counter() < deadline:
        started = time.perf_counter()
        response = await client.post(endpoint, json=payload)
        response.raise_for_status()
        results.append((time.perf_counter() - started) * 1000)


async def run(args) -> None:
    results: list[float] = []
    payload = build_payload(args.batch_size)
    async with httpx.AsyncClient(timeout=30.0) as client:
        deadline = time.perf_counter() + args.seconds
        await asyncio.gather(
            *(
                worker(client, args.endpoint, deadline, payload, results)
                for _ in range(args.workers)
            )
        )
    if not results:
        raise RuntimeError("no requests completed")
    report = {
        "requests": len(results),
        "seconds": args.seconds,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "requests_per_second": len(results) / args.seconds,
        "latency_ms": {
            "p50": float(np.percentile(results, 50)),
            "p95": float(np.percentile(results, 95)),
            "p99": float(np.percentile(results, 99)),
        },
    }
    print(report)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/rerank")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seconds", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=50)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
