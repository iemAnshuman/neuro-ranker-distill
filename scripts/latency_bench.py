import argparse, asyncio, httpx, random, time, numpy as np

p = argparse.ArgumentParser()
p.add_argument("--endpoint", required=True)
p.add_argument("--qps", type=int, default=100, help="Number of concurrent workers")
p.add_argument("--seconds", type=int, default=10)
args = p.parse_args()

payload = {
    "query": "what is a neural ranking model",
    "texts": [
        "Neural ranking models learn to order documents given a query.",
        "Inverse propensity scoring debiases clicks.",
    ],
    "freshness_timestamps": [1724899200, 1726800000],
    "use_freshness": True,
    "banned_terms": ["nsfw"],
    "top_k": 2,
}


async def worker(client, end, results):
    while time.time() < end:
        t0 = time.time()
        try:
            r = await client.post(args.endpoint, json=payload)
            r.raise_for_status()
            latency = (time.time() - t0) * 1000
            results.append(latency)
        except httpx.RequestError as exc:
            print(f"An error occurred while requesting {exc.request.url!r}.")
        await asyncio.sleep(0)  # Yield control to event loop


async def main():
    async with httpx.AsyncClient(timeout=5.0) as client:
        end = time.time() + args.seconds
        all_latencies = []
        tasks = [
            asyncio.create_task(worker(client, end, all_latencies))
            for _ in range(args.qps)
        ]
        await asyncio.gather(*tasks)

        if all_latencies:
            p50 = np.percentile(all_latencies, 50)
            p95 = np.percentile(all_latencies, 95)
            print(f"Benchmark complete. Total requests: {len(all_latencies)}")
            print(f"p50 latency: {p50:.2f} ms")
            print(f"p95 latency: {p95:.2f} ms")
        else:
            print("No requests were completed.")


if __name__ == "__main__":
    asyncio.run(main())
