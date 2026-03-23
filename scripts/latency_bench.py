import argparse, asyncio, httpx, time, numpy as np

p = argparse.ArgumentParser(description="Load test the NeuroRank API with realistic payloads.")
p.add_argument("--endpoint", required=True, help="e.g., http://localhost:8000/rerank")
p.add_argument("--qps", type=int, default=10, help="Number of concurrent workers")
p.add_argument("--seconds", type=int, default=10, help="Duration of the benchmark")
p.add_argument("--batch_size", type=int, default=50, help="Number of candidate passages per request")
args = p.parse_args()

# Generate a realistic mock passage (approx 60 words, typical for MS MARCO)
mock_passage = (
    "Neural ranking models, particularly those based on transformer architectures "
    "have revolutionized information retrieval. Unlike traditional lexical models "
    "such as BM25, neural models capture deep semantic context. Knowledge distillation "
    "further allows us to compress these massive teacher models into efficient student "
    "models, maintaining high accuracy while significantly reducing inference latency, "
    "making them suitable for real-time production deployment at scale."
)

# Build a payload that simulates a real first-stage retriever output
payload = {
    "query": "what is a neural ranking model and how does knowledge distillation improve it?",
    "texts": [mock_passage for _ in range(args.batch_size)],
    "freshness_timestamps": [int(time.time()) for _ in range(args.batch_size)],
    "use_freshness": False,
    "top_k": 10,
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
            print(f"Request error: {exc}")
        except httpx.HTTPStatusError as exc:
            print(f"HTTP error {exc.response.status_code}: {exc.response.text}")
        await asyncio.sleep(0)  # Yield control to event loop

async def main():
    # Increased timeout to handle the larger payload processing
    async with httpx.AsyncClient(timeout=10.0) as client:
        end = time.time() + args.seconds
        all_latencies = []
        tasks = [
            asyncio.create_task(worker(client, end, all_latencies))
            for _ in range(args.qps)
        ]
        
        print(f"Warming up and running benchmark for {args.seconds} seconds...")
        print(f"Targeting {args.batch_size} passages per request with {args.qps} concurrent workers.")
        
        await asyncio.gather(*tasks)

        if all_latencies:
            p50 = np.percentile(all_latencies, 50)
            p95 = np.percentile(all_latencies, 95)
            p99 = np.percentile(all_latencies, 99)
            print(f"\n--- Benchmark Complete ---")
            print(f"Total requests completed: {len(all_latencies)}")
            print(f"p50 latency: {p50:.2f} ms")
            print(f"p95 latency: {p95:.2f} ms")
            print(f"p99 latency: {p99:.2f} ms")
            print(f"Average throughput: {len(all_latencies)/args.seconds:.2f} req/sec")
        else:
            print("No requests were completed. Check the server endpoint and logs.")

if __name__ == "__main__":
    asyncio.run(main())