"""Async load test against a running gateway.

    python bench/loadtest.py --requests 5000 --concurrency 100
    python bench/loadtest.py --unique        # every prompt unique -> measures the uncached path
    python bench/loadtest.py --pool 200      # prompts drawn from 200 questions -> realistic cache mix

Reports throughput, latency percentiles, error rate and cache hit rate.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import time
from collections import Counter

import httpx


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    k = (len(values) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


async def run(args) -> dict:
    headers = {"Authorization": f"Bearer {args.key}"}
    latencies: list[float] = []
    statuses: Counter = Counter()
    cache: Counter = Counter()
    models: Counter = Counter()
    rng = random.Random(42)
    counter = iter(range(args.requests))
    limits = httpx.Limits(max_connections=args.concurrency, max_keepalive_connections=args.concurrency)

    async with httpx.AsyncClient(base_url=args.url, headers=headers, timeout=60, limits=limits) as client:
        async def worker():
            for i in counter:
                q = f"Question number {i}: explain topic {i}" if args.unique else \
                    f"Explain topic {int(rng.paretovariate(1.2)) % args.pool}"
                body = {"model": args.model, "messages": [{"role": "user", "content": q}]}
                t0 = time.perf_counter()
                try:
                    r = await client.post("/v1/chat/completions", json=body)
                    statuses[r.status_code] += 1
                    cache[r.headers.get("x-cache", "?").split("-")[0]] += 1
                    models[r.headers.get("x-gateway-model", "none")] += 1
                except httpx.HTTPError:
                    statuses["network_error"] += 1
                latencies.append((time.perf_counter() - t0) * 1000)

        started = time.perf_counter()
        await asyncio.gather(*(worker() for _ in range(args.concurrency)))
        elapsed = time.perf_counter() - started

    ok = statuses.get(200, 0)
    return {
        "requests": args.requests,
        "concurrency": args.concurrency,
        "duration_s": round(elapsed, 2),
        "throughput_rps": round(args.requests / elapsed, 1),
        "latency_ms": {"p50": round(percentile(latencies, 50), 1), "p95": round(percentile(latencies, 95), 1),
                       "p99": round(percentile(latencies, 99), 1), "mean": round(statistics.fmean(latencies), 1)},
        "success_rate": round(ok / args.requests, 4),
        "status_codes": dict(statuses),
        "cache_hit_rate": round(cache.get("HIT", 0) / max(ok, 1), 4),
        "served_by": dict(models),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--key", default="sk-admin-dev")
    ap.add_argument("--model", default="auto")
    ap.add_argument("--requests", type=int, default=2000)
    ap.add_argument("--concurrency", type=int, default=50)
    ap.add_argument("--pool", type=int, default=200, help="distinct questions when not --unique")
    ap.add_argument("--unique", action="store_true")
    ap.add_argument("--out", help="write JSON results to this file")
    args = ap.parse_args()
    result = asyncio.run(run(args))
    print(json.dumps(result, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
