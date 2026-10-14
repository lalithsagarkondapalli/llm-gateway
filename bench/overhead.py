"""Gateway overhead: how much latency does the gateway itself add on top of the provider?

Uses a zero-latency mock provider so every millisecond measured is gateway work:
auth, rate limiting, cache lookup (embedding + vector search), routing, accounting, metrics,
JSON (de)serialization and the FastAPI/ASGI stack. Runs in-process (httpx ASGITransport), so the
numbers aren't distorted by the load generator competing for CPU.

    python bench/overhead.py --requests 5000 --cache-entries 10000
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gateway.app import create_app  # noqa: E402
from gateway.config import GatewayConfig  # noqa: E402
from gateway.providers import MockProvider  # noqa: E402
from gateway.service import GatewayService  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from loadtest import percentile  # noqa: E402

CONFIG = {
    "auth": {"keys": [{"key": "sk-bench-key", "name": "bench", "rate_limit_rpm": 10**9, "admin": True}]},
    "cache": {"enabled": True, "max_entries": 200_000},
    "providers": [{"name": "zero", "type": "mock", "latency_ms": 0}],
    "models": [{"name": "m", "provider": "zero", "input_cost_per_1k": 0.001, "output_cost_per_1k": 0.002}],
}


def stats(samples: list[float]) -> dict:
    return {p: round(percentile(samples, int(p[1:])), 3) for p in ("p50", "p95", "p99")}


async def main_async(args) -> dict:
    cfg = GatewayConfig.model_validate(CONFIG)
    service = GatewayService(cfg, providers={"zero": MockProvider("zero", latency_ms=0, seed=0)})
    app = create_app(cfg, service)

    # pre-fill the cache so lookups search a realistically sized index
    for i in range(args.cache_entries):
        r = {"messages": [{"role": "user", "content": f"warm entry {i} about subject {i * 7}"}]}
        from gateway.schemas import ChatCompletionRequest
        service.cache.put(ChatCompletionRequest.model_validate(r), {"model": "m", "usage": {}})

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://bench",
                                 headers={"Authorization": "Bearer sk-bench-key"}) as client:
        async def timed(body):
            t0 = time.perf_counter()
            r = await client.post("/v1/chat/completions", json=body)
            assert r.status_code == 200, r.text
            return (time.perf_counter() - t0) * 1000

        miss = [await timed({"messages": [{"role": "user", "content": f"fresh question {i} on topic x{i}"}]})
                for i in range(args.requests)]
        hit = [await timed({"messages": [{"role": "user", "content": f"fresh question {i % 100} on topic x{i % 100}"}]})
               for i in range(args.requests)]
        nocache = [await timed({"temperature": 1.0, "messages": [{"role": "user", "content": f"q {i}"}]})
                   for i in range(args.requests)]

    return {
        "requests_per_scenario": args.requests,
        "cache_entries_at_start": args.cache_entries,
        "overhead_ms": {
            "cache_miss_full_pipeline": stats(miss),
            "cache_hit": stats(hit),
            "cache_disabled": stats(nocache),
        },
        "single_core_capacity_rps_estimate": round(1000 / percentile(miss, 50)),
        "python": sys.version.split()[0],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", type=int, default=3000)
    ap.add_argument("--cache-entries", type=int, default=10_000)
    ap.add_argument("--out")
    args = ap.parse_args()
    report = asyncio.run(main_async(args))
    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
