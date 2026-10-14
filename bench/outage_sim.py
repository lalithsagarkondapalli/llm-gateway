"""Outage simulation: knock out the primary provider mid-load and measure availability.

Phases (each --phase seconds):
  1. baseline  - all providers healthy
  2. outage    - primary provider fails 100% of requests (via the chaos endpoint)
  3. recovery  - provider restored; circuit breaker should close again after cooldown

Requests use unique prompts so the cache can't hide the outage.

    python bench/outage_sim.py --provider mock-alpha --phase 10
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections import Counter, defaultdict

import httpx


async def main_async(args) -> dict:
    headers = {"Authorization": f"Bearer {args.key}"}
    phase_stats: dict[str, Counter] = defaultdict(Counter)
    phase_models: dict[str, Counter] = defaultdict(Counter)
    phase = "baseline"
    seq = 0
    stop = False

    async with httpx.AsyncClient(base_url=args.url, headers=headers, timeout=30) as client:
        async def chaos(rate: float):
            r = await client.post(f"/admin/providers/{args.provider}/chaos", json={"failure_rate": rate})
            r.raise_for_status()

        async def worker():
            nonlocal seq
            while not stop:
                seq += 1
                body = {"messages": [{"role": "user", "content": f"outage test {seq} {time.time()}"}]}
                current = phase
                try:
                    r = await client.post("/v1/chat/completions", json=body)
                    phase_stats[current]["ok" if r.status_code == 200 else f"http_{r.status_code}"] += 1
                    phase_models[current][r.headers.get("x-gateway-model", "none")] += 1
                    if r.headers.get("x-attempts", "").count(",") > 0:
                        phase_stats[current]["served_after_fallback"] += 1
                except httpx.HTTPError:
                    phase_stats[current]["network_error"] += 1

        await chaos(0.0)
        workers = [asyncio.create_task(worker()) for _ in range(args.concurrency)]
        await asyncio.sleep(args.phase)
        phase = "outage"
        await chaos(1.0)
        await asyncio.sleep(args.phase)
        phase = "recovery"
        await chaos(0.0)
        await asyncio.sleep(args.phase)
        stop = True
        await asyncio.gather(*workers)

    report = {}
    for name in ("baseline", "outage", "recovery"):
        s = phase_stats[name]
        total = sum(v for k, v in s.items() if k != "served_after_fallback")
        report[name] = {"requests": total, "success_rate": round(s["ok"] / total, 4) if total else None,
                        "served_after_fallback": s["served_after_fallback"], "served_by": dict(phase_models[name]),
                        "outcomes": {k: v for k, v in s.items() if k != "served_after_fallback"}}
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--key", default="sk-admin-dev")
    ap.add_argument("--provider", default="mock-alpha")
    ap.add_argument("--phase", type=float, default=10.0, help="seconds per phase")
    ap.add_argument("--concurrency", type=int, default=20)
    ap.add_argument("--out")
    args = ap.parse_args()
    report = asyncio.run(main_async(args))
    print(json.dumps(report, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
