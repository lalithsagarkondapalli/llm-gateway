"""Quality-vs-cost eval: does smart routing save money without hurting answers?

Runs the same dataset through several routing setups (e.g. `auto` vs always-the-big-model)
and reports accuracy, cost and latency for each.

Needs REAL providers to be meaningful (mock providers return placeholder text, so accuracy
will be ~0 - the run still exercises the pipeline end to end):

    GATEWAY_CONFIG=config.real.yaml python -m gateway &
    python evals/run_eval.py --config config.real.yaml --models auto,gpt-4o --key $GATEWAY_ADMIN_KEY

Scoring: an answer is correct if it contains any of the `expected` strings (case-insensitive).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gateway.config import load_config  # noqa: E402
from gateway.usage import request_cost  # noqa: E402


def score(answer: str, expected: list[str]) -> bool:
    a = answer.lower().replace("**", "")
    return any(e.lower() in a for e in expected)


async def run_setup(client: httpx.AsyncClient, model: str, items: list[dict], prices: dict, concurrency: int):
    sem = asyncio.Semaphore(concurrency)
    rows = []

    async def one(item):
        async with sem:
            body = {"model": model, "temperature": 0, "messages": [{"role": "user", "content": item["prompt"]}]}
            t0 = time.perf_counter()
            r = await client.post("/v1/chat/completions", json=body, headers={"X-Cache-Bypass": "1"})
            latency = (time.perf_counter() - t0) * 1000
            if r.status_code != 200:
                rows.append({**item, "ok": False, "correct": False, "cost": 0, "latency_ms": latency,
                             "served_by": None, "error": r.text[:200]})
                return
            data = r.json()
            served = r.headers.get("x-gateway-model")
            usage = data["usage"]
            cost = request_cost(prices[served], usage["prompt_tokens"], usage["completion_tokens"]) \
                if served in prices else 0.0
            answer = data["choices"][0]["message"]["content"]
            rows.append({**item, "ok": True, "correct": score(answer, item["expected"]), "cost": cost,
                         "latency_ms": latency, "served_by": served, "answer": answer[-200:]})

    await asyncio.gather(*(one(i) for i in items))
    return rows


def summarize(model: str, rows: list[dict]) -> dict:
    def acc(subset):
        return round(sum(r["correct"] for r in subset) / len(subset), 4) if subset else None
    return {
        "setup": model,
        "accuracy": acc(rows),
        "accuracy_easy": acc([r for r in rows if r["difficulty"] == "easy"]),
        "accuracy_hard": acc([r for r in rows if r["difficulty"] == "hard"]),
        "total_cost_usd": round(sum(r["cost"] for r in rows), 6),
        "p50_latency_ms": round(statistics.median(r["latency_ms"] for r in rows), 1),
        "errors": sum(not r["ok"] for r in rows),
        "served_by": {m: sum(r["served_by"] == m for r in rows) for m in {r["served_by"] for r in rows}},
    }


async def main_async(args):
    prices = {m.name: m for m in load_config(args.config).models}
    items = [json.loads(line) for line in Path(args.dataset).read_text().splitlines() if line.strip()]
    results, details = [], {}
    async with httpx.AsyncClient(base_url=args.url, timeout=120,
                                 headers={"Authorization": f"Bearer {args.key}"}) as client:
        for model in args.models.split(","):
            rows = await run_setup(client, model, items, prices, args.concurrency)
            results.append(summarize(model, rows))
            details[model] = rows

    base = results[-1]
    print(f"\n{'setup':<16}{'accuracy':>10}{'easy':>8}{'hard':>8}{'cost $':>12}{'vs last':>10}{'p50 ms':>10}")
    for r in results:
        rel = f"{(1 - r['total_cost_usd'] / base['total_cost_usd']) * 100:.0f}% less" \
            if base["total_cost_usd"] and r is not base else "-"
        print(f"{r['setup']:<16}{r['accuracy']:>10.1%}{r['accuracy_easy']:>8.1%}{r['accuracy_hard']:>8.1%}"
              f"{r['total_cost_usd']:>12.5f}{rel:>10}{r['p50_latency_ms']:>10}")
    if args.out:
        Path(args.out).write_text(json.dumps({"summary": results, "details": details}, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--key", default="sk-admin-dev")
    ap.add_argument("--config", default="config.yaml", help="gateway config (for model prices)")
    ap.add_argument("--dataset", default=str(Path(__file__).with_name("dataset.jsonl")))
    ap.add_argument("--models", default="auto,gamma-pro",
                    help="comma-separated setups; put the 'always use the big model' baseline last")
    ap.add_argument("--concurrency", type=int, default=5)
    ap.add_argument("--out")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
