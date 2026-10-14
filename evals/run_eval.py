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
import re
import statistics
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gateway.config import load_config  # noqa: E402
from gateway.usage import request_cost  # noqa: E402


def score(answer: str, expected: list[str]) -> bool:
    """Correct if any expected string appears as a whole token, so "9" doesn't match "19" or "0.9"."""
    a = re.sub(r"[*_`]", "", answer.lower())
    a = re.sub(r"\s+", " ", a)
    for e in expected:
        pattern = r"(?<![\w.])" + re.escape(e.lower()) + r"(?![\w]|\.\d)"
        if re.search(pattern, a):
            return True
    return False


class Pacer:
    """Spaces request starts to stay under a provider's requests-per-minute limit (free tiers)."""

    def __init__(self, rpm: float):
        self.interval = 60.0 / rpm if rpm > 0 else 0.0
        self._next = 0.0
        self._lock = asyncio.Lock()

    async def wait(self):
        async with self._lock:
            now = time.monotonic()
            delay = max(0.0, self._next - now)
            self._next = max(now, self._next) + self.interval
        if delay:
            await asyncio.sleep(delay)


async def run_setup(client: httpx.AsyncClient, model: str, items: list[dict], prices: dict, concurrency: int,
                    pacer: Pacer, retries: int = 5):
    sem = asyncio.Semaphore(concurrency)
    rows = []

    async def one(item):
        async with sem:
            body = {"model": model, "temperature": 0, "messages": [{"role": "user", "content": item["prompt"]}]}
            for attempt in range(retries + 1):
                await pacer.wait()
                t0 = time.perf_counter()
                r = await client.post("/v1/chat/completions", json=body,
                                      headers={"X-Cache-Bypass": "1", "X-No-Fallback": "1"})
                latency = (time.perf_counter() - t0) * 1000
                if r.status_code not in (429, 503) or attempt == retries:
                    break
                await asyncio.sleep(min(10 * (attempt + 1), 30))  # rate-limited: back off, retry same model
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
    rows.sort(key=lambda r: r["id"])
    return rows


def summarize(model: str, rows: list[dict], large: set[str] | None = None) -> dict:
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
        "routed_as_intended": sum(
            (r["served_by"] in large) == (r["difficulty"] == "hard") for r in rows if r["ok"]
        ) if large else None,
        "served_by": {m: sum(r["served_by"] == m for r in rows) for m in {r["served_by"] for r in rows}},
    }


async def main_async(args):
    models_cfg = load_config(args.config).models
    prices = {m.name: m for m in models_cfg}
    large = {m.name for m in models_cfg if m.tier == "large"}
    pacer = Pacer(args.rpm)
    items = [json.loads(line) for line in Path(args.dataset).read_text().splitlines() if line.strip()]
    results, details = [], {}
    async with httpx.AsyncClient(base_url=args.url, timeout=120,
                                 headers={"Authorization": f"Bearer {args.key}"}) as client:
        for model in args.models.split(","):
            print(f"running setup '{model}' on {len(items)} prompts...", flush=True)
            rows = await run_setup(client, model, items, prices, args.concurrency, pacer)
            results.append(summarize(model, rows, large if model == "auto" else None))
            details[model] = rows

    base = results[-1]
    print(f"\n{'setup':<16}{'accuracy':>10}{'easy':>8}{'hard':>8}{'cost $':>12}{'vs last':>10}{'p50 ms':>10}")
    for r in results:
        rel = f"{(1 - r['total_cost_usd'] / base['total_cost_usd']) * 100:.0f}% less" \
            if base["total_cost_usd"] and r is not base else "-"
        print(f"{r['setup']:<16}{r['accuracy']:>10.1%}{r['accuracy_easy']:>8.1%}{r['accuracy_hard']:>8.1%}"
              f"{r['total_cost_usd']:>12.5f}{rel:>10}{r['p50_latency_ms']:>10}")
    if args.out:
        meta = {"date": time.strftime("%Y-%m-%d"), "config": args.config, "dataset_size": len(items),
                "models": {m.name: {"upstream": m.upstream, "tier": m.tier} for m in models_cfg}}
        Path(args.out).write_text(json.dumps({"meta": meta, "summary": results, "details": details}, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--key", default="sk-admin-dev")
    ap.add_argument("--config", default="config.yaml", help="gateway config (for model prices)")
    ap.add_argument("--dataset", default=str(Path(__file__).with_name("dataset.jsonl")))
    ap.add_argument("--models", default="auto,gamma-pro",
                    help="comma-separated setups; put the 'always use the big model' baseline last")
    ap.add_argument("--concurrency", type=int, default=5)
    ap.add_argument("--rpm", type=float, default=0, help="max requests/min across the run (0 = unpaced)")
    ap.add_argument("--out")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
