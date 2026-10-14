"""Offline benchmark of the semantic cache: hit rate, precision (wrong-answer rate) and cost saved.

Builds synthetic traffic: N base questions, each asked in several phrasings, with popularity
following a Zipf-like distribution (a few questions are very common - like real traffic).

A configurable share of traffic (--unique-frac) is one-off long-tail questions that are never
repeated, so the cache can't hit on everything.

A hit is *correct* if the cached answer came from the same base question, *false* otherwise.
False hits are the real risk of semantic caching (serving the wrong answer), so we sweep the
similarity threshold to show the hit-rate vs precision trade-off.

    python bench/cache_benchmark.py --requests 10000
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gateway.cache import SemanticCache  # noqa: E402
from gateway.schemas import ChatCompletionRequest  # noqa: E402

TOPICS = [
    "a linked list", "binary search", "a hash map", "recursion", "dynamic programming", "a REST API",
    "Docker", "Kubernetes", "a database index", "SQL joins", "garbage collection", "the CAP theorem",
    "OAuth", "JWT tokens", "TCP vs UDP", "DNS", "load balancing", "caching", "a message queue", "Git rebase",
    "Python decorators", "JavaScript closures", "React hooks", "async/await", "a mutex", "deadlocks",
    "Big O notation", "quicksort", "merge sort", "a binary tree", "graph BFS", "Dijkstra's algorithm",
    "gradient descent", "overfitting", "a neural network", "transformers", "embeddings", "vector databases",
    "unit testing", "CI/CD", "microservices", "event sourcing", "rate limiting", "consistent hashing",
    "sharding", "replication", "WebSockets", "GraphQL", "HTTP/2", "TLS",
]
TEMPLATES = [
    "What is {t}?",
    "what's {t}",
    "Can you explain {t}?",
    "Explain {t} please",
    "Please explain {t} to me",
    "What is {t}? Explain simply.",
    "Could you tell me what {t} is?",
    "explain {t}",
]
TASKS = ["What is {t}?", "How do I debug {t}?", "Give me an example of {t}", "What are the downsides of {t}?"]


def build_questions(rng: random.Random, n_base: int) -> list[tuple[int, list[str]]]:
    """Return (base_id, phrasings) for n_base distinct intents."""
    questions = []
    for topic in TOPICS:
        for task in TASKS:
            base = task.format(t=topic)
            if task == "What is {t}?":
                phrasings = [tpl.format(t=topic) for tpl in TEMPLATES]
            else:
                phrasings = [base, base.lower(), base.rstrip("?") + " please?", base.replace("?", "")]
            questions.append(phrasings)
    rng.shuffle(questions)
    return list(enumerate(questions[:n_base]))


def req(text: str) -> ChatCompletionRequest:
    return ChatCompletionRequest.model_validate({"messages": [{"role": "user", "content": text}]})


def simulate(threshold: float, traffic: list[tuple[int, str]], cost_per_call: float) -> dict:
    """threshold > 1 disables the semantic layer -> exact-match-only baseline."""
    cache = SemanticCache(threshold=threshold, max_entries=100_000)
    correct = false = 0
    for base_id, text in traffic:
        r = req(text)
        hit = cache.get(r)
        if hit:
            correct += hit.response["base_id"] == base_id
            false += hit.response["base_id"] != base_id
        else:
            cache.put(r, {"base_id": base_id})
    n = len(traffic)
    hits = correct + false
    return {
        "threshold": "exact-only" if threshold > 1 else threshold,
        "hit_rate": round(hits / n, 4),
        "semantic_hits": cache.semantic_hits,
        "precision": round(correct / hits, 4) if hits else 1.0,
        "false_hits": false,
        "upstream_calls_avoided": hits,
        "cost_saved_pct": round(hits / n * 100, 2),
        "cost_saved_usd": round(hits * cost_per_call, 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", type=int, default=10_000)
    ap.add_argument("--base-questions", type=int, default=200)
    ap.add_argument("--zipf", type=float, default=1.1)
    ap.add_argument("--unique-frac", type=float, default=0.4, help="share of one-off, never-repeated prompts")
    ap.add_argument("--cost-per-call", type=float, default=0.00042,
                    help="USD per upstream call (default ~ 300 in / 600 out tokens on a small model)")
    ap.add_argument("--thresholds", default="1.01,0.80,0.85,0.90,0.92,0.95")
    ap.add_argument("--out")
    args = ap.parse_args()

    rng = random.Random(7)
    questions = build_questions(rng, args.base_questions)
    weights = [1 / (rank + 1) ** args.zipf for rank in range(len(questions))]
    traffic = []
    for i in range(args.requests):
        if rng.random() < args.unique_frac:
            topic, task = rng.choice(TOPICS), rng.choice(["compare {t} with", "write a poem about {t} and",
                                                           "summarize this ticket about {t} for", "translate {t} for"])
            traffic.append((-(i + 1), f"{task.format(t=topic)} customer #{i} order {rng.randint(1, 10**6)}"))
            continue
        base_id, phrasings = rng.choices(questions, weights=weights)[0]
        traffic.append((base_id, rng.choice(phrasings)))

    results = [simulate(float(t), traffic, args.cost_per_call) for t in args.thresholds.split(",")]
    report = {"requests": args.requests, "distinct_intents": len(questions), "unique_frac": args.unique_frac,
              "results": results}
    print(json.dumps(report, indent=2))
    print("\nthreshold    hit_rate  semantic_hits  precision  false_hits")
    for r in results:
        print(f"  {r['threshold']!s:<11} {r['hit_rate']:>7.1%} {r['semantic_hits']:>13} {r['precision']:>10.2%} "
              f"{r['false_hits']:>10}")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
