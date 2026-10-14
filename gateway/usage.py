"""Per-API-key usage and cost accounting."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass

from .config import ModelConfig


def request_cost(model: ModelConfig, prompt_tokens: int, completion_tokens: int) -> float:
    return (prompt_tokens / 1000) * model.input_cost_per_1k + (completion_tokens / 1000) * model.output_cost_per_1k


@dataclass
class KeyUsage:
    requests: int = 0
    errors: int = 0
    cache_hits: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    saved_by_cache_usd: float = 0.0


class UsageTracker:
    def __init__(self):
        self._usage: dict[str, KeyUsage] = defaultdict(KeyUsage)

    def record(self, key_name: str, *, prompt_tokens: int = 0, completion_tokens: int = 0, cost: float = 0.0,
               cache_hit: bool = False, saved: float = 0.0, error: bool = False) -> None:
        u = self._usage[key_name]
        u.requests += 1
        u.errors += int(error)
        u.cache_hits += int(cache_hit)
        u.prompt_tokens += prompt_tokens
        u.completion_tokens += completion_tokens
        u.cost_usd += cost
        u.saved_by_cache_usd += saved

    def report(self) -> dict:
        return {k: {**asdict(v), "cost_usd": round(v.cost_usd, 6),
                    "saved_by_cache_usd": round(v.saved_by_cache_usd, 6)}
                for k, v in self._usage.items()}
