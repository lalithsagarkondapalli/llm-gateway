"""Routing: decide which models to try, in order, for a request.

The router returns an ordered list of candidates. The gateway tries them in order,
falling back to the next one when a call fails or a circuit breaker is open.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .circuit_breaker import CircuitBreaker, State
from .config import CircuitBreakerConfig, ModelConfig, RoutingConfig
from .schemas import ChatCompletionRequest

_HARD_KEYWORDS = re.compile(
    r"\b(prove|derive|step[- ]by[- ]step|analy[sz]e|debug|refactor|optimi[sz]e|architecture|"
    r"trade-?offs?|compare|explain why|reason|algorithm|complexity|implement|design|"
    r"theorem|integral|regex|sql|concurrency)\b",
    re.IGNORECASE,
)
_CODE_HINT = re.compile(r"```|def |class |function |SELECT |#include|=>|\{\s*\n")


def complexity_score(request: ChatCompletionRequest) -> float:
    """Cheap heuristic in [0, 1]: higher means the prompt likely needs a stronger model.

    Signals: prompt length, reasoning/coding keywords, code blocks, multi-turn depth.
    Deliberately simple and explainable; swap in a small classifier later (see README roadmap).
    """
    text = " ".join(m.content for m in request.messages if m.role != "assistant")
    words = len(text.split())
    score = 0.0
    score += min(words / 400, 1.0) * 0.3                         # long prompts
    score += min(len(_HARD_KEYWORDS.findall(text)) / 2, 1.0) * 0.5  # reasoning keywords
    score += 0.25 if _CODE_HINT.search(text) else 0.0            # code present
    score += min(len(request.messages) / 10, 1.0) * 0.05         # long conversations
    return round(min(score, 1.0), 3)


@dataclass
class ModelStats:
    ewma_latency_ms: float | None = None
    requests: int = 0
    failures: int = 0
    alpha: float = 0.2

    def observe(self, latency_ms: float, ok: bool) -> None:
        self.requests += 1
        if not ok:
            self.failures += 1
            return
        self.ewma_latency_ms = latency_ms if self.ewma_latency_ms is None else (
            self.alpha * latency_ms + (1 - self.alpha) * self.ewma_latency_ms)


@dataclass
class RouteDecision:
    candidates: list[ModelConfig]
    reason: str
    complexity: float | None = None
    skipped_open_circuits: list[str] = field(default_factory=list)


class Router:
    def __init__(self, models: list[ModelConfig], routing: RoutingConfig, breaker_cfg: CircuitBreakerConfig):
        self.models = {m.name: m for m in models}
        self.cfg = routing
        self.breakers = {m.name: CircuitBreaker(breaker_cfg.failure_threshold, breaker_cfg.cooldown_seconds)
                         for m in models}
        self.stats = {m.name: ModelStats() for m in models}

    @staticmethod
    def _cost(m: ModelConfig) -> float:
        return m.input_cost_per_1k + m.output_cost_per_1k

    def _latency(self, m: ModelConfig) -> float:
        lat = self.stats[m.name].ewma_latency_ms
        return lat if lat is not None else 0.0  # unexplored models get tried first

    def _order(self, request: ChatCompletionRequest) -> tuple[list[ModelConfig], str, float | None]:
        models = list(self.models.values())
        if request.model != "auto":
            if request.model not in self.models:
                raise KeyError(request.model)
            primary = self.models[request.model]
            # fallbacks: same tier first (cheapest first), then the other tier
            rest = sorted((m for m in models if m.name != primary.name),
                          key=lambda m: (m.tier != primary.tier, self._cost(m)))
            return [primary, *rest], "explicit", None

        if self.cfg.strategy == "cost":
            return sorted(models, key=self._cost), "cost", None
        if self.cfg.strategy == "latency":
            return sorted(models, key=self._latency), "latency", None

        score = complexity_score(request)
        want = "large" if score >= self.cfg.complexity_threshold else "small"
        ordered = sorted(models, key=lambda m: (m.tier != want, self._cost(m)))
        return ordered, f"complexity:{want}", score

    def route(self, request: ChatCompletionRequest, allow_fallback: bool = True) -> RouteDecision:
        """allow_fallback=False pins the request to the single top choice (used by evals, so a
        rate-limited model is retried by the caller instead of silently swapped for another)."""
        ordered, reason, score = self._order(request)
        if not allow_fallback:
            top = ordered[0]
            ok = self.breakers[top.name].allow_request()
            return RouteDecision([top] if ok else [], reason + ":pinned", score, [] if ok else [top.name])
        allowed, skipped = [], []
        for m in ordered:
            if len(allowed) >= self.cfg.max_attempts:
                break  # don't consume a half-open trial slot we won't use
            (allowed if self.breakers[m.name].allow_request() else skipped).append(m)
        return RouteDecision(allowed, reason, score, [m.name for m in skipped])

    def record(self, model: str, latency_ms: float, ok: bool) -> None:
        self.stats[model].observe(latency_ms, ok)
        breaker = self.breakers[model]
        breaker.record_success() if ok else breaker.record_failure()

    def snapshot(self) -> list[dict]:
        out = []
        for name, m in self.models.items():
            s = self.stats[name]
            out.append({
                "model": name,
                "provider": m.provider,
                "tier": m.tier,
                "circuit": self.breakers[name].state.value,
                "requests": s.requests,
                "failures": s.failures,
                "ewma_latency_ms": round(s.ewma_latency_ms, 1) if s.ewma_latency_ms else None,
            })
        return out

    def circuit_states(self) -> dict[str, State]:
        return {name: b.state for name, b in self.breakers.items()}
