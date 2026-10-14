"""Core request pipeline: cache -> route -> call provider (with fallback) -> account -> cache store."""
from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from .cache import SemanticCache
from .config import GatewayConfig, ModelConfig
from .metrics import Metrics
from .providers import Provider, ProviderError, build_provider
from .providers.base import CompletionResult, estimate_tokens
from .router import Router
from .schemas import ChatCompletionRequest, ChatCompletionResponse, ChatMessage, Choice, Usage, stream_chunk
from .usage import UsageTracker, request_cost


class NoHealthyModelError(Exception):
    def __init__(self, errors: list[str]):
        super().__init__("all candidate models failed or are unavailable")
        self.errors = errors


@dataclass
class RequestMeta:
    model: str | None = None
    provider: str | None = None
    cache: str = "MISS"           # MISS | HIT-exact | HIT-semantic | BYPASS
    route_reason: str = ""
    complexity: float | None = None
    attempts: list[str] = field(default_factory=list)

    def headers(self) -> dict[str, str]:
        h = {"X-Cache": self.cache, "X-Route-Reason": self.route_reason,
             "X-Attempts": ",".join(self.attempts)}
        if self.model:
            h["X-Gateway-Model"] = self.model
        if self.provider:
            h["X-Gateway-Provider"] = self.provider
        if self.complexity is not None:
            h["X-Complexity-Score"] = str(self.complexity)
        return h


class GatewayService:
    def __init__(self, config: GatewayConfig, providers: dict[str, Provider] | None = None):
        self.config = config
        self.providers = providers or {p.name: build_provider(p) for p in config.providers}
        self.models: dict[str, ModelConfig] = {m.name: m for m in config.models}
        self.router = Router(config.models, config.routing, config.circuit_breaker)
        self.cache = SemanticCache(dim=config.cache.embedding_dim, threshold=config.cache.similarity_threshold,
                                   ttl_seconds=config.cache.ttl_seconds,
                                   max_entries=config.cache.max_entries) if config.cache.enabled else None
        self.usage = UsageTracker()
        self.metrics = Metrics()

    # ---------------------------------------------------------------- helpers
    def _cacheable(self, req: ChatCompletionRequest, bypass: bool) -> bool:
        return self.cache is not None and not bypass and req.temperature <= self.config.cache.max_temperature

    def _refresh_gauges(self) -> None:
        for name, state in self.router.circuit_states().items():
            self.metrics.circuit_open.labels(name).set(1 if state.value == "open" else 0)
        if self.cache is not None:
            self.metrics.cache_entries.set(len(self.cache))

    def _build_response(self, model: ModelConfig, result: CompletionResult) -> dict:
        resp = ChatCompletionResponse(
            model=model.name,
            choices=[Choice(message=ChatMessage(role="assistant", content=result.content),
                            finish_reason=result.finish_reason)],
            usage=Usage(prompt_tokens=result.prompt_tokens, completion_tokens=result.completion_tokens,
                        total_tokens=result.prompt_tokens + result.completion_tokens),
        )
        return resp.model_dump()

    def _account(self, key_name: str, model: ModelConfig, prompt_tokens: int, completion_tokens: int) -> None:
        cost = request_cost(model, prompt_tokens, completion_tokens)
        self.metrics.tokens.labels(model.name, "prompt").inc(prompt_tokens)
        self.metrics.tokens.labels(model.name, "completion").inc(completion_tokens)
        self.metrics.cost.labels(model.name).inc(cost)
        self.usage.record(key_name, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, cost=cost)

    def _cache_hit(self, key_name: str, req: ChatCompletionRequest, meta: RequestMeta, cached: dict,
                   kind: str, started: float) -> None:
        model = self.models.get(cached["model"])
        u = cached["usage"]
        saved = request_cost(model, u["prompt_tokens"], u["completion_tokens"]) if model else 0.0
        meta.cache, meta.model = f"HIT-{kind}", cached["model"]
        meta.provider = model.provider if model else None
        self.metrics.saved.inc(saved)
        self.metrics.requests.labels(cached["model"], meta.provider or "", "ok", "hit").inc()
        self.metrics.latency.labels(cached["model"], "hit").observe(time.perf_counter() - started)
        self.usage.record(key_name, cache_hit=True, saved=saved)

    def _check_model(self, req: ChatCompletionRequest) -> None:
        if req.model != "auto" and req.model not in self.models:
            raise KeyError(req.model)

    # ---------------------------------------------------------- non-streaming
    async def complete(self, req: ChatCompletionRequest, key_name: str,
                       bypass_cache: bool = False) -> tuple[dict, RequestMeta]:
        started = time.perf_counter()
        self._check_model(req)
        meta = RequestMeta()
        use_cache = self._cacheable(req, bypass_cache)
        if not use_cache:
            meta.cache = "BYPASS"
        elif hit := self.cache.get(req):
            self._cache_hit(key_name, req, meta, hit.response, hit.kind, started)
            return hit.response, meta

        decision = self.router.route(req)
        meta.route_reason, meta.complexity = decision.reason, decision.complexity
        errors: list[str] = [f"{m}: circuit open" for m in decision.skipped_open_circuits]
        prev: str | None = None

        for model in decision.candidates:
            provider = self.providers[model.provider]
            meta.attempts.append(model.name)
            if prev:
                self.metrics.fallbacks.labels(prev, model.name).inc()
            t0 = time.perf_counter()
            try:
                result = await provider.complete(req, model.upstream)
            except ProviderError as exc:
                elapsed = time.perf_counter() - t0
                self.router.record(model.name, elapsed * 1000, ok=False)
                self.metrics.upstream_latency.labels(provider.name, model.name, "error").observe(elapsed)
                errors.append(str(exc))
                prev = model.name
                continue
            elapsed = time.perf_counter() - t0
            self.router.record(model.name, elapsed * 1000, ok=True)
            self.metrics.upstream_latency.labels(provider.name, model.name, "ok").observe(elapsed)

            body = self._build_response(model, result)
            meta.model, meta.provider = model.name, provider.name
            self._account(key_name, model, result.prompt_tokens, result.completion_tokens)
            if use_cache:
                self.cache.put(req, body)
            self.metrics.requests.labels(model.name, provider.name, "ok", "miss").inc()
            self.metrics.latency.labels(model.name, "miss").observe(time.perf_counter() - started)
            self._refresh_gauges()
            return body, meta

        self.usage.record(key_name, error=True)
        self.metrics.requests.labels("none", "none", "error", "miss").inc()
        self._refresh_gauges()
        raise NoHealthyModelError(errors)

    # -------------------------------------------------------------- streaming
    async def stream(self, req: ChatCompletionRequest, key_name: str,
                     bypass_cache: bool = False) -> tuple[AsyncIterator[str], RequestMeta]:
        """Returns (SSE line iterator, meta). Fallback happens *before* the first token is sent,
        so the client never sees a half answer from a failed provider followed by another one."""
        started = time.perf_counter()
        self._check_model(req)
        meta = RequestMeta()
        chunk_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        use_cache = self._cacheable(req, bypass_cache)
        if not use_cache:
            meta.cache = "BYPASS"
        elif hit := self.cache.get(req):
            self._cache_hit(key_name, req, meta, hit.response, hit.kind, started)
            content = hit.response["choices"][0]["message"]["content"]

            async def replay() -> AsyncIterator[str]:
                yield _sse(stream_chunk(chunk_id, meta.model, {"role": "assistant", "content": content}))
                yield _sse(stream_chunk(chunk_id, meta.model, {}, "stop"))
                yield "data: [DONE]\n\n"
            return replay(), meta

        decision = self.router.route(req)
        meta.route_reason, meta.complexity = decision.reason, decision.complexity
        errors: list[str] = [f"{m}: circuit open" for m in decision.skipped_open_circuits]

        for model in decision.candidates:
            provider = self.providers[model.provider]
            meta.attempts.append(model.name)
            t0 = time.perf_counter()
            it = provider.stream(req, model.upstream).__aiter__()
            try:
                first = await it.__anext__()
            except StopAsyncIteration:
                first = ""
            except ProviderError as exc:
                self.router.record(model.name, (time.perf_counter() - t0) * 1000, ok=False)
                errors.append(str(exc))
                if len(meta.attempts) > 1:
                    self.metrics.fallbacks.labels(meta.attempts[-2], model.name).inc()
                continue
            self.router.record(model.name, (time.perf_counter() - t0) * 1000, ok=True)
            meta.model, meta.provider = model.name, provider.name
            return self._relay(req, key_name, model, it, first, chunk_id, use_cache, started), meta

        self.usage.record(key_name, error=True)
        self.metrics.requests.labels("none", "none", "error", "miss").inc()
        self._refresh_gauges()
        raise NoHealthyModelError(errors)

    async def _relay(self, req: ChatCompletionRequest, key_name: str, model: ModelConfig, it: AsyncIterator[str],
                     first: str, chunk_id: str, use_cache: bool, started: float) -> AsyncIterator[str]:
        parts = [first]
        yield _sse(stream_chunk(chunk_id, model.name, {"role": "assistant", "content": first}))
        status = "ok"
        try:
            async for delta in it:
                parts.append(delta)
                yield _sse(stream_chunk(chunk_id, model.name, {"content": delta}))
            yield _sse(stream_chunk(chunk_id, model.name, {}, "stop"))
        except ProviderError as exc:
            status = "error"
            yield _sse({"error": {"message": f"upstream failed mid-stream: {exc}", "type": "upstream_error"}})
        yield "data: [DONE]\n\n"

        content = "".join(parts)
        prompt_tokens = estimate_tokens(" ".join(m.content for m in req.messages))
        completion_tokens = estimate_tokens(content)
        self._account(key_name, model, prompt_tokens, completion_tokens)
        if status == "ok" and use_cache:
            self.cache.put(req, self._build_response(model, CompletionResult(content, prompt_tokens,
                                                                               completion_tokens)))
        self.metrics.requests.labels(model.name, model.provider, status, "miss").inc()
        self.metrics.latency.labels(model.name, "miss").observe(time.perf_counter() - started)
        self._refresh_gauges()

    async def aclose(self) -> None:
        for p in self.providers.values():
            await p.aclose()


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"
