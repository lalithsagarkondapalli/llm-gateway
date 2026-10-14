"""Mock provider: simulates latency and failures so the gateway runs with no API keys.

Latency and failure rate can be changed at runtime (see POST /admin/providers/{name}/chaos),
which is how the outage-simulation benchmark works.
"""
from __future__ import annotations

import asyncio
import hashlib
import random
from collections.abc import AsyncIterator

from ..schemas import ChatCompletionRequest
from .base import CompletionResult, Provider, ProviderError, estimate_tokens


class MockProvider(Provider):
    def __init__(self, name: str, latency_ms: float = 50.0, failure_rate: float = 0.0, seed: int | None = None):
        super().__init__(name)
        self.latency_ms = latency_ms
        self.failure_rate = failure_rate
        self._rng = random.Random(seed)

    def _reply(self, request: ChatCompletionRequest, model: str) -> str:
        prompt = request.messages[-1].content
        digest = hashlib.sha1(prompt.encode()).hexdigest()[:8]
        return f"[{model}] Mock answer ({digest}) to: {prompt[:80]}"

    async def _maybe_fail_and_wait(self) -> None:
        jitter = self._rng.uniform(0.8, 1.2)
        await asyncio.sleep(self.latency_ms * jitter / 1000)
        if self._rng.random() < self.failure_rate:
            raise ProviderError(self.name, "simulated upstream failure", status_code=503)

    async def complete(self, request: ChatCompletionRequest, upstream_model: str) -> CompletionResult:
        await self._maybe_fail_and_wait()
        text = self._reply(request, upstream_model)
        prompt_text = " ".join(m.content for m in request.messages)
        return CompletionResult(text, estimate_tokens(prompt_text), estimate_tokens(text))

    async def stream(self, request: ChatCompletionRequest, upstream_model: str) -> AsyncIterator[str]:
        await self._maybe_fail_and_wait()
        for word in self._reply(request, upstream_model).split(" "):
            yield word + " "
            await asyncio.sleep(0.002)
