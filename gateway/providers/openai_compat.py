"""Adapter for any OpenAI-compatible API: OpenAI, Groq, Together, Ollama, vLLM, LM Studio..."""
from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx

from ..schemas import ChatCompletionRequest
from .base import CompletionResult, Provider, ProviderError, estimate_tokens

RETRYABLE = {408, 409, 429, 500, 502, 503, 504}


def _error_detail(resp: httpx.Response) -> str:
    """Pull the provider's own error message (e.g. 'model X does not exist') for logs and 503 bodies."""
    try:
        err = resp.json().get("error", {})
        msg = err.get("message") if isinstance(err, dict) else str(err)
    except Exception:
        msg = resp.text
    return (msg or resp.reason_phrase or "").strip()[:200]


class OpenAICompatProvider(Provider):
    def __init__(self, name: str, base_url: str, api_key: str | None, timeout: float = 30.0,
                 client: httpx.AsyncClient | None = None):
        super().__init__(name)
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = client or httpx.AsyncClient(base_url=base_url.rstrip("/"), headers=headers, timeout=timeout)

    def _payload(self, request: ChatCompletionRequest, upstream_model: str, stream: bool) -> dict:
        body = {
            "model": upstream_model,
            "messages": [m.model_dump() for m in request.messages],
            "temperature": request.temperature,
            "stream": stream,
        }
        if request.max_tokens is not None:
            body["max_tokens"] = request.max_tokens
        return body

    def _raise_for(self, resp: httpx.Response) -> None:
        if resp.status_code >= 400:
            raise ProviderError(self.name, f"HTTP {resp.status_code}: {_error_detail(resp)}",
                                status_code=resp.status_code)

    async def complete(self, request: ChatCompletionRequest, upstream_model: str) -> CompletionResult:
        try:
            resp = await self._client.post("/chat/completions", json=self._payload(request, upstream_model, False))
        except httpx.HTTPError as exc:
            raise ProviderError(self.name, f"network error: {exc!r}") from exc
        self._raise_for(resp)
        data = resp.json()
        choice = data["choices"][0]
        content = choice["message"].get("content") or ""
        usage = data.get("usage") or {}
        prompt_text = " ".join(m.content for m in request.messages)
        return CompletionResult(
            content=content,
            prompt_tokens=usage.get("prompt_tokens", estimate_tokens(prompt_text)),
            completion_tokens=usage.get("completion_tokens", estimate_tokens(content)),
            finish_reason=choice.get("finish_reason") or "stop",
        )

    async def stream(self, request: ChatCompletionRequest, upstream_model: str) -> AsyncIterator[str]:
        payload = self._payload(request, upstream_model, True)
        try:
            async with self._client.stream("POST", "/chat/completions", json=payload) as resp:
                if resp.status_code >= 400:
                    await resp.aread()
                self._raise_for(resp)
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    delta = json.loads(data)["choices"][0].get("delta", {}).get("content")
                    if delta:
                        yield delta
        except httpx.HTTPError as exc:
            raise ProviderError(self.name, f"network error: {exc!r}") from exc

    async def aclose(self) -> None:
        await self._client.aclose()
