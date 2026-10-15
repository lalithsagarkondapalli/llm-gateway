"""Adapter for the Anthropic Messages API, translated to/from the OpenAI format."""
from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx

from ..schemas import ChatCompletionRequest
from .base import CompletionResult, Provider, ProviderError
from .openai_compat import _error_detail

API_VERSION = "2023-06-01"


class AnthropicProvider(Provider):
    def __init__(self, name: str, api_key: str | None, base_url: str = "https://api.anthropic.com",
                 timeout: float = 30.0, client: httpx.AsyncClient | None = None):
        super().__init__(name)
        headers = {"x-api-key": api_key or "", "anthropic-version": API_VERSION}
        self._client = client or httpx.AsyncClient(base_url=base_url.rstrip("/"), headers=headers, timeout=timeout)

    def _payload(self, request: ChatCompletionRequest, upstream_model: str, stream: bool) -> dict:
        system = "\n".join(m.content for m in request.messages if m.role == "system")
        messages = [
            {"role": "assistant" if m.role == "assistant" else "user", "content": m.content}
            for m in request.messages
            if m.role != "system"
        ]
        body = {
            "model": upstream_model,
            "messages": messages,
            "max_tokens": request.max_tokens or 1024,
            "temperature": request.temperature,
            "stream": stream,
        }
        if system:
            body["system"] = system
        return body

    async def complete(self, request: ChatCompletionRequest, upstream_model: str) -> CompletionResult:
        try:
            resp = await self._client.post("/v1/messages", json=self._payload(request, upstream_model, False))
        except httpx.HTTPError as exc:
            raise ProviderError(self.name, f"network error: {exc!r}") from exc
        if resp.status_code >= 400:
            raise ProviderError(self.name, f"HTTP {resp.status_code}: {_error_detail(resp)}",
                                status_code=resp.status_code)
        data = resp.json()
        text = "".join(block.get("text", "") for block in data.get("content", []) if block.get("type") == "text")
        usage = data.get("usage", {})
        stop = data.get("stop_reason")
        return CompletionResult(
            content=text,
            prompt_tokens=usage.get("input_tokens", 0),
            completion_tokens=usage.get("output_tokens", 0),
            finish_reason="length" if stop == "max_tokens" else "stop",
        )

    async def stream(self, request: ChatCompletionRequest, upstream_model: str) -> AsyncIterator[str]:
        payload = self._payload(request, upstream_model, True)
        try:
            async with self._client.stream("POST", "/v1/messages", json=payload) as resp:
                if resp.status_code >= 400:
                    await resp.aread()
                    raise ProviderError(self.name, f"HTTP {resp.status_code}: {_error_detail(resp)}",
                                        status_code=resp.status_code)
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    event = json.loads(line[5:].strip())
                    if event.get("type") == "content_block_delta":
                        text = event.get("delta", {}).get("text")
                        if text:
                            yield text
                    elif event.get("type") == "message_stop":
                        break
        except httpx.HTTPError as exc:
            raise ProviderError(self.name, f"network error: {exc!r}") from exc

    async def aclose(self) -> None:
        await self._client.aclose()
