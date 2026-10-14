"""Provider interface. Every upstream (OpenAI, Anthropic, mock...) implements this."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass

from ..schemas import ChatCompletionRequest


class ProviderError(Exception):
    """Raised when an upstream call fails in a way that should trigger fallback."""

    def __init__(self, provider: str, message: str, status_code: int | None = None):
        super().__init__(f"[{provider}] {message}")
        self.provider = provider
        self.status_code = status_code


@dataclass
class CompletionResult:
    content: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str = "stop"


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token) used when the upstream gives no usage."""
    return max(1, len(text) // 4)


class Provider(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    async def complete(self, request: ChatCompletionRequest, upstream_model: str) -> CompletionResult:
        ...

    @abstractmethod
    def stream(self, request: ChatCompletionRequest, upstream_model: str) -> AsyncIterator[str]:
        """Yield text deltas. Must raise ProviderError *before* the first yield on failure."""
        ...

    async def aclose(self) -> None:  # pragma: no cover - default no-op
        return None
