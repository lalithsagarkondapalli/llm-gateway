from __future__ import annotations

from ..config import ProviderConfig
from .anthropic import AnthropicProvider
from .base import CompletionResult, Provider, ProviderError
from .mock import MockProvider
from .openai_compat import OpenAICompatProvider


def build_provider(cfg: ProviderConfig) -> Provider:
    if cfg.type == "mock":
        return MockProvider(cfg.name, latency_ms=cfg.latency_ms, failure_rate=cfg.failure_rate)
    if cfg.type == "openai":
        return OpenAICompatProvider(cfg.name, cfg.base_url or "https://api.openai.com/v1", cfg.api_key,
                                    timeout=cfg.timeout_seconds)
    if cfg.type == "anthropic":
        return AnthropicProvider(cfg.name, cfg.api_key, base_url=cfg.base_url or "https://api.anthropic.com",
                                 timeout=cfg.timeout_seconds)
    raise ValueError(f"unknown provider type: {cfg.type}")


__all__ = ["build_provider", "Provider", "ProviderError", "CompletionResult", "MockProvider",
           "OpenAICompatProvider", "AnthropicProvider"]
