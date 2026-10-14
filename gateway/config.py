"""Gateway configuration loaded from YAML with ${ENV_VAR} substitution."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

_ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)(?::-([^}]*))?\}")


class ApiKeyConfig(BaseModel):
    key: str = Field(min_length=8)  # rejects empty keys from an unset ${ENV_VAR}
    name: str
    rate_limit_rpm: int = 60
    admin: bool = False


class ProviderConfig(BaseModel):
    name: str
    type: Literal["mock", "openai", "anthropic"]
    base_url: str | None = None
    api_key: str | None = None
    timeout_seconds: float = 30.0
    # mock-only knobs (also changeable at runtime via the chaos endpoint)
    latency_ms: float = 50.0
    failure_rate: float = 0.0


class ModelConfig(BaseModel):
    name: str  # public name clients use
    provider: str
    upstream_model: str | None = None  # name sent to the provider (defaults to `name`)
    tier: Literal["small", "large"] = "small"
    input_cost_per_1k: float = 0.0
    output_cost_per_1k: float = 0.0

    @property
    def upstream(self) -> str:
        return self.upstream_model or self.name


class RoutingConfig(BaseModel):
    strategy: Literal["complexity", "cost", "latency"] = "complexity"
    complexity_threshold: float = 0.45
    max_attempts: int = 3


class CircuitBreakerConfig(BaseModel):
    failure_threshold: int = 3
    cooldown_seconds: float = 15.0


class CacheConfig(BaseModel):
    enabled: bool = True
    similarity_threshold: float = 0.92
    ttl_seconds: float = 3600
    max_entries: int = 10_000
    embedding_dim: int = 256
    max_temperature: float = 0.3  # don't cache creative (high-temperature) requests


class AuthConfig(BaseModel):
    enabled: bool = True
    keys: list[ApiKeyConfig] = Field(default_factory=list)


class GatewayConfig(BaseModel):
    auth: AuthConfig = Field(default_factory=AuthConfig)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    circuit_breaker: CircuitBreakerConfig = Field(default_factory=CircuitBreakerConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    providers: list[ProviderConfig]
    models: list[ModelConfig]

    def model_post_init(self, __context) -> None:
        names = {p.name for p in self.providers}
        for m in self.models:
            if m.provider not in names:
                raise ValueError(f"model '{m.name}' references unknown provider '{m.provider}'")


def _substitute_env(text: str) -> str:
    def repl(match: re.Match) -> str:
        var, default = match.group(1), match.group(2)
        return os.environ.get(var, default if default is not None else "")

    return _ENV_PATTERN.sub(repl, text)


def load_config(path: str | os.PathLike | None = None) -> GatewayConfig:
    path = Path(path or os.environ.get("GATEWAY_CONFIG", "config.yaml"))
    raw = yaml.safe_load(_substitute_env(path.read_text()))
    return GatewayConfig.model_validate(raw)
