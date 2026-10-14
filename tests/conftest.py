import httpx
import pytest

from gateway.app import create_app
from gateway.config import GatewayConfig
from gateway.providers import MockProvider
from gateway.service import GatewayService

ADMIN = {"Authorization": "Bearer sk-admin-test"}
USER = {"Authorization": "Bearer sk-user-test"}


def make_config(**overrides) -> GatewayConfig:
    raw = {
        "auth": {"enabled": True, "keys": [
            {"key": "sk-admin-test", "name": "admin", "rate_limit_rpm": 100000, "admin": True},
            {"key": "sk-user-test", "name": "user", "rate_limit_rpm": 5},
        ]},
        "routing": {"strategy": "complexity", "complexity_threshold": 0.45, "max_attempts": 3},
        "circuit_breaker": {"failure_threshold": 2, "cooldown_seconds": 60},
        "cache": {"enabled": True, "similarity_threshold": 0.9},
        "providers": [
            {"name": "p-small", "type": "mock", "latency_ms": 0},
            {"name": "p-small2", "type": "mock", "latency_ms": 0},
            {"name": "p-large", "type": "mock", "latency_ms": 0},
        ],
        "models": [
            {"name": "small-a", "provider": "p-small", "tier": "small", "input_cost_per_1k": 0.1,
             "output_cost_per_1k": 0.2},
            {"name": "small-b", "provider": "p-small2", "tier": "small", "input_cost_per_1k": 0.2,
             "output_cost_per_1k": 0.3},
            {"name": "large", "provider": "p-large", "tier": "large", "input_cost_per_1k": 1.0,
             "output_cost_per_1k": 2.0},
        ],
    }
    raw.update(overrides)
    return GatewayConfig.model_validate(raw)


def chat(content: str, **kw) -> dict:
    return {"messages": [{"role": "user", "content": content}], **kw}


@pytest.fixture
def config():
    return make_config()


@pytest.fixture
def service(config):
    providers = {p.name: MockProvider(p.name, latency_ms=0, seed=1) for p in config.providers}
    return GatewayService(config, providers=providers)


@pytest.fixture
async def client(config, service):
    app = create_app(config, service)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
        yield c
