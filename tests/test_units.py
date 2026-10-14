"""Unit tests for router, cache, rate limiter, cost and config."""
import pytest

from gateway.cache import HashingEmbedder, SemanticCache, normalize
from gateway.config import GatewayConfig
from gateway.ratelimit import RateLimiter
from gateway.router import Router, complexity_score
from gateway.schemas import ChatCompletionRequest
from gateway.usage import request_cost

from .conftest import chat, make_config


def req(content: str, **kw) -> ChatCompletionRequest:
    return ChatCompletionRequest.model_validate(chat(content, **kw))


# ------------------------------------------------------------------ routing
def test_complexity_score_easy_vs_hard():
    easy = complexity_score(req("What is the capital of France?"))
    hard = complexity_score(req("Analyze the trade-offs, then design and implement an algorithm "
                                "step by step to optimize this:\n```\ndef f(x):\n  return x\n```"))
    assert easy < 0.45 <= hard


def _router(strategy="complexity"):
    cfg = make_config(routing={"strategy": strategy, "complexity_threshold": 0.45, "max_attempts": 3})
    return Router(cfg.models, cfg.routing, cfg.circuit_breaker)


def test_router_sends_easy_prompts_to_cheap_tier():
    d = _router().route(req("hi"))
    assert [m.name for m in d.candidates] == ["small-a", "small-b", "large"]
    assert d.reason == "complexity:small"


def test_router_sends_hard_prompts_to_large_tier():
    d = _router().route(req("Prove this theorem step by step and analyze the algorithm complexity"))
    assert d.candidates[0].name == "large"


def test_cost_strategy_orders_by_price():
    names = [m.name for m in _router("cost").route(req("x")).candidates]
    assert names == ["small-a", "small-b", "large"]


def test_latency_strategy_prefers_fastest_observed():
    r = _router("latency")
    for name, ms in [("small-a", 300), ("small-b", 20), ("large", 100)]:
        r.record(name, ms, ok=True)
    assert [m.name for m in r.route(req("x")).candidates] == ["small-b", "large", "small-a"]


def test_max_attempts_limits_candidates():
    cfg = make_config(routing={"strategy": "cost", "max_attempts": 2})
    r = Router(cfg.models, cfg.routing, cfg.circuit_breaker)
    assert len(r.route(req("x")).candidates) == 2


# -------------------------------------------------------------------- cache
def test_normalize_handles_case_punctuation_contractions():
    assert normalize("What's the CAPITAL of France?!") == normalize("what is the capital of france")


def test_embedder_similar_texts_closer_than_unrelated():
    e = HashingEmbedder()
    a = e.embed("how do I reverse a linked list in python")
    b = e.embed("how can I reverse a linked list using python")
    c = e.embed("best pizza places in tampa florida")
    assert a @ b > 0.8 > a @ c


def test_semantic_cache_hit_and_partition_by_model():
    cache = SemanticCache(threshold=0.8)
    cache.put(req("how do I reverse a linked list in python"), {"answer": 1})
    hit = cache.get(req("how can I reverse a linked list using python"))
    assert hit and hit.kind == "semantic" and hit.response == {"answer": 1}
    assert cache.get(req("how do I reverse a linked list in python", model="large")) is None
    assert cache.get(req("best pizza in tampa")) is None


def test_semantic_hit_blocked_when_numbers_differ():
    cache = SemanticCache(threshold=0.5)
    cache.put(req("what is 17 times 23 in python"), {"answer": 391})
    assert cache.get(req("what is 17 times 24 in python")) is None
    assert cache.get(req("what is 17 times 23 using python")).response == {"answer": 391}


def test_cache_ttl_and_lru_eviction():
    now = [0.0]
    cache = SemanticCache(ttl_seconds=10, max_entries=2, clock=lambda: now[0])
    cache.put(req("one"), {"n": 1})
    cache.put(req("two"), {"n": 2})
    cache.get(req("one"))                 # touch -> "two" becomes LRU
    cache.put(req("three"), {"n": 3})
    assert cache.get(req("two")) is None and cache.get(req("one"))
    now[0] = 11
    assert cache.get(req("one")) is None  # expired
    assert cache.stats()["hits"] == 2


# --------------------------------------------------------------- rate limit
def test_token_bucket_refills_over_time():
    now = [0.0]
    rl = RateLimiter(clock=lambda: now[0])
    assert all(rl.check("k", 60)[0] for _ in range(60))
    allowed, retry = rl.check("k", 60)
    assert not allowed and retry == pytest.approx(1.0)
    now[0] = 1.0
    assert rl.check("k", 60)[0]


# --------------------------------------------------------------- cost/config
def test_request_cost():
    m = make_config().models[2]  # $1 in / $2 out per 1K
    assert request_cost(m, 500, 250) == pytest.approx(1.0)


def test_config_rejects_unknown_provider():
    raw = make_config().model_dump()
    raw["models"][0]["provider"] = "ghost"
    with pytest.raises(ValueError):
        GatewayConfig.model_validate(raw)


def test_config_rejects_empty_api_key():
    raw = make_config().model_dump()
    raw["auth"]["keys"][0]["key"] = ""
    with pytest.raises(ValueError):
        GatewayConfig.model_validate(raw)


def test_env_substitution(tmp_path, monkeypatch):
    from gateway.config import load_config
    monkeypatch.setenv("MY_KEY", "sk-from-env-123")
    p = tmp_path / "c.yaml"
    p.write_text("auth:\n  keys:\n    - {key: '${MY_KEY}', name: a}\nproviders:\n  - {name: m, type: mock}\n"
                 "models:\n  - {name: x, provider: m}\n")
    assert load_config(p).auth.keys[0].key == "sk-from-env-123"
