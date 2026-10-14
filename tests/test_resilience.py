"""Fallback + circuit breaker behaviour, driven through the chaos endpoint."""
from gateway.circuit_breaker import CircuitBreaker, State

from .conftest import ADMIN, chat


async def test_fallback_when_primary_fails(client):
    await client.post("/admin/providers/p-small/chaos", json={"failure_rate": 1.0}, headers=ADMIN)
    r = await client.post("/v1/chat/completions", json=chat("hi", model="small-a"), headers=ADMIN)
    assert r.status_code == 200
    assert r.headers["x-gateway-model"] == "small-b"      # same tier fallback first
    assert r.headers["x-attempts"] == "small-a,small-b"


async def test_circuit_opens_and_skips_bad_model(client, service):
    await client.post("/admin/providers/p-small/chaos", json={"failure_rate": 1.0}, headers=ADMIN)
    for i in range(2):  # failure_threshold=2
        await client.post("/v1/chat/completions", json=chat(f"x{i}"), headers={**ADMIN, "X-Cache-Bypass": "1"})
    assert service.router.breakers["small-a"].state is State.OPEN
    r = await client.post("/v1/chat/completions", json=chat("next"), headers=ADMIN)
    assert r.headers["x-attempts"] == "small-b"           # small-a no longer tried at all
    providers = (await client.get("/admin/providers", headers=ADMIN)).json()["models"]
    assert {m["model"]: m["circuit"] for m in providers}["small-a"] == "open"


async def test_all_providers_down_returns_503(client):
    for p in ("p-small", "p-small2", "p-large"):
        await client.post(f"/admin/providers/{p}/chaos", json={"failure_rate": 1.0}, headers=ADMIN)
    r = await client.post("/v1/chat/completions", json=chat("hi"), headers=ADMIN)
    assert r.status_code == 503
    assert r.json()["error"]["type"] == "upstream_unavailable"


async def test_stream_falls_back_before_first_token(client):
    await client.post("/admin/providers/p-small/chaos", json={"failure_rate": 1.0}, headers=ADMIN)
    async with client.stream("POST", "/v1/chat/completions", json=chat("s", model="small-a", stream=True),
                             headers=ADMIN) as r:
        assert r.status_code == 200
        assert r.headers["x-gateway-model"] == "small-b"


async def test_cache_serves_during_total_outage(client):
    await client.post("/v1/chat/completions", json=chat("popular question"), headers=ADMIN)
    for p in ("p-small", "p-small2", "p-large"):
        await client.post(f"/admin/providers/{p}/chaos", json={"failure_rate": 1.0}, headers=ADMIN)
    r = await client.post("/v1/chat/completions", json=chat("popular question"), headers=ADMIN)
    assert r.status_code == 200 and r.headers["x-cache"].startswith("HIT")


def test_circuit_breaker_state_machine():
    now = [0.0]
    cb = CircuitBreaker(failure_threshold=2, cooldown_seconds=10, clock=lambda: now[0])
    cb.record_failure()
    assert cb.state is State.CLOSED
    cb.record_failure()
    assert cb.state is State.OPEN and not cb.allow_request()
    now[0] = 10
    assert cb.state is State.HALF_OPEN
    assert cb.allow_request() is True       # one trial request
    assert cb.allow_request() is False      # second concurrent request blocked
    cb.record_failure()
    assert cb.state is State.OPEN           # failed trial re-opens
    now[0] = 20
    assert cb.allow_request()
    cb.record_success()
    assert cb.state is State.CLOSED


def test_success_resets_failure_count():
    cb = CircuitBreaker(failure_threshold=2)
    cb.record_failure()
    cb.record_success()
    cb.record_failure()
    assert cb.state is State.CLOSED
