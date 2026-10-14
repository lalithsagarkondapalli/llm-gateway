import json

from .conftest import ADMIN, USER, chat


async def test_completion_is_openai_compatible(client):
    r = await client.post("/v1/chat/completions", json=chat("hello there"), headers=ADMIN)
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["usage"]["total_tokens"] == body["usage"]["prompt_tokens"] + body["usage"]["completion_tokens"]
    assert r.headers["x-gateway-model"] == "small-a"


async def test_requires_auth(client):
    assert (await client.post("/v1/chat/completions", json=chat("hi"))).status_code == 401
    bad = {"Authorization": "Bearer nope-nope-nope"}
    assert (await client.post("/v1/chat/completions", json=chat("hi"), headers=bad)).status_code == 401


async def test_admin_endpoints_need_admin(client):
    assert (await client.get("/admin/usage", headers=USER)).status_code == 403
    assert (await client.get("/admin/usage", headers=ADMIN)).status_code == 200


async def test_unknown_model_404(client):
    r = await client.post("/v1/chat/completions", json=chat("hi", model="nope"), headers=ADMIN)
    assert r.status_code == 404


async def test_explicit_model(client):
    r = await client.post("/v1/chat/completions", json=chat("hi", model="large"), headers=ADMIN)
    assert r.headers["x-gateway-model"] == "large"


async def test_models_endpoint(client):
    ids = [m["id"] for m in (await client.get("/v1/models", headers=USER)).json()["data"]]
    assert ids == ["auto", "small-a", "small-b", "large"]


async def test_streaming_sse(client):
    async with client.stream("POST", "/v1/chat/completions", json=chat("stream me", stream=True),
                             headers=ADMIN) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        lines = [line async for line in r.aiter_lines() if line.startswith("data:")]
    assert lines[-1] == "data: [DONE]"
    chunks = [json.loads(line[5:]) for line in lines[:-1]]
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    assert "stream me" in text
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


async def test_streaming_result_is_cached(client):
    async with client.stream("POST", "/v1/chat/completions", json=chat("cache my stream", stream=True),
                             headers=ADMIN) as r:
        [_ async for _ in r.aiter_lines()]
    r2 = await client.post("/v1/chat/completions", json=chat("cache my stream"), headers=ADMIN)
    assert r2.headers["x-cache"] == "HIT-exact"


async def test_rate_limit_returns_429_with_retry_after(client):
    codes = [(await client.post("/v1/chat/completions", json=chat(f"q{i}"), headers=USER)).status_code
             for i in range(7)]
    assert codes[:5] == [200] * 5
    assert codes[5] == 429
    r = await client.post("/v1/chat/completions", json=chat("again"), headers=USER)
    assert int(r.headers["retry-after"]) >= 1


async def test_usage_and_metrics(client):
    await client.post("/v1/chat/completions", json=chat("count me"), headers=ADMIN)
    await client.post("/v1/chat/completions", json=chat("count me"), headers=ADMIN)
    usage = (await client.get("/admin/usage", headers=ADMIN)).json()
    assert usage["usage"]["admin"]["requests"] == 2
    assert usage["usage"]["admin"]["cache_hits"] == 1
    assert usage["usage"]["admin"]["saved_by_cache_usd"] > 0
    metrics = (await client.get("/metrics")).text
    assert 'gateway_requests_total{cache="hit"' in metrics
    assert "gateway_cost_usd_total" in metrics


async def test_cache_bypass_header_and_high_temperature(client):
    await client.post("/v1/chat/completions", json=chat("same"), headers=ADMIN)
    r = await client.post("/v1/chat/completions", json=chat("same"), headers={**ADMIN, "X-Cache-Bypass": "1"})
    assert r.headers["x-cache"] == "BYPASS"
    r = await client.post("/v1/chat/completions", json=chat("same", temperature=0.9), headers=ADMIN)
    assert r.headers["x-cache"] == "BYPASS"
