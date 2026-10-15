"""Provider adapters tested against fake upstream servers (httpx.MockTransport) - no network."""
import json

import httpx
import pytest

from gateway.providers import AnthropicProvider, OpenAICompatProvider, ProviderError
from gateway.schemas import ChatCompletionRequest

REQ = ChatCompletionRequest.model_validate({"messages": [
    {"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}]})


def client(handler, base="https://up.test"):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url=base)


async def test_openai_complete_parses_usage():
    def handler(request: httpx.Request):
        body = json.loads(request.content)
        assert body["model"] == "gpt-x" and body["messages"][0]["role"] == "system"
        return httpx.Response(200, json={"choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
                                         "usage": {"prompt_tokens": 11, "completion_tokens": 2}})
    p = OpenAICompatProvider("oa", "https://up.test", "k", client=client(handler))
    r = await p.complete(REQ, "gpt-x")
    assert (r.content, r.prompt_tokens, r.completion_tokens) == ("hello", 11, 2)


async def test_openai_stream():
    sse = "".join(f'data: {json.dumps({"choices": [{"delta": {"content": t}}]})}\n\n' for t in ["he", "llo"])
    sse += "data: [DONE]\n\n"
    p = OpenAICompatProvider("oa", "https://up.test", "k",
                             client=client(lambda r: httpx.Response(200, text=sse)))
    assert "".join([t async for t in p.stream(REQ, "m")]) == "hello"


@pytest.mark.parametrize("status", [429, 500, 503])
async def test_openai_errors_raise_provider_error(status):
    p = OpenAICompatProvider("oa", "https://up.test", "k", client=client(lambda r: httpx.Response(status)))
    with pytest.raises(ProviderError) as e:
        await p.complete(REQ, "m")
    assert e.value.status_code == status


async def test_upstream_error_message_is_surfaced():
    body = {"error": {"message": "The model `old-model` does not exist", "code": "model_not_found"}}
    p = OpenAICompatProvider("groq", "https://up.test", "k", client=client(lambda r: httpx.Response(404, json=body)))
    with pytest.raises(ProviderError, match="does not exist"):
        await p.complete(REQ, "old-model")
    with pytest.raises(ProviderError, match="does not exist"):
        [t async for t in p.stream(REQ, "old-model")]


async def test_network_error_raises_provider_error():
    def boom(request):
        raise httpx.ConnectError("down")
    p = OpenAICompatProvider("oa", "https://up.test", "k", client=client(boom))
    with pytest.raises(ProviderError):
        await p.complete(REQ, "m")


async def test_anthropic_translates_system_and_response():
    def handler(request: httpx.Request):
        body = json.loads(request.content)
        assert body["system"] == "be brief"
        assert body["messages"] == [{"role": "user", "content": "hi"}]
        return httpx.Response(200, json={"content": [{"type": "text", "text": "yo"}], "stop_reason": "end_turn",
                                         "usage": {"input_tokens": 5, "output_tokens": 1}})
    p = AnthropicProvider("an", "k", client=client(handler))
    r = await p.complete(REQ, "claude-x")
    assert (r.content, r.prompt_tokens, r.completion_tokens) == ("yo", 5, 1)


async def test_anthropic_stream():
    events = [{"type": "message_start"},
              {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "hi "}},
              {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "there"}},
              {"type": "message_stop"}]
    sse = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)
    p = AnthropicProvider("an", "k", client=client(lambda r: httpx.Response(200, text=sse)))
    assert "".join([t async for t in p.stream(REQ, "m")]) == "hi there"
