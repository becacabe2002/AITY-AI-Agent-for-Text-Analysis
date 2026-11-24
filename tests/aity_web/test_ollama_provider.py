import json

import httpx
import pytest

from aity_web.llm import user_message
from aity_web.llm.providers.ollama import OllamaError, OllamaProvider


def _body(content='{"summary": "s"}', done_reason="stop", **extra):
    return {"model": "m", "message": {"role": "assistant", "content": content},
            "done": True, "done_reason": done_reason,
            "prompt_eval_count": 10, "eval_count": 5, **extra}


@pytest.fixture
def run():
    async def _run(body=None, status=200, **call):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url)
            seen["json"] = json.loads(request.content)
            return httpx.Response(status, json=_body() if body is None else body)

        adapter = OllamaProvider("http://ollama:11434/", transport=httpx.MockTransport(handler))
        args = dict(api_key=None, model="qwen3.5:4b", messages=[user_message("hi")],
                    system_prompt="sys") | call
        return await adapter.complete(**args), seen
    return _run


@pytest.mark.asyncio
async def test_grammar_mode_request_and_parse(run):
    resp, seen = await run(response_schema={"type": "object"}, max_output_tokens=500, temperature=0)
    req = seen["json"]
    assert seen["url"] == "http://ollama:11434/api/chat"
    assert req["stream"] is False and req["think"] is False
    assert req["format"] == {"type": "object"}
    assert req["messages"][0] == {"role": "system", "content": "sys"}
    assert req["options"] == {"num_ctx": 16384, "num_predict": 500, "temperature": 0}
    assert "keep_alive" not in req
    assert resp.structured == {"summary": "s"} and resp.impacts is None
    assert resp.usage.prompt_tokens == 10 and resp.usage.total_tokens == 15
    assert resp.usage.thought_tokens is None and resp.finish_reason == "stop"


@pytest.mark.asyncio
async def test_prompt_mode_puts_schema_in_system_prompt(run):
    _, seen = await run(response_schema={"type": "object"}, structured_output_mode="prompt")
    req = seen["json"]
    assert req["format"] == "json"
    assert req["messages"][0]["content"].startswith("sys\n\n")
    assert '"type": "object"' in req["messages"][0]["content"]


@pytest.mark.asyncio
async def test_free_text_call_has_no_format(run):
    resp, seen = await run(_body("plain prose"))
    assert "format" not in seen["json"]
    assert resp.text == "plain prose" and resp.structured is None


@pytest.mark.asyncio
async def test_think_rules(run):
    with pytest.raises(ValueError):
        await run(think=True)                                   # qwen3: hard off
    _, seen = await run(model="gemma3:4b")
    assert "think" not in seen["json"]                          # not sent to others by default
    _, seen = await run(model="gemma3:4b", think=False)
    assert seen["json"]["think"] is False


@pytest.mark.asyncio
async def test_length_and_unknown_done_reasons(run):
    resp, _ = await run(_body('{"summ', done_reason="length"), response_schema={})
    assert resp.truncated and resp.structured is None
    resp, _ = await run(_body(done_reason="load"))
    assert resp.finish_reason == "other"


@pytest.mark.asyncio
async def test_overrides_and_keep_alive(run):
    _, seen = await run(num_ctx=8192, keep_alive="10m", seed=7)
    assert seen["json"]["options"]["num_ctx"] == 8192
    assert seen["json"]["options"]["seed"] == 7
    assert seen["json"]["keep_alive"] == "10m"


@pytest.mark.asyncio
async def test_rejects_unknown_options_before_network(run):
    with pytest.raises(TypeError):
        await run(reasoning_effort="low")
    with pytest.raises(ValueError):
        await run(structured_output_mode="xml")


@pytest.mark.asyncio
async def test_http_error_carries_ollama_message(run):
    with pytest.raises(OllamaError) as exc:
        await run({"error": "model 'nope' not found"}, status=404, model="nope")
    assert exc.value.status_code == 404 and "not found" in exc.value.message


def test_no_api_key_needed():
    assert OllamaProvider.requires_api_key is False
    assert OllamaProvider()._require_api_key(None) == ""
