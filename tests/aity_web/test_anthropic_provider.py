from types import SimpleNamespace

import pytest
from anthropic import AsyncAnthropic

from aity_web.llm import assistant_message, user_message
from aity_web.llm.providers.anthropic import AnthropicProvider


class _FakeClient:
    def __init__(self, response):
        self.response, self.kwargs = response, None
        self.messages = SimpleNamespace(create=self._create)
    async def _create(self, **kwargs):
        self.kwargs = kwargs
        return self.response
    async def __aenter__(self): return self
    async def __aexit__(self, *exc): return None


def _usage(inp=10, out=5, cache_w=None, cache_r=None, thinking=None):
    details = None if thinking is None else SimpleNamespace(thinking_tokens=thinking)
    return SimpleNamespace(input_tokens=inp, output_tokens=out,
                           cache_creation_input_tokens=cache_w, cache_read_input_tokens=cache_r,
                           output_tokens_details=details)


def _raw(text='{"summary": "s"}', stop="end_turn", usage=None, content=None):
    if content is None:
        content = [SimpleNamespace(type="text", text=text)]
    return SimpleNamespace(content=content, stop_reason=stop,
                           usage=usage if usage is not None else _usage())


@pytest.fixture
def run(monkeypatch):
    async def _run(raw, **call):
        fake = _FakeClient(raw)
        adapter = AnthropicProvider()
        monkeypatch.setattr(adapter, "_make_client", lambda key: fake)
        args = dict(api_key="sk-ant-test", model="claude-x", messages=[user_message("hi")],
                    system_prompt="sys") | call
        return await adapter.complete(**args), fake.kwargs
    return _run


@pytest.mark.asyncio
async def test_structured_request_and_parse(run):
    schema = {"$schema": "https://json-schema.org/draft/2020-12/schema", "title": "P",
              "type": "object", "additionalProperties": False}
    resp, req = await run(_raw(), response_schema=schema, max_output_tokens=500)
    assert req["system"] == "sys" and req["max_tokens"] == 500
    assert req["messages"] == [{"role": "user", "content": "hi"}]
    assert req["output_config"] == {"format": {"type": "json_schema", "schema": {
        "title": "P", "type": "object", "additionalProperties": False}}}   # $schema dropped
    assert "tools" not in req and "temperature" not in req and "thinking" not in req
    assert resp.text == '{"summary": "s"}' and resp.structured == {"summary": "s"}
    assert resp.finish_reason == "stop"


@pytest.mark.asyncio
async def test_schema_not_mutated(run):
    schema = {"$schema": "x", "type": "object"}
    await run(_raw(), response_schema=schema)
    assert "$schema" in schema


@pytest.mark.asyncio
async def test_default_max_tokens_and_roles(run):
    _, req = await run(_raw(), messages=[user_message("a"), assistant_message("b"),
                                         user_message("c")])
    assert req["max_tokens"] == AnthropicProvider.default_max_output_tokens
    assert [m["role"] for m in req["messages"]] == ["user", "assistant", "user"]


@pytest.mark.asyncio
async def test_empty_system_prompt_omitted(run):
    _, req = await run(_raw(), system_prompt="")
    assert "system" not in req


@pytest.mark.asyncio
async def test_thinking_blocks_skipped(run):
    content = [SimpleNamespace(type="thinking", thinking="hmm", text=None),
               SimpleNamespace(type="text", text='{"a": 1}')]
    resp, _ = await run(_raw(content=content), response_schema={"type": "object"})
    assert resp.text == '{"a": 1}' and resp.structured == {"a": 1}


@pytest.mark.asyncio
async def test_refusal_not_parsed(run):
    resp, _ = await run(_raw(text='{"summary": "x"}', stop="refusal"),
                        response_schema={"type": "object"})
    assert resp.finish_reason == "content_filter" and resp.structured is None


@pytest.mark.asyncio
@pytest.mark.parametrize("stop,expected", [
    ("max_tokens", "length"),
    ("model_context_window_exceeded", "length"),
    ("stop_sequence", "stop"),
    ("tool_use", "tool_use"),
    ("pause_turn", "other"),
    (None, "other"),
])
async def test_finish_reason_map(run, stop, expected):
    resp, _ = await run(_raw(stop=stop))
    assert resp.finish_reason == expected


@pytest.mark.asyncio
async def test_usage(run):
    resp, _ = await run(_raw(usage=_usage(inp=10, out=7, cache_w=2, cache_r=3, thinking=4)))
    u = resp.usage
    assert (u.prompt_tokens, u.output_tokens, u.thought_tokens, u.total_tokens) == (15, 7, 4, 22)


@pytest.mark.asyncio
async def test_usage_without_details(run):
    resp, _ = await run(_raw())
    assert resp.usage.thought_tokens is None and resp.usage.total_tokens == 15


@pytest.mark.asyncio
async def test_options_passthrough(run):
    thinking = {"type": "enabled", "budget_tokens": 2048}
    _, req = await run(_raw(), temperature=0.0, stop=["END"], top_k=5, thinking=thinking)
    assert req["temperature"] == 0.0 and req["stop_sequences"] == ["END"]
    assert req["top_k"] == 5 and req["thinking"] == thinking and "top_p" not in req


@pytest.mark.asyncio
async def test_unknown_option_rejected(run):
    with pytest.raises(TypeError, match="seed"):
        await run(_raw(), seed=1)


@pytest.mark.asyncio
async def test_missing_key_fails_before_call(run):
    with pytest.raises(ValueError, match="API key"):
        await run(_raw(), api_key="")


def test_real_client_settings():
    client = AnthropicProvider()._make_client("sk-ant-test")
    assert isinstance(client, AsyncAnthropic) and client.max_retries == 2


@pytest.mark.asyncio
async def test_impacts_copied_from_ecologits(run):
    raw = _raw()
    raw.impacts = object()                      # what EcoLogits' patch attaches
    resp, _ = await run(raw)
    assert resp.impacts is raw.impacts


@pytest.mark.asyncio
async def test_impacts_none_without_ecologits(run):
    resp, _ = await run(_raw())
    assert resp.impacts is None
