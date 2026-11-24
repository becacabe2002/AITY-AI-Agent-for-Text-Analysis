from types import SimpleNamespace

import pytest
from google.genai import types

from aity_web.llm import assistant_message, user_message
from aity_web.llm.providers.gemini import GeminiProvider


class _FakeClient:
    def __init__(self, response):
        self.response, self.kwargs = response, None
        self.models = SimpleNamespace(generate_content=self._generate)
    async def _generate(self, **kwargs):
        self.kwargs = kwargs
        return self.response
    async def __aenter__(self): return self
    async def __aexit__(self, *exc): return None


def _usage(prompt=10, cand=5, thoughts=3, total=18):
    return SimpleNamespace(prompt_token_count=prompt, candidates_token_count=cand,
                           thoughts_token_count=thoughts, total_token_count=total)


def _raw(parts=None, finish=types.FinishReason.STOP, usage=None, candidates=True, block=None):
    if parts is None:
        parts = [SimpleNamespace(text="thinking...", thought=True),
                 SimpleNamespace(text='{"summary": "s"}', thought=None)]
    cands = [SimpleNamespace(content=SimpleNamespace(parts=parts), finish_reason=finish)]
    return SimpleNamespace(candidates=cands if candidates else None,
                           prompt_feedback=SimpleNamespace(block_reason=block),
                           usage_metadata=usage if usage is not None else _usage())


@pytest.fixture
def run(monkeypatch):
    async def _run(raw, **call):
        fake = _FakeClient(raw)
        adapter = GeminiProvider()
        monkeypatch.setattr(adapter, "_make_client", lambda key: fake)
        args = dict(api_key="g-test", model="gemini-x", messages=[user_message("hi")],
                    system_prompt="sys") | call
        return await adapter.complete(**args), fake.kwargs
    return _run


@pytest.mark.asyncio
async def test_structured_request_and_parse(run):
    schema = {"$schema": "https://json-schema.org/draft/2020-12/schema", "title": "P",
              "type": "object"}
    resp, req = await run(_raw(), response_schema=schema, max_output_tokens=500)
    cfg = req["config"]
    assert cfg.system_instruction == "sys"
    assert cfg.response_mime_type == "application/json"
    assert cfg.response_json_schema == {"title": "P", "type": "object"}   # $schema dropped
    assert cfg.response_schema is None                                   # not the legacy field
    assert cfg.max_output_tokens == 500 and cfg.temperature is None
    assert cfg.thinking_config is None                                   # model default
    assert resp.text == '{"summary": "s"}'                               # thought part skipped
    assert resp.structured == {"summary": "s"} and resp.finish_reason == "stop"


@pytest.mark.asyncio
async def test_schema_not_mutated(run):
    schema = {"$schema": "x", "type": "object"}
    await run(_raw(), response_schema=schema)
    assert "$schema" in schema


@pytest.mark.asyncio
async def test_roles_and_contents(run):
    _, req = await run(_raw(), messages=[user_message("a"), assistant_message("b"),
                                         user_message("c")])
    assert [c.role for c in req["contents"]] == ["user", "model", "user"]
    assert [c.parts[0].text for c in req["contents"]] == ["a", "b", "c"]
    assert req["model"] == "gemini-x"


@pytest.mark.asyncio
async def test_usage_adds_thoughts_to_output(run):
    resp, _ = await run(_raw())
    u = resp.usage
    assert (u.prompt_tokens, u.output_tokens, u.thought_tokens, u.total_tokens) == (10, 8, 3, 18)


@pytest.mark.asyncio
async def test_usage_without_thoughts(run):
    resp, _ = await run(_raw(usage=_usage(thoughts=None, total=None)))
    assert resp.usage.output_tokens == 5 and resp.usage.thought_tokens is None
    assert resp.usage.total_tokens == 15


@pytest.mark.asyncio
async def test_free_text_call_has_no_structured(run):
    resp, req = await run(_raw())
    assert resp.structured is None and req["config"].response_json_schema is None


@pytest.mark.asyncio
@pytest.mark.parametrize("finish,expected", [
    (types.FinishReason.MAX_TOKENS, "length"),
    (types.FinishReason.SAFETY, "content_filter"),
    (types.FinishReason.RECITATION, "content_filter"),
    (types.FinishReason.OTHER, "other"),
    (None, "other"),
])
async def test_finish_reason_map(run, finish, expected):
    resp, _ = await run(_raw(finish=finish))
    assert resp.finish_reason == expected


@pytest.mark.asyncio
async def test_blocked_prompt_has_no_candidates(run):
    resp, _ = await run(_raw(candidates=False, block="SAFETY"), response_schema={"type": "object"})
    assert resp.text == "" and resp.structured is None
    assert resp.finish_reason == "content_filter"


@pytest.mark.asyncio
async def test_options_and_thinking(run):
    _, req = await run(_raw(), temperature=0.0, seed=7, top_k=20, stop=["END"],
                       thinking_level="low")
    cfg = req["config"]
    assert cfg.temperature == 0.0 and cfg.seed == 7 and cfg.top_k == 20
    assert cfg.stop_sequences == ["END"]
    assert cfg.thinking_config.thinking_level == types.ThinkingLevel.LOW
    assert cfg.max_output_tokens == GeminiProvider.default_max_output_tokens


@pytest.mark.asyncio
async def test_thinking_budget(run):
    _, req = await run(_raw(), thinking_budget=0)
    assert req["config"].thinking_config.thinking_budget == 0


@pytest.mark.asyncio
async def test_empty_system_prompt_omitted(run):
    _, req = await run(_raw(), system_prompt="")
    assert req["config"].system_instruction is None


@pytest.mark.asyncio
async def test_unknown_option_rejected(run):
    with pytest.raises(TypeError, match="reasoning_effort"):
        await run(_raw(), reasoning_effort="low")


@pytest.mark.asyncio
async def test_missing_key_fails_before_call(run):
    with pytest.raises(ValueError, match="API key"):
        await run(_raw(), api_key=None)


def test_real_client_is_async_and_closable():
    client = GeminiProvider()._make_client("g-test")
    assert hasattr(client, "__aenter__") and hasattr(client.models, "generate_content")


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
