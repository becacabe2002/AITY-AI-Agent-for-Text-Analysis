from types import SimpleNamespace
import pytest
from aity_web.llm import user_message
from aity_web.llm.providers.openai import OpenAIProvider


class _FakeClient:
    def __init__(self, response):
        self.response, self.kwargs = response, None
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
    async def _create(self, **kwargs):
        self.kwargs = kwargs
        return self.response
    async def __aenter__(self): return self
    async def __aexit__(self, *exc): return None


def _raw(content='{"summary": "s"}', finish="stop", refusal=None, reasoning=2):
    details = None if reasoning is None else SimpleNamespace(reasoning_tokens=reasoning)
    return SimpleNamespace(
        choices=[SimpleNamespace(finish_reason=finish,
                                 message=SimpleNamespace(content=content, refusal=refusal))],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15,
                              completion_tokens_details=details),
    )


@pytest.fixture
def run(monkeypatch):
    async def _run(raw, **call):
        fake = _FakeClient(raw)
        adapter = OpenAIProvider()
        monkeypatch.setattr(adapter, "_make_client", lambda key, base_url: fake)
        args = dict(api_key="sk-test", model="gpt-x", messages=[user_message("hi")],
                    system_prompt="sys") | call
        return await adapter.complete(**args), fake.kwargs
    return _run


@pytest.mark.asyncio
async def test_structured_request_and_parse(run):
    resp, req = await run(_raw(), response_schema={"type": "object"}, max_output_tokens=500)
    assert req["messages"][0] == {"role": "system", "content": "sys"}
    assert req["response_format"]["json_schema"]["strict"] is True
    assert req["max_completion_tokens"] == 500 and "temperature" not in req
    assert resp.structured == {"summary": "s"} and resp.usage.thought_tokens == 2

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
