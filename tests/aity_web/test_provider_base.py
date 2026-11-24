import dataclasses
from types import SimpleNamespace

import pytest

from aity_web.llm import TokenUsage, user_message
from aity_web.llm.providers import FINISH_REASONS, BaseProviderAdapter, ProviderResponse


class _Fake(BaseProviderAdapter):
    name = "openai"

    def __init__(self, response: ProviderResponse):
        self.response = response
        self.calls: list[dict] = []

    async def complete(self, *, api_key, model, messages, system_prompt, response_schema=None, **kwargs):
        self._require_api_key(api_key)
        self.calls.append(dict(api_key=api_key, model=model, messages=messages,
                               system_prompt=system_prompt, response_schema=response_schema, **kwargs))
        return self.response


class _NoKey(_Fake):
    name = "ollama"
    requires_api_key = False


# --- ProviderResponse ---

def test_response_defaults():
    r = ProviderResponse(text="hi")
    assert r.structured is None
    assert r.usage.is_empty
    assert r.impacts is None
    assert r.finish_reason == "stop"
    assert not r.truncated


def test_response_is_frozen():
    r = ProviderResponse(text="hi")
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.text = "x"


@pytest.mark.parametrize("reason", FINISH_REASONS)
def test_response_accepts_normalised_finish_reasons(reason):
    assert ProviderResponse(text="", finish_reason=reason).finish_reason == reason


@pytest.mark.parametrize("reason", ["STOP", "max_tokens", "end_turn", ""])
def test_response_rejects_unnormalised_finish_reason(reason):
    with pytest.raises(ValueError):
        ProviderResponse(text="", finish_reason=reason)


def test_length_means_truncated():
    assert ProviderResponse(text="{", finish_reason="length").truncated


@pytest.mark.parametrize("kwargs", [
    {"text": None},
    {"text": "x", "structured": "{}"},
    {"text": "x", "usage": {"prompt_tokens": 1}},
])
def test_response_type_checks(kwargs):
    with pytest.raises(TypeError):
        ProviderResponse(**kwargs)


def test_raw_ignored_in_equality_and_repr():
    a = ProviderResponse(text="x", raw=object())
    b = ProviderResponse(text="x", raw=object())
    assert a == b
    assert "raw" not in repr(a)


# --- BaseProviderAdapter ---

def test_base_is_abstract():
    with pytest.raises(TypeError):
        BaseProviderAdapter()


def test_requires_api_key_defaults_true():
    assert BaseProviderAdapter.requires_api_key is True
    assert _NoKey.requires_api_key is False


@pytest.mark.asyncio
async def test_complete_is_keyword_only():
    fake = _Fake(ProviderResponse(text="ok"))
    with pytest.raises(TypeError):
        await fake.complete("key", "model", [], "sys")


@pytest.mark.asyncio
async def test_complete_returns_response_and_passes_kwargs():
    resp = ProviderResponse(text='{"a": 1}', structured={"a": 1},
                            usage=TokenUsage(prompt_tokens=3, output_tokens=2))
    fake = _Fake(resp)
    out = await fake.complete(api_key="k", model="m", messages=[user_message("doc")],
                              system_prompt="sys", response_schema={"type": "object"},
                              max_output_tokens=512)
    assert out is resp
    assert fake.calls[0]["max_output_tokens"] == 512


@pytest.mark.asyncio
async def test_missing_key_fails_before_call():
    fake = _Fake(ProviderResponse(text="ok"))
    with pytest.raises(ValueError, match="API key"):
        await fake.complete(api_key=None, model="m", messages=[], system_prompt="s")
    assert fake.calls == []


@pytest.mark.asyncio
async def test_no_key_provider_accepts_none():
    fake = _NoKey(ProviderResponse(text="ok"))
    assert (await fake.complete(api_key=None, model="m", messages=[], system_prompt="s")).text == "ok"


# --- _extract_text ---

extract = BaseProviderAdapter._extract_text


def test_extract_plain_and_none():
    assert extract("hello") == "hello"
    assert extract(None) == ""
    assert extract([]) == ""


def test_extract_dict_parts_skips_non_text():
    parts = [
        {"type": "text", "text": "a"},
        {"type": "tool_use", "input": {"x": 1}},
        {"type": "text", "text": "b"},
        {"text": "c"},
    ]
    assert extract(parts) == "abc"


def test_extract_sdk_objects_and_skips_thoughts():
    parts = [
        SimpleNamespace(type="text", text="x"),
        SimpleNamespace(text="hidden", thought=True),   # Gemini thought part
        SimpleNamespace(text="y", thought=False),
        SimpleNamespace(type="tool_use", input={}),
    ]
    assert extract(parts) == "xy"
