"""W2 stub analysis service, end to end with fake adapters (no network)."""

from __future__ import annotations

import json

import pytest

import aity_web.services.analysis_service as svc
from aity_web.llm.catalog import (
    ModelNotActiveError,
    UnknownModelError,
    UnknownProviderError,
    load_catalog,
)
from aity_web.llm.prompts import PromptVariantNotAvailableError, UnknownPromptVariantError
from aity_web.llm.providers.base import BaseProviderAdapter, ProviderResponse
from aity_web.llm.registry import LLMProviderRegistry
from aity_web.llm.types import TokenUsage
from aity_web.models.llm_config import LLMConfig
from aity_web.services.analysis_service import MissingApiKeyError, analyze

GEMINI = LLMConfig(provider="gemini", model="gemini-3.5-flash-lite")
OLLAMA = LLMConfig(provider="ollama", model="qwen3.5:4b")
DOC = "Quantum software libraries give developers abstractions over circuits."

GOOD = {
    "summary": "  A short summary.  ",
    "keywords": ["Qiskit", "cirq", "qiskit ", "pennylane", "transpilation", "testing"],
    "topics": [
        {"label": "Quantum Libraries", "description": "The text compares libraries."},
        {"label": "Developer Experience", "description": "It lists pain points."},
    ],
}


def _resp(structured=GOOD, *, finish="stop", text=None) -> ProviderResponse:
    return ProviderResponse(
        text=text if text is not None else json.dumps(structured),
        structured=structured,
        usage=TokenUsage(prompt_tokens=100, output_tokens=50),
        finish_reason=finish,
    )


class FakeAdapter(BaseProviderAdapter):
    """Scripted adapter; records every complete() call."""

    def __init__(self, result) -> None:
        self.result = result
        self.calls: list[dict] = []

    async def complete(self, **kwargs) -> ProviderResponse:
        self.calls.append(kwargs)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class FakeGemini(FakeAdapter):
    name = "gemini"
    requires_api_key = True


class FakeOllama(FakeAdapter):
    name = "ollama"
    requires_api_key = False


class FakeShim:
    def __init__(self, events: list[str], track: dict | None = None) -> None:
        self.events, self.track = events, track
        self.stopped_with: tuple | None = None

    async def start(self):
        self.events.append("start")
        return "t-1"

    async def stop(self, track_id, model=None):
        self.events.append("stop")
        self.stopped_with = (track_id, model)
        return self.track


TRACK = {  # raw shim readings; the backend computes the energy
    "cpu_seconds": 30.0, "duration_s": 40.0, "vcpus": 8.0,
    "resident_bytes": 4_200_000_000, "model": "qwen3.5:4b",
}


@pytest.fixture(autouse=True)
def no_ecologits(monkeypatch):
    calls = []
    monkeypatch.setattr(svc, "init_ecologits", lambda: calls.append(1))
    return calls


def _registry(gemini_result=None, ollama_result=None):
    """Real shipped catalogue + fake adapters (catalogue offers all four)."""
    reg = LLMProviderRegistry(load_catalog())
    gem = FakeGemini(gemini_result if gemini_result is not None else _resp())
    oll = FakeOllama(ollama_result if ollama_result is not None else _resp())
    reg.register(FakeGemini, factory=lambda: gem)
    reg.register(FakeOllama, factory=lambda: oll)
    return reg, gem, oll


# --- happy path and what the adapter receives ---


@pytest.mark.asyncio
async def test_ok_outcome_is_normalised(no_ecologits):
    reg, gem, _ = _registry()
    out = await analyze(DOC, GEMINI, api_key="k", registry=reg)

    assert out.status == "ok" and out.ok and out.error is None and out.issues == ()
    assert out.payload.summary == "A short summary."
    assert out.payload.keywords == ["qiskit", "cirq", "pennylane", "transpilation", "testing"]
    assert (out.provider, out.model, out.prompt_variant) == ("gemini", "gemini-3.5-flash-lite", "default_analysis")
    assert out.finish_reason == "stop" and out.usage.total_tokens == 150
    assert out.sustainability.latency_seconds >= 0
    assert no_ecologits == [1]


@pytest.mark.asyncio
async def test_adapter_gets_rendered_prompt_schema_and_key():
    reg, gem, _ = _registry()
    out = await analyze(DOC, GEMINI, api_key="k", registry=reg)

    (call,) = gem.calls
    assert call["api_key"] == "k" and call["model"] == "gemini-3.5-flash-lite"
    assert call["response_schema"]["title"] == "LLMAnalysisPayload"
    assert "research assistant" in call["system_prompt"]          # standard profile
    assert DOC in call["messages"][0]["content"] and DOC not in call["system_prompt"]
    assert out.prompt_profile == "standard"


@pytest.mark.asyncio
async def test_ollama_uses_small_profile_catalogue_options_and_no_key():
    reg, _, oll = _registry()
    out = await analyze(DOC, OLLAMA, api_key="ignored", registry=reg, shim=FakeShim([]))

    (call,) = oll.calls
    assert call["api_key"] is None
    assert call["num_ctx"] == 16384 and call["structured_output_mode"] == "grammar"
    assert call["system_prompt"].startswith("Analyse the document. Return JSON only.")
    assert out.prompt_profile == "small"


# --- partial / failed ---


@pytest.mark.asyncio
async def test_semantic_shortfall_is_partial():
    reg, *_ = _registry(_resp({**GOOD, "keywords": ["a", "b"]}))
    out = await analyze(DOC, GEMINI, api_key="k", registry=reg)
    assert out.status == "partial" and out.ok and out.payload is not None
    assert out.issues == ("keywords: 2 (expected 5–10)",)


@pytest.mark.parametrize(
    "resp, error",
    [
        (_resp(finish="length"), "output hit the token limit"),
        (_resp(finish="content_filter"), "the provider blocked the request or the output"),
        (_resp(None, text="not json"), "output was not valid JSON"),
        (_resp({"summary": "s", "keywords": []}), "output does not match the analysis schema (1 errors)"),
        (_resp({**GOOD, "extra": 1}), "output does not match the analysis schema (1 errors)"),
    ],
)
@pytest.mark.asyncio
async def test_unusable_responses_fail(resp, error):
    reg, *_ = _registry(resp)
    out = await analyze(DOC, GEMINI, api_key="k", registry=reg)
    assert out.status == "failed" and not out.ok and out.payload is None
    assert out.error == error
    assert out.finish_reason == resp.finish_reason


@pytest.mark.asyncio
async def test_provider_exception_becomes_sanitised_failure():
    reg, *_ = _registry(RuntimeError("401: key sk-SECRET rejected"))
    out = await analyze(DOC, GEMINI, api_key="k", registry=reg)
    assert out.status == "failed"
    assert out.error == "provider call failed (RuntimeError)"
    assert "SECRET" not in out.error
    assert out.finish_reason is None and out.usage.is_empty
    assert out.sustainability.estimate_status == "unavailable"


# --- Ollama shim ordering ---


@pytest.mark.asyncio
async def test_ollama_shim_wraps_call_and_feeds_collector():
    events: list[str] = []
    reg, _, oll = _registry()

    async def complete(**kwargs):
        events.append("complete")
        return _resp()

    oll.complete = complete
    shim = FakeShim(events, TRACK)
    out = await analyze(DOC, OLLAMA, registry=reg, shim=shim)

    assert events == ["start", "complete", "stop"]
    assert shim.stopped_with == ("t-1", "qwen3.5:4b")
    assert out.sustainability.estimation_source == "aity_cpu_model"
    assert out.sustainability.energy_kwh == pytest.approx(
        (30.0 * 3.125 + 40.0 * 4.2 * 0.375) * 1.08 / 3.6e6)


@pytest.mark.asyncio
async def test_ollama_shim_stopped_even_when_call_fails():
    events: list[str] = []
    reg, *_ = _registry(ollama_result=TimeoutError("read timeout"))
    out = await analyze(DOC, OLLAMA, registry=reg, shim=FakeShim(events, TRACK))
    assert events == ["start", "stop"]
    assert out.status == "failed" and out.error == "provider call failed (TimeoutError)"


@pytest.mark.asyncio
async def test_ollama_without_shim_result_is_unavailable_not_failed():
    reg, *_ = _registry()
    out = await analyze(DOC, OLLAMA, registry=reg, shim=FakeShim([], None))
    assert out.status == "ok"
    assert out.sustainability.estimate_status == "unavailable"


@pytest.mark.asyncio
async def test_cloud_path_never_touches_the_shim():
    events: list[str] = []
    reg, *_ = _registry()
    await analyze(DOC, GEMINI, api_key="k", registry=reg, shim=FakeShim(events, TRACK))
    assert events == []


# --- config errors raise before any call ---


@pytest.mark.parametrize(
    "config, text, api_key, exc",
    [
        (GEMINI, DOC, None, MissingApiKeyError),
        (GEMINI, DOC, "", MissingApiKeyError),
        (GEMINI, "   ", "k", ValueError),
        (LLMConfig(provider="gemini", model="gemini-0"), DOC, "k", UnknownModelError),
        (LLMConfig(provider="openai", model="gpt-5.4-mini-2026-03-17"), DOC, "k", UnknownProviderError),
        (LLMConfig(provider="gemini", model="gemini-3.5-flash-lite", prompt_variant="nope"), DOC, "k",
         UnknownPromptVariantError),
        (LLMConfig(provider="gemini", model="gemini-3.5-flash-lite", prompt_variant="qa_v1"), DOC, "k",
         PromptVariantNotAvailableError),
    ],
)
@pytest.mark.asyncio
async def test_config_errors_raise_before_the_call(config, text, api_key, exc):
    reg, gem, oll = _registry()   # openai is not registered here -> UnknownProviderError
    with pytest.raises(exc):
        await analyze(text, config, api_key=api_key, registry=reg)
    assert gem.calls == [] and oll.calls == []


def test_default_registry_is_built_once(monkeypatch):
    built = []
    svc.default_registry.cache_clear()
    monkeypatch.setattr(svc, "build_default_registry", lambda: built.append(1) or object())
    try:
        assert svc.default_registry() is svc.default_registry()
        assert built == [1]
    finally:
        svc.default_registry.cache_clear()


@pytest.mark.asyncio
async def test_inactive_builtin_model_is_refused_before_the_shim(no_ecologits):
    """ADR-009: an old session's Built-in model can be viewed, not re-run."""
    reg = LLMProviderRegistry(load_catalog(), pinned={"ollama": "qwen3.5:4b"})
    oll = FakeOllama(_resp())
    reg.register(FakeOllama, factory=lambda: oll)
    events: list[str] = []
    with pytest.raises(ModelNotActiveError):
        await analyze(DOC, LLMConfig(provider="ollama", model="gemma3:4b"),
                      registry=reg, shim=FakeShim(events))
    assert oll.calls == [] and events == []
