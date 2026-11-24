"""LLMProviderRegistry and ModelHandle."""

from __future__ import annotations

import pytest

from aity_web.llm.catalog import (
    DEFAULT_MODELS_JSON,
    CatalogError,
    ModelCatalog,
    ModelNotActiveError,
    UnknownModelError,
    UnknownProviderError,
    load_catalog,
)
from aity_web.llm.providers.anthropic import AnthropicProvider
from aity_web.llm.providers.base import BaseProviderAdapter, ProviderResponse
from aity_web.llm.providers.gemini import GeminiProvider
from aity_web.llm.providers.ollama import DEFAULT_BASE_URL, OllamaProvider
from aity_web.llm.providers.openai import OpenAIProvider
from aity_web.llm.registry import (
    OLLAMA_BASE_URL_ENV,
    OLLAMA_MODEL_ENV,
    LLMProviderRegistry,
    ModelHandle,
    build_default_registry,
)
from aity_web.llm.types import user_message


class RecordingAdapter(BaseProviderAdapter):
    name = "openai"
    requires_api_key = True

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def complete(self, **kwargs) -> ProviderResponse:
        self.calls.append(kwargs)
        return ProviderResponse(text="{}", structured={})


@pytest.fixture(autouse=True)
def _no_operator_ollama_model(monkeypatch):
    """Tests must not depend on the shell's $AITY_OLLAMA_MODEL."""
    monkeypatch.delenv(OLLAMA_MODEL_ENV, raising=False)


def _catalog(options=None) -> ModelCatalog:
    return ModelCatalog.model_validate(
        {
            "catalog_version": 1,
            "providers": {
                "openai": {
                    "display_name": "OpenAI",
                    "models": [
                        {"id": "m1", "display_name": "M1", "tier": "fast", "default": True,
                         "options": options or {}},
                        {"id": "old", "display_name": "Old", "tier": "fast", "selectable": False},
                    ],
                }
            },
        }
    )


# --- default registry ---


def test_default_registry_has_four_adapters_and_no_berts(monkeypatch):
    monkeypatch.delenv(OLLAMA_BASE_URL_ENV, raising=False)
    registry = build_default_registry(load_catalog(DEFAULT_MODELS_JSON))
    assert registry.names() == ("gemini", "openai", "anthropic", "ollama")
    assert registry.adapter_class("gemini") is GeminiProvider
    assert registry.adapter_class("openai") is OpenAIProvider
    assert registry.adapter_class("anthropic") is AnthropicProvider
    assert registry.adapter_class("ollama") is OllamaProvider
    with pytest.raises(UnknownProviderError):
        registry.adapter_class("berts")


def test_requires_api_key_only_for_cloud_providers():
    registry = build_default_registry(load_catalog(DEFAULT_MODELS_JSON))
    assert {n: registry.requires_api_key(n) for n in registry.names()} == {
        "gemini": True, "openai": True, "anthropic": True, "ollama": False,
    }


def test_create_binds_adapter_and_catalogue_entry():
    registry = build_default_registry(load_catalog(DEFAULT_MODELS_JSON))
    handle = registry.create("ollama", "qwen3.5:4b")
    assert isinstance(handle, ModelHandle)
    assert isinstance(handle.adapter, OllamaProvider)
    assert handle.provider == "ollama"
    assert handle.model == "qwen3.5:4b"
    assert handle.prompt_profile == "small"
    assert handle.requires_api_key is False


def test_adapters_are_shared_between_handles():
    registry = build_default_registry(load_catalog(DEFAULT_MODELS_JSON))
    a = registry.create("gemini", "gemini-3.5-flash-lite")
    b = registry.create("gemini", "gemini-3.6-flash")
    assert a.adapter is b.adapter


def test_ollama_base_url_argument_env_and_default(monkeypatch):
    catalog = load_catalog(DEFAULT_MODELS_JSON)
    monkeypatch.delenv(OLLAMA_BASE_URL_ENV, raising=False)
    assert build_default_registry(catalog).adapter("ollama").base_url == DEFAULT_BASE_URL
    monkeypatch.setenv(OLLAMA_BASE_URL_ENV, "http://env:11434")
    assert build_default_registry(catalog).adapter("ollama").base_url == "http://env:11434"
    adapter = build_default_registry(catalog, ollama_base_url="http://arg:1").adapter("ollama")
    assert adapter.base_url == "http://arg:1"


def test_create_rejects_unknown_provider_and_model():
    registry = build_default_registry(load_catalog(DEFAULT_MODELS_JSON))
    with pytest.raises(UnknownProviderError):
        registry.create("berts", "keybert")
    with pytest.raises(UnknownModelError):
        registry.create("openai", "gpt-2")
    with pytest.raises(UnknownModelError):          # model of another provider
        registry.create("openai", "claude-haiku-5-5")


def test_provider_registered_but_not_in_catalogue_is_not_offered():
    registry = LLMProviderRegistry(_catalog())
    registry.register(RecordingAdapter)
    registry.register(OllamaProvider)
    registry.check()
    assert registry.names() == ("openai",)
    with pytest.raises(UnknownProviderError):
        registry.create("ollama", "qwen3.5:4b")


def test_non_selectable_model_still_resolves():
    registry = LLMProviderRegistry(_catalog())
    registry.register(RecordingAdapter)
    assert registry.create("openai", "old").spec.selectable is False


# --- AITY Built-in: one operator-chosen model (ADR-009) ---




def test_builtin_defaults_to_catalogue_default():
    registry = build_default_registry(load_catalog(DEFAULT_MODELS_JSON))
    assert registry.pinned_model("ollama").id == "qwen3.5:4b"
    assert [m.id for m in registry.offered_models("ollama")] == ["qwen3.5:4b"]
    assert registry.create("ollama", "qwen3.5:4b").model == "qwen3.5:4b"
    with pytest.raises(ModelNotActiveError, match="viewed but not re-run"):
        registry.create("ollama", "gemma3:4b")


def test_builtin_model_from_env_then_argument(monkeypatch):
    catalog = load_catalog(DEFAULT_MODELS_JSON)
    monkeypatch.setenv(OLLAMA_MODEL_ENV, "gemma3:4b")
    registry = build_default_registry(catalog)
    assert registry.create("ollama", "gemma3:4b").model == "gemma3:4b"
    with pytest.raises(ModelNotActiveError):
        registry.create("ollama", "qwen3.5:4b")
    # the argument beats the environment
    assert build_default_registry(catalog, ollama_model="qwen3.5:4b").pinned_model("ollama").id == "qwen3.5:4b"


def test_inactive_builtin_model_still_resolves_for_display():
    registry = build_default_registry(load_catalog(DEFAULT_MODELS_JSON))
    assert registry.catalog.model("ollama", "gemma3:4b").display_name == "Gemma 3 4B"


def test_unknown_builtin_model_fails_at_startup(monkeypatch):
    monkeypatch.setenv(OLLAMA_MODEL_ENV, "llama3.1:8b")
    with pytest.raises(CatalogError, match="llama3.1:8b"):
        build_default_registry(load_catalog(DEFAULT_MODELS_JSON))


def test_builtin_model_set_but_ollama_not_offered(monkeypatch):
    monkeypatch.setenv(OLLAMA_MODEL_ENV, "qwen3.5:4b")
    with pytest.raises(CatalogError, match="no ollama models"):
        build_default_registry(_catalog())


def test_cloud_providers_are_not_pinned():
    registry = build_default_registry(load_catalog(DEFAULT_MODELS_JSON))
    assert registry.pinned_model("gemini") is None
    offered = registry.offered_models("gemini")
    assert len(offered) >= 1 and all(m.selectable for m in offered)


def test_pinning_is_off_unless_asked():
    registry = LLMProviderRegistry(_catalog())
    registry.register(RecordingAdapter)
    assert registry.pinned_model("openai") is None
    assert registry.create("openai", "old").model == "old"


# --- registration ---


def test_duplicate_registration_fails():
    registry = LLMProviderRegistry(_catalog())
    registry.register(RecordingAdapter)
    with pytest.raises(ValueError, match="already registered"):
        registry.register(OpenAIProvider)


def test_adapter_with_unknown_name_is_rejected():
    class Berts(RecordingAdapter):
        name = "berts"

    with pytest.raises(ValueError, match="must be one of"):
        LLMProviderRegistry(_catalog()).register(Berts)


def test_check_fails_when_catalogue_provider_has_no_adapter():
    registry = LLMProviderRegistry(_catalog())
    with pytest.raises(ValueError, match="no registered adapter"):
        registry.check()


def test_factory_runs_once():
    built: list[RecordingAdapter] = []

    def factory() -> RecordingAdapter:
        built.append(RecordingAdapter())
        return built[-1]

    registry = LLMProviderRegistry(_catalog())
    registry.register(RecordingAdapter, factory=factory)
    registry.create("openai", "m1")
    registry.create("openai", "m1")
    assert len(built) == 1


# --- ModelHandle.complete ---


@pytest.mark.asyncio
async def test_handle_fills_model_and_catalogue_options():
    registry = LLMProviderRegistry(_catalog(options={"temperature": 0.0, "max_output_tokens": 100}))
    registry.register(RecordingAdapter)
    handle = registry.create("openai", "m1")

    await handle.complete(
        api_key="sk-test",
        messages=[user_message("doc")],
        system_prompt="sys",
        response_schema={"type": "object"},
        max_output_tokens=500,          # call-time kwarg wins over the catalogue
    )

    (call,) = handle.adapter.calls
    assert call["model"] == "m1"
    assert call["api_key"] == "sk-test"
    assert call["system_prompt"] == "sys"
    assert call["response_schema"] == {"type": "object"}
    assert call["temperature"] == 0.0
    assert call["max_output_tokens"] == 500


@pytest.mark.asyncio
async def test_handle_does_not_mutate_catalogue_options():
    catalog = _catalog(options={"temperature": 0.0})
    registry = LLMProviderRegistry(catalog)
    registry.register(RecordingAdapter)
    handle = registry.create("openai", "m1")
    await handle.complete(api_key="k", messages=[], system_prompt="s", temperature=1.0)
    assert catalog.model("openai", "m1").options == {"temperature": 0.0}


@pytest.mark.asyncio
async def test_catalogue_options_are_accepted_by_the_ollama_adapter():
    """Every option in models.json must be a kwarg OllamaProvider accepts.
    Each entry is a possible operator choice, so pin each one in turn."""
    catalog = load_catalog(DEFAULT_MODELS_JSON)
    for spec in catalog.provider("ollama").models:
        registry = build_default_registry(catalog, ollama_model=spec.id)
        handle = registry.create("ollama", spec.id)
        request = handle.adapter._build_request(  # noqa: SLF001 - checks the real request
            spec.id, [user_message("doc")], "sys", {"type": "object"}, dict(spec.options)
        )
        assert request["options"]["num_ctx"] == spec.options["num_ctx"]
