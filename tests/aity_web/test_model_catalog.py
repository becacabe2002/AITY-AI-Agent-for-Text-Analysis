"""Model catalogue (config/models.json) and its loader."""

from __future__ import annotations

import json

import pytest

from aity_web.llm.catalog import (
    DEFAULT_MODELS_JSON,
    MODELS_JSON_ENV,
    CatalogError,
    ModelCatalog,
    UnknownModelError,
    UnknownProviderError,
    load_catalog,
)
from aity_web.llm.types import PROVIDER_NAMES


def _entry(model_id: str, **extra):
    return {"id": model_id, "display_name": model_id, "tier": "fast", **extra}


def _catalog(models, provider="openai"):
    return {"catalog_version": 1, "providers": {provider: {"display_name": "X", "models": models}}}


def _write(tmp_path, data) -> str:
    path = tmp_path / "models.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


# --- shipped file ---


def test_shipped_catalogue_offers_all_four_providers():
    catalog = load_catalog(DEFAULT_MODELS_JSON)
    assert set(catalog.providers) == set(PROVIDER_NAMES)


def test_shipped_defaults_are_the_smoke_tested_models():
    catalog = load_catalog(DEFAULT_MODELS_JSON)
    defaults = {name: p.default_model for name, p in catalog.providers.items()}
    assert {n: m.id for n, m in defaults.items()} == {
        "gemini": "gemini-3.5-flash-lite",
        "openai": "gpt-5.4-mini-2026-03-17",
        "anthropic": "claude-haiku-5-5",
        "ollama": "qwen3.5:4b",
    }
    assert all(m.verified is not None for m in defaults.values())


def test_shipped_builtin_models_use_small_prompts_and_benchmark_context():
    ollama = load_catalog(DEFAULT_MODELS_JSON).provider("ollama")
    assert ollama.display_name == "AITY Built-in"
    for spec in ollama.models:
        assert spec.prompt_profile == "small"
        assert spec.options["num_ctx"] == 16384
        assert spec.options["structured_output_mode"] == "grammar"


def test_shipped_cloud_models_use_standard_prompts_and_no_options():
    catalog = load_catalog(DEFAULT_MODELS_JSON)
    for name in ("gemini", "openai", "anthropic"):
        for spec in catalog.provider(name).models:
            assert spec.prompt_profile == "standard"
            assert spec.options == {}


# --- lookup ---


def test_lookup_errors():
    catalog = ModelCatalog.model_validate(_catalog([_entry("a", default=True)]))
    assert catalog.model("openai", "a").id == "a"
    with pytest.raises(UnknownModelError):
        catalog.model("openai", "missing")
    with pytest.raises(UnknownProviderError):
        catalog.model("gemini", "a")


def test_selectable_models_hide_retired_entries():
    catalog = ModelCatalog.model_validate(
        _catalog([_entry("new", default=True), _entry("old", selectable=False)])
    )
    provider = catalog.provider("openai")
    assert [m.id for m in provider.selectable_models] == ["new"]
    assert provider.get("old") is not None


# --- loading ---


def test_env_var_overrides_shipped_file(tmp_path, monkeypatch):
    monkeypatch.setenv(MODELS_JSON_ENV, _write(tmp_path, _catalog([_entry("only", default=True)])))
    catalog = load_catalog()
    assert list(catalog.providers) == ["openai"]


def test_argument_wins_over_env_var(tmp_path, monkeypatch):
    monkeypatch.setenv(MODELS_JSON_ENV, str(tmp_path / "does-not-exist.json"))
    assert load_catalog(DEFAULT_MODELS_JSON).providers


def test_missing_file_raises_catalog_error(tmp_path):
    with pytest.raises(CatalogError, match="cannot read"):
        load_catalog(tmp_path / "nope.json")


@pytest.mark.parametrize(
    "data, match",
    [
        (_catalog([_entry("a"), _entry("b")]), "exactly one default"),
        (_catalog([_entry("a", default=True), _entry("b", default=True)]), "exactly one default"),
        (_catalog([_entry("a", default=True), _entry("a")]), "duplicate model ids"),
        (_catalog([_entry("a", default=True, selectable=False)]), "must be selectable"),
        (_catalog([_entry("a", default=True)], provider="berts"), "providers"),
        (_catalog([_entry("a", default=True, colour="red")]), "colour"),
        (_catalog([_entry("a", default=True, tier="huge")]), "tier"),
        (_catalog([_entry("a", default=True, prompt_profile="tiny")]), "prompt_profile"),
        (_catalog([]), "models"),
        ({**_catalog([_entry("a", default=True)]), "catalog_version": 2}, "catalog_version"),
    ],
)
def test_invalid_catalogues_are_rejected(tmp_path, data, match):
    with pytest.raises(CatalogError, match=match):
        load_catalog(_write(tmp_path, data))
