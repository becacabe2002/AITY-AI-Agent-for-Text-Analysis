"""Prompt library: rendering, overrides, placeholders and immutability."""

from __future__ import annotations

import pytest

from aity_web.llm.catalog import DEFAULT_MODELS_JSON, load_catalog
from aity_web.llm.prompts import (
    DEFAULT_PARAMS,
    PROMPT_VARIANTS,
    PromptTemplate,
    PromptVariant,
    PromptVariantNotAvailableError,
    UnknownPromptVariantError,
    available_variants,
    get_prompt_variant,
    render_prompt,
)
from aity_web.models import KEYWORDS_MAX, KEYWORDS_MIN, TOPICS_MAX, TOPICS_MIN
from aity_web.models.llm_config import DEFAULT_PROMPT_VARIANT

# Prompt variant names are immutable (decided 2026-11-19). If one of these
# fails, a variant's text was edited: revert it and ship the change under a
# new name (e.g. default_analysis_v2) with its own entry here.
PINNED_FINGERPRINTS = {
    "default_analysis": "846c3f6fef5e8a63c231d23e109edf172327f5e3f726ab0fb117e46095ff76a0",
    "extraction_v1": "ec12705bca58a706c38f4f5b7b404b66770705cb479a1089b1c1ce718e838f06",
    "qa_v1": "0af194e77eaa8193ca46793fe7fd557391d7421255960a07d20343832fdb0e10",
}


def test_variant_texts_have_not_changed():
    assert {n: v.fingerprint() for n, v in PROMPT_VARIANTS.items()} == PINNED_FINGERPRINTS


def test_llm_config_default_variant_exists_and_is_available():
    assert get_prompt_variant(DEFAULT_PROMPT_VARIANT).available


def test_available_variants_exclude_placeholders():
    assert [v.name for v in available_variants()] == ["default_analysis"]


def test_unknown_variant():
    with pytest.raises(UnknownPromptVariantError):
        get_prompt_variant("default_analysis_v9")


@pytest.mark.parametrize("name", ["extraction_v1", "qa_v1"])
def test_placeholders_cannot_render(name):
    with pytest.raises(PromptVariantNotAvailableError):
        render_prompt(name, "some text")


# --- rendering ---


def test_default_params_match_payload_limits():
    assert DEFAULT_PARAMS["keywords_min"] == KEYWORDS_MIN
    assert DEFAULT_PARAMS["keywords_max"] == KEYWORDS_MAX
    assert DEFAULT_PARAMS["topics_min"] == TOPICS_MIN
    assert DEFAULT_PARAMS["topics_max"] == TOPICS_MAX


@pytest.mark.parametrize("profile", ["standard", "small"])
def test_render_fills_every_parameter(profile):
    prompt = render_prompt("default_analysis", "A short paper.", profile=profile)
    assert prompt.variant == "default_analysis"
    assert prompt.profile == profile
    assert "$" not in prompt.system_prompt
    assert f"{KEYWORDS_MIN} to {KEYWORDS_MAX}" in prompt.system_prompt
    assert f"{TOPICS_MIN} to {TOPICS_MAX}" in prompt.system_prompt
    (message,) = prompt.messages
    assert message["role"] == "user"
    assert "<document>\nA short paper.\n</document>" in message["content"]
    assert "A short paper." not in prompt.system_prompt


def test_small_profile_is_shorter_than_standard():
    standard = render_prompt("default_analysis", "x", profile="standard").system_prompt
    small = render_prompt("default_analysis", "x", profile="small").system_prompt
    assert len(small) < len(standard)


def test_variant_without_override_falls_back_to_standard():
    variant = PromptVariant(
        name="plain_v1", description="d", template=PromptTemplate(system="s", user="$document")
    )
    prompt = variant.render("doc", profile="small")
    assert prompt.profile == "standard"
    assert prompt.system_prompt == "s"


def test_params_override_defaults():
    prompt = render_prompt("default_analysis", "x", max_summary_words=120, keywords_max=8)
    assert "at most 120 words" in prompt.system_prompt
    assert f"{KEYWORDS_MIN} to 8" in prompt.system_prompt


def test_document_is_inserted_verbatim():
    doc = "Costs $5, uses ${braces} and {json: 1}."
    (message,) = render_prompt("default_analysis", doc).messages
    assert doc in message["content"]


@pytest.mark.parametrize(
    "params, match",
    [
        ({"colour": 3}, "unknown prompt parameters"),
        ({"keywords_max": 0}, "positive int"),
        ({"keywords_max": "10"}, "positive int"),
        ({"keywords_max": True}, "positive int"),
        ({"keywords_min": 9, "keywords_max": 3}, "greater than"),
        ({"topics_min": 7}, "greater than"),
    ],
)
def test_bad_params_are_rejected(params, match):
    with pytest.raises(ValueError, match=match):
        render_prompt("default_analysis", "x", **params)


def test_empty_document_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        render_prompt("default_analysis", "   ")


# --- definition checks ---


@pytest.mark.parametrize(
    "system, user, match",
    [
        ("s $unknown", "$document", "unknown template parameters"),
        ("s $document", "$document", "user template"),
        ("s", "no document here", "must contain"),
        ("costs $", "$document", "invalid"),
    ],
)
def test_bad_templates_are_rejected(system, user, match):
    with pytest.raises(ValueError, match=match):
        PromptTemplate(system=system, user=user)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"name": "Default-Analysis"}, "invalid prompt variant name"),
        ({"name": "x_v1"}, "needs a template"),
        ({"name": "x_v1", "status": "placeholder",
          "template": PromptTemplate(system="s", user="$document")}, "has no text"),
        ({"name": "x_v1", "template": PromptTemplate(system="s", user="$document"),
          "overrides": {"tiny": PromptTemplate(system="s", user="$document")}}, "override"),
    ],
)
def test_bad_variants_are_rejected(kwargs, match):
    with pytest.raises(ValueError, match=match):
        PromptVariant(description="d", **kwargs)


def test_overrides_are_read_only():
    with pytest.raises(TypeError):
        get_prompt_variant("default_analysis").overrides["small"] = None  # type: ignore[index]


def test_every_catalogue_profile_renders():
    profiles = {
        spec.prompt_profile
        for provider in load_catalog(DEFAULT_MODELS_JSON).providers.values()
        for spec in provider.models
    }
    for profile in profiles:
        render_prompt("default_analysis", "x", profile=profile)
