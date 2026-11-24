import pytest
from pydantic import ValidationError

from aity_web.llm.types import PROVIDER_NAMES
from aity_web.models import (
    DEFAULT_PROMPT_VARIANT,
    LLMAnalysisPayload,
    LLMConfig,
    build_provider_schema,
    llm_analysis_payload_schema,
    normalize_payload,
    payload_issues,
)

VALID = {
    "summary": "A short summary.",
    "keywords": ["quantum software", "libraries"],
    "topics": [{"label": "Software Engineering", "description": "Why it is here."}],
}


# --- LLMAnalysisPayload -------------------------------------------------------

def test_payload_accepts_valid_dict():
    p = LLMAnalysisPayload.model_validate(VALID)
    assert p.topics[0].label == "Software Engineering"
    assert p.model_dump() == VALID


@pytest.mark.parametrize("missing", ["summary", "keywords", "topics"])
def test_payload_rejects_missing_field(missing):
    bad = {k: v for k, v in VALID.items() if k != missing}
    with pytest.raises(ValidationError):
        LLMAnalysisPayload.model_validate(bad)


@pytest.mark.parametrize(
    "bad",
    [
        {**VALID, "notes": "extra field"},                       # additionalProperties: false
        {**VALID, "keywords": [{"keyword": "x"}]},               # keywords must be flat strings
        {**VALID, "topics": ["Software Engineering"]},           # topics are objects now
        {**VALID, "topics": [{"label": "X"}]},                   # description required
        {**VALID, "summary": 42},                                # no coercion to str
    ],
)
def test_payload_rejects_wrong_shape(bad):
    with pytest.raises(ValidationError):
        LLMAnalysisPayload.model_validate(bad)


def test_payload_allows_semantically_weak_answer():
    # Empty summary / no keywords is the service layer's call ("partial"), not a parse failure.
    LLMAnalysisPayload.model_validate({"summary": "", "keywords": [], "topics": []})


def test_schema_structure():
    # The shape every provider is constrained against (research note, Layer 1).
    # Description wording is not pinned here: the model is its only home.
    s = build_provider_schema()
    assert s["type"] == "object"
    assert s["required"] == ["summary", "keywords", "topics"]
    props = s["properties"]
    assert props["summary"]["type"] == "string"
    assert props["keywords"]["type"] == "array"
    assert props["keywords"]["items"]["type"] == "string"
    assert props["topics"]["type"] == "array"
    topic = props["topics"]["items"]
    assert topic["type"] == "object"
    assert topic["required"] == ["label", "description"]
    assert {k: v["type"] for k, v in topic["properties"].items()} == {
        "label": "string",
        "description": "string",
    }


def test_schema_has_descriptions_everywhere():
    # Every property and every string array item carries guidance for the model.
    # (Object array items, like a topic, are described by their own properties.)
    def check(node, path):
        if node.get("type") == "object":
            for name, child in node["properties"].items():
                assert child.get("description"), f"no description at {path}.{name}"
                check(child, f"{path}.{name}")
        elif node.get("type") == "array":
            items = node["items"]
            if items.get("type") == "string":
                assert items.get("description"), f"no description at {path}[]"
            check(items, f"{path}[]")

    check(build_provider_schema(), "$")


BANNED_KEYS = {
    "$ref", "$defs", "oneOf", "allOf", "anyOf", "not", "pattern", "minItems", "maxItems",
    "minLength", "maxLength", "default", "format", "enum", "const", "title",
}


def test_schema_follows_shared_dialect():
    # Walks keys, not text, so a description may mention "format" or "default".
    # Also catches keys a Pydantic upgrade might start emitting.
    schema = build_provider_schema()

    def check(node, path):
        bad = BANNED_KEYS & set(node)
        assert not bad, f"{path}: {sorted(bad)}"
        if node.get("type") == "object":
            assert node["additionalProperties"] is False, path
            assert set(node["required"]) == set(node["properties"]), path
            for name, child in node["properties"].items():
                check(child, f"{path}.{name}")
        elif node.get("type") == "array":
            check(node["items"], f"{path}[]")

    top = {k: v for k, v in schema.items() if k not in ("$schema", "title")}
    check(top, "$")


def test_schema_copy_is_independent():
    a = llm_analysis_payload_schema()
    a["properties"].clear()
    assert llm_analysis_payload_schema()["properties"]


GOOD = {
    "summary": "  A short summary.  ",
    "keywords": ["Quantum", "quantum", " SDK ", "", "compilers", "testing", "qiskit"],
    "topics": [
        {"label": " Software Engineering ", "description": "Why. "},
        {"label": "Tooling", "description": "Why."},
    ],
}


def test_normalize_cleans_keywords_and_whitespace():
    p = normalize_payload(LLMAnalysisPayload.model_validate(GOOD))
    assert p.summary == "A short summary."
    assert p.keywords == ["quantum", "sdk", "compilers", "testing", "qiskit"]  # order kept
    assert p.topics[0].label == "Software Engineering"
    assert p.topics[0].description == "Why."


def test_normalize_is_idempotent():
    once = normalize_payload(LLMAnalysisPayload.model_validate(GOOD))
    assert normalize_payload(once) == once


def test_issues_empty_for_good_payload():
    assert payload_issues(normalize_payload(LLMAnalysisPayload.model_validate(GOOD))) == []


def test_issues_reports_every_shortfall():
    weak = LLMAnalysisPayload.model_validate(
        {
            "summary": "   ",
            "keywords": ["a", "a", "b"],
            "topics": [{"label": "", "description": "x"}],
        }
    )
    issues = payload_issues(normalize_payload(weak))
    assert issues == [
        "summary is empty",
        "keywords: 2 (expected 5–10)",
        "topics: 1 (expected 2–6)",
        "topics[0].label is empty",
    ]


def test_issues_reports_too_many():
    many = LLMAnalysisPayload.model_validate(
        {
            "summary": "s",
            "keywords": [f"k{i}" for i in range(11)],
            "topics": [{"label": f"t{i}", "description": "d"} for i in range(7)],
        }
    )
    assert payload_issues(many) == ["keywords: 11 (expected 5–10)", "topics: 7 (expected 2–6)"]


# --- LLMConfig ----------------------------------------------------------------

def test_config_defaults_prompt_variant():
    cfg = LLMConfig(provider="ollama", model="qwen3.5:4b")
    assert cfg.prompt_variant == DEFAULT_PROMPT_VARIANT
    assert cfg.needs_api_key is False


@pytest.mark.parametrize("provider", ["gemini", "openai", "anthropic"])
def test_cloud_providers_need_key(provider):
    assert LLMConfig(provider=provider, model="m").needs_api_key


def test_provider_names_match_registry_keys():
    assert set(PROVIDER_NAMES) == {"gemini", "openai", "anthropic", "ollama"}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"provider": "berts", "model": "keybert"},              # BERTs is never a provider
        {"provider": "Gemini", "model": "m"},                   # registry keys are lowercase
        {"provider": "openai", "model": "   "},                 # blank model
        {"provider": "openai", "model": "m", "temperature": 0}, # sampling is not session config
    ],
)
def test_config_rejects_invalid(kwargs):
    with pytest.raises(ValidationError):
        LLMConfig(**kwargs)


def test_config_is_frozen_and_hashable():
    cfg = LLMConfig(provider="openai", model=" gpt-5.4-mini ")
    assert cfg.model == "gpt-5.4-mini"
    with pytest.raises(ValidationError):
        cfg.model = "other"
    assert hash(cfg) == hash(LLMConfig(provider="openai", model="gpt-5.4-mini"))


def test_config_json_round_trip():
    cfg = LLMConfig(provider="anthropic", model="claude-haiku-5-5")
    assert LLMConfig.model_validate_json(cfg.model_dump_json()) == cfg
