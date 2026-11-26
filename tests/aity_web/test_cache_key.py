"""Cache key: clean_text(), content_hash() and the CACHE_KEY_VERSION pins - ADR-004 §7.6.

PINNED TEST. When one of the pins below fails, the cache key of every stored
row no longer describes what a new run would do. Fix it in ONE commit:

  1. bump CACHE_KEY_VERSION in src/aity_web/services/cache_key.py
  2. add a new entry PINS[<new version>] with the new values (leave the old
     entries as history; never edit them)
  3. say in the commit message what changed

Old rows then stop serving as cache sources (the W6 lookup filters on the
current version) but stay valid results.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from aity_web.services.cache_key import (
    CACHE_KEY_VERSION,
    canonical_json,
    clean_text,
    content_hash,
    models_json_fingerprint,
    payload_schema_fingerprint,
    text_sha256,
)

# A fixed raw input: combining accent (NFC), CRLF and lone CR, trailing
# spaces/tabs, a run of blank lines, a no-break space (kept: only
# space/tab/\f/\v count as trailing whitespace), leading indent (kept).
FIXED_RAW = (
    "  Café — quantum  \t\r\n\r\n\r\n\r\nLine two  \r\n  indented\rlast   \n\n"
)
FIXED_PROVIDER, FIXED_MODEL, FIXED_VARIANT = "gemini", "gemini-3.5-flash-lite", "default_analysis"

PINS: dict[int, dict[str, str]] = {
    1: {
        "clean_text": "Café — quantum\n\nLine two \n  indented\nlast",
        "text_sha256": "f739703305d574be36f471c6e96caafafc4a1990c714bdeb6ea3382ff0aba8a1",
        "content_hash": "6dbbfed72312bf7e70664d87cbe07824a78c6449e6c9f80d1f1e99d072ea6874",
        "models_json": "b708880e0cf543cf72fb246025fde6e02fc8925d3f2feb8e4d4b40ef1e5376c2",
        "payload_schema": "626d2526e7f3d9adf0691682b03e2530b40f728ab2cbd9487951f0e25e41e294",
    },
}


@pytest.fixture
def pins() -> dict[str, str]:
    assert CACHE_KEY_VERSION in PINS, (
        f"CACHE_KEY_VERSION={CACHE_KEY_VERSION} has no PINS entry: add one with the new values"
    )
    return PINS[CACHE_KEY_VERSION]


# --- the pins ---


def test_pinned_clean_text(pins):
    assert clean_text(FIXED_RAW) == pins["clean_text"], "clean_text() changed: bump CACHE_KEY_VERSION"
    assert text_sha256(clean_text(FIXED_RAW)) == pins["text_sha256"]


def test_pinned_content_hash(pins):
    digest = content_hash(
        text_sha256=pins["text_sha256"], provider=FIXED_PROVIDER,
        model=FIXED_MODEL, prompt_variant=FIXED_VARIANT,
    )
    assert digest == pins["content_hash"], "content_hash() changed: bump CACHE_KEY_VERSION"


def test_pinned_shipped_models_json(pins):
    assert models_json_fingerprint() == pins["models_json"], (
        "config/models.json changed: bump CACHE_KEY_VERSION and add a PINS entry"
    )


def test_pinned_payload_schema(pins):
    assert payload_schema_fingerprint() == pins["payload_schema"], (
        "the provider-facing LLMAnalysisPayload schema changed: bump CACHE_KEY_VERSION"
    )


def test_pin_history_is_consistent():
    """Versions are 1..N, the current one is the newest, and each bump
    really changed something (no empty bumps)."""
    versions = sorted(PINS)
    assert versions == list(range(1, len(versions) + 1))
    assert CACHE_KEY_VERSION == versions[-1]
    for old, new in zip(versions, versions[1:]):
        assert PINS[old] != PINS[new], f"PINS[{new}] equals PINS[{old}]"


# --- the formula, independent of the implementation ---


def test_content_hash_is_sha256_of_this_exact_json():
    sha = "a" * 64
    expected_json = (
        '{"model":"m","prompt_variant":"default_analysis","provider":"openai",'
        '"text_sha256":"' + sha + '"}'
    )
    expected = hashlib.sha256(expected_json.encode("utf-8")).hexdigest()
    assert content_hash(text_sha256=sha, provider="openai", model="m",
                        prompt_variant="default_analysis") == expected


@pytest.mark.parametrize(
    "field, value",
    [("text_sha256", "c" * 64), ("provider", "anthropic"), ("model", "other"),
     ("prompt_variant", "default_analysis_v2")],
)
def test_every_input_changes_the_hash(field, value):
    base = dict(text_sha256="b" * 64, provider="gemini", model="m", prompt_variant="v")
    assert content_hash(**base) != content_hash(**{**base, field: value})


def test_non_ascii_model_ids_hash_as_utf8():
    sha = "d" * 64
    expected = hashlib.sha256(
        ('{"model":"mä","prompt_variant":"v","provider":"ollama","text_sha256":"'
         + sha + '"}').encode("utf-8")
    ).hexdigest()
    assert content_hash(text_sha256=sha, provider="ollama", model="mä",
                        prompt_variant="v") == expected


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(text_sha256="A" * 64),        # uppercase hex
        dict(text_sha256="a" * 63),        # too short
        dict(text_sha256="not-a-hash"),
        dict(provider=""),
        dict(model="   "),
        dict(prompt_variant=""),
    ],
)
def test_bad_inputs_are_refused(kwargs):
    base = dict(text_sha256="a" * 64, provider="gemini", model="m", prompt_variant="v")
    with pytest.raises(ValueError):
        content_hash(**{**base, **kwargs})


def test_canonical_json_ignores_key_order_and_formatting(tmp_path):
    a = {"b": [1, {"y": 2, "x": "é"}], "a": None}
    b = json.loads(json.dumps(a, indent=4))
    assert canonical_json(a) == canonical_json(b) == '{"a":null,"b":[1,{"x":"é","y":2}]}'.encode()

    crlf = tmp_path / "crlf.json"
    lf = tmp_path / "lf.json"
    crlf.write_bytes(b'{\r\n  "x": 1,\r\n  "y": [2]\r\n}\r\n')
    lf.write_bytes(b'{"y":[2],"x":1}')
    assert models_json_fingerprint(crlf) == models_json_fingerprint(lf)


def test_canonical_json_refuses_nan():
    with pytest.raises(ValueError):
        canonical_json({"x": float("nan")})


# --- clean_text rules one by one ---


@pytest.mark.parametrize(
    "raw, cleaned",
    [
        ("é", "é"),                    # NFC
        ("a\r\nb\rc", "a\nb\nc"),                 # line endings
        ("a  \t\nb", "a\nb"),                     # trailing whitespace per line
        ("a\n\n\n\n\nb", "a\n\nb"),               # at most one blank line
        ("a\n\nb", "a\n\nb"),                     # one blank line kept
        ("\n\n  a  \n\n", "a"),                   # outer whitespace
        ("  a\n  b", "a\n  b"),                   # inner indentation kept
        ("   \n\t\n", ""),                        # nothing left
    ],
)
def test_clean_text_rules(raw, cleaned):
    assert clean_text(raw) == cleaned


def test_clean_text_is_idempotent():
    once = clean_text(FIXED_RAW)
    assert clean_text(once) == once
