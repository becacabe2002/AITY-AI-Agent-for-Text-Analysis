"""Text normalisation and the analysis cache key - ADR-004 §7.6.

    cleaned = clean_text(raw)                       # at upload and at analysis time
    sha     = text_sha256(cleaned)                  # stored as Document.text_sha256
    key     = content_hash(text_sha256=sha, provider=..., model=..., prompt_variant=...)

Every AnalysisResult row stores content_hash and CACHE_KEY_VERSION from W3 on;
the per-user lookup that serves cached rows arrives in W6 (ADR-008) and
matches on `cache_key_version = CACHE_KEY_VERSION`.

When to bump CACHE_KEY_VERSION (same commit as the change):
* what clean_text() returns, or the content_hash() formula, changes
* the shipped config/models.json changes in content (a model's options,
  prompt profile, ...): it decides what a provider is asked
* the provider-facing LLMAnalysisPayload schema changes (field names,
  descriptions, limits): the model sees it
* an adapter default or adapter rule changes (not pinned by a test)
* the operator changes model behaviour outside the repo (an AITY_MODELS_JSON
  override, or a provider changes an undated model id): bump by hand

The pinned test tests/aity_web/test_cache_key.py fails on the first three
until its pins are updated together with the bump. A prompt text change
needs no bump: a new text ships under a new prompt variant name, and the
name is part of the hash.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Final

CACHE_KEY_VERSION: Final[int] = 1

_TRAILING_WS = re.compile(r"[ \t\f\v]+$", re.MULTILINE)
_BLANK_RUNS = re.compile(r"\n{3,}")
_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")


def clean_text(raw: str) -> str:
    """Normalise document text before hashing and before it is sent to a model.

    * Unicode NFC
    * \\r\\n and \\r become \\n
    * trailing whitespace removed from every line
    * three or more newlines collapsed to two (at most one blank line)
    * leading and trailing whitespace stripped
    """
    text = unicodedata.normalize("NFC", raw)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _TRAILING_WS.sub("", text)
    text = _BLANK_RUNS.sub("\n\n", text)
    return text.strip()


def text_sha256(cleaned: str) -> str:
    """Lowercase hex SHA-256 of an already cleaned text (UTF-8)."""
    return hashlib.sha256(cleaned.encode("utf-8")).hexdigest()


def canonical_json(value: Any) -> bytes:
    """The one JSON encoding used for hashing: sorted keys, no spaces,
    non-ASCII kept as UTF-8. Key order and file formatting never change it."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def content_hash(*, text_sha256: str, provider: str, model: str, prompt_variant: str) -> str:
    """SHA-256 of canonical JSON {"model", "prompt_variant", "provider", "text_sha256"}.

    The cache key of one analysis. cache_key_version is stored next to it
    on the row and is not part of the digest.
    ValueError if text_sha256 is not a lowercase hex SHA-256 or a name is empty."""
    if not _SHA256_HEX.match(text_sha256):
        raise ValueError("text_sha256 must be a lowercase hex SHA-256")
    for label, value in (("provider", provider), ("model", model), ("prompt_variant", prompt_variant)):
        if not value or not value.strip():
            raise ValueError(f"{label} is empty")
    doc = {
        "text_sha256": text_sha256,
        "provider": provider,
        "model": model,
        "prompt_variant": prompt_variant,
    }
    return hashlib.sha256(canonical_json(doc)).hexdigest()


# --- fingerprints of the inputs the pinned test watches ---


def models_json_fingerprint(path: str | Path | None = None) -> str:
    """SHA-256 of the canonical JSON of a models.json (default: the shipped one).

    Hashes the parsed content, not the bytes, so a line-ending change from
    git on Windows or a reformat does not count as a change."""
    if path is None:
        from aity_web.llm.catalog import DEFAULT_MODELS_JSON

        path = DEFAULT_MODELS_JSON
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return hashlib.sha256(canonical_json(data)).hexdigest()


def payload_schema_fingerprint() -> str:
    """SHA-256 of the canonical JSON of the provider-facing payload schema."""
    from aity_web.models.analysis_payload import llm_analysis_payload_schema

    return hashlib.sha256(canonical_json(llm_analysis_payload_schema())).hexdigest()


__all__ = [
    "CACHE_KEY_VERSION",
    "canonical_json",
    "clean_text",
    "content_hash",
    "models_json_fingerprint",
    "payload_schema_fingerprint",
    "text_sha256",
]
