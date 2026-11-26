"""Text normalisation and the content-hash cache key of an analysis."""

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
    """Normalise document text before hashing and before it is sent to a model."""
    text = unicodedata.normalize("NFC", raw)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _TRAILING_WS.sub("", text)
    text = _BLANK_RUNS.sub("\n\n", text)
    return text.strip()


def text_sha256(cleaned: str) -> str:
    """Lowercase hex SHA-256 of an already cleaned text (UTF-8)."""
    return hashlib.sha256(cleaned.encode("utf-8")).hexdigest()


def canonical_json(value: Any) -> bytes:
    """Encode a value as canonical JSON bytes for hashing."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def content_hash(*, text_sha256: str, provider: str, model: str, prompt_variant: str) -> str:
    """Return the cache key of one analysis as a SHA-256 hex digest."""
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
    """Hash the parsed content of models.json (default: the shipped file)."""
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
