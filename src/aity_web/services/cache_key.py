"""Text normalisation and (Group D) the analysis cache key - ADR-004 §7.6.

W3 Group C needs clean_text() at upload time (text_sha256, char_count).
Group D adds content_hash(), CACHE_KEY_VERSION and the pinned test
(tests/aity_web/test_cache_key.py) to this module.

clean_text() is part of the cache key: text_sha256 is the SHA-256 of its
output. Changing what it does changes every hash, so a change here needs a
CACHE_KEY_VERSION bump (Group D pins its output on a fixed input).
"""

from __future__ import annotations

import hashlib
import re
import unicodedata

_TRAILING_WS = re.compile(r"[ \t\f\v]+$", re.MULTILINE)
_BLANK_RUNS = re.compile(r"\n{3,}")


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


__all__ = ["clean_text", "text_sha256"]
