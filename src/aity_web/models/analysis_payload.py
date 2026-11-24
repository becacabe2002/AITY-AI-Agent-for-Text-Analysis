"""
LLMAnalysisPayload is what the LLM generates, which then wrapped in AnalysisResult.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, Annotated

from pydantic import BaseModel, ConfigDict, Field

KEYWORDS_MIN, KEYWORDS_MAX = 5, 10
TOPICS_MIN, TOPICS_MAX = 2, 6

_JSON_SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"

# no extra undefined fields, make instance immutable and turn off coercion
_STRICT = ConfigDict(extra="forbid", frozen=True, strict=True)

Keyword = Annotated[
    str,
    Field(description="A single keyword or short noun phrase, no punctuation, 1-4 words."),
]

class Topic(BaseModel):
    model_config = _STRICT

    label: str = Field(description="A 2-6 word topic name in Title Case.")
    description: str = Field(description="One sentence explaining why this topic is in the document.")

class LLMAnalysisPayload(BaseModel):
    model_config = _STRICT

    summary: str = Field(description="An under-200-words summary of the document in the same language as the source.")
    keywords: list[Keyword] = Field(
        description=(
            f"{KEYWORDS_MIN}–{KEYWORDS_MAX} key terms or short phrases, "
            "lowercase, deduplicated, ordered by salience."
        )
    )
    topics: list[Topic] = Field(
        description=(
            f"{TOPICS_MIN}–{TOPICS_MAX} higher-level topics the document covers."
            " Each topic is an object with a short label and a one-sentence explanation."
        )
    )

# --- Provider-facing schema ---

def _inline(node: Any, defs: dict[str, Any]) -> Any:
    """Replace $ref with the referenced definition and drop Pydantic's titles."""
    if isinstance(node, dict):
        if "$ref" in node:
            return _inline(defs[node["$ref"].rsplit("/", 1)[-1]], defs) # get from the top $def
        return {k: _inline(v, defs) for k, v in node.items() if k not in ("$defs", "title")}
    if isinstance(node, list):
        return [_inline(item, defs) for item in node]
    return node

def build_provider_schema() -> dict[str, Any]:
    raw = LLMAnalysisPayload.model_json_schema()
    body = _inline(raw, raw.get("$defs", {}))
    return {"$schema": _JSON_SCHEMA_DIALECT, "title": "LLMAnalysisPayload", **body}

@lru_cache(maxsize=1)
def _schema_text() -> str:
    return json.dumps(build_provider_schema(), ensure_ascii=False)


def llm_analysis_payload_schema() -> dict[str, Any]:
    """Return a fresh copy of the provider-facing schema.

    A new dict each call, so an adapter that wraps or edits it (e.g. Anthropic's
    tool input_schema) cannot change the schema for other callers.
    """
    return json.loads(_schema_text())


# --- Normalize and check semantic ---

def normalize_payload(payload: LLMAnalysisPayload) -> LLMAnalysisPayload:
    """Clean a validated payload before it is persisted.

    * summary, topic labels and descriptions: surrounding whitespace stripped;
    * keywords: stripped, lowercased, empty ones dropped, duplicates removed
      (only keep first occurrence).

    Topics are not deduplicated or reordered: their labels are free text and
    merging them would be a guess.
    """
    keywords: list[str] = []
    seen: set[str] = set()
    for kw in payload.keywords:
        clean = kw.strip().lower()
        if clean and clean not in seen:
            seen.add(clean)
            keywords.append(clean)

    return LLMAnalysisPayload(
        summary=payload.summary.strip(),
        keywords=keywords,
        topics=[
            Topic(label=t.label.strip(), description=t.description.strip())
            for t in payload.topics
        ],
    )

def payload_issues(payload: LLMAnalysisPayload) -> list[str]:
    """
    Issues of a normalized payload.
    * No issues -> ok status
    * Have issues -> partial status
    The issues here never triggers a retry.
    """
    issues: list[str] = []
    if not payload.summary.strip():
        issues.append("summary is empty")

    n_kw = len(payload.keywords)
    if not KEYWORDS_MIN <= n_kw <= KEYWORDS_MAX:
        issues.append(f"keywords: {n_kw} (expected {KEYWORDS_MIN}–{KEYWORDS_MAX})")

    n_tp = len(payload.topics)
    if not TOPICS_MIN <= n_tp <= TOPICS_MAX:
        issues.append(f"topics: {n_tp} (expected {TOPICS_MIN}–{TOPICS_MAX})")

    for i, topic in enumerate(payload.topics):
        if not topic.label.strip():
            issues.append(f"topics[{i}].label is empty")
        if not topic.description.strip():
            issues.append(f"topics[{i}].description is empty")
    return issues