"""Adapter contract and response type shared by every LLM provider."""

from __future__ import annotations

import json

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, get_args

from aity_web.llm.types import Message, TokenUsage

# --- finish_reason ---

FinishReason = Literal["stop", "length", "content_filter", "tool_use", "other"]
FINISH_REASONS: tuple[str, ...] = get_args(FinishReason)


# --- ProviderResponse ---


@dataclass(frozen=True, slots=True)
class ProviderResponse:
    """Result of one adapter call: text, structured output, usage, impacts and finish reason."""

    text: str
    structured: dict[str, Any] | None = None
    usage: TokenUsage = field(default_factory=TokenUsage)
    impacts: Any | None = None
    finish_reason: FinishReason = "stop"
    raw: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError(f"text must be str, got {type(self.text).__name__}")
        if self.structured is not None and not isinstance(self.structured, dict):
            raise TypeError(f"structured must be dict or None, got {type(self.structured).__name__}")
        if not isinstance(self.usage, TokenUsage):
            raise TypeError(f"usage must be TokenUsage, got {type(self.usage).__name__}")
        if self.finish_reason not in FINISH_REASONS:
            raise ValueError(
                f"finish_reason {self.finish_reason!r} not normalised; expected one of {FINISH_REASONS}"
            )

    @property
    def truncated(self) -> bool:
        """True when the model stopped at the output-token limit."""
        return self.finish_reason == "length"


# --- BaseProviderAdapter ---


class BaseProviderAdapter(ABC):
    """Base class of the provider adapters; stateless with respect to users."""

    #: Registry key; one of aity_web.llm.types.PROVIDER_NAMES for real adapters.
    name: ClassVar[str]
    #: True for the three BYOK cloud providers; False for Ollama (AITY Built-in).
    #: ApiKeyService reads this to decide whether to look up a credential.
    requires_api_key: ClassVar[bool] = True

    @abstractmethod
    async def complete(
        self,
        *,
        api_key: str | None,
        model: str,
        messages: Sequence[Message],
        system_prompt: str,
        response_schema: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Run one non-streaming completion and return a ProviderResponse."""

    # --- helpers for subclasses ---

    def _require_api_key(self, api_key: str | None) -> str:
        """Return the key, or fail before any network call if it is missing."""
        if self.requires_api_key and not api_key:
            raise ValueError(f"{self.name}: an API key is required for this provider")
        return api_key or ""

    @staticmethod
    def _extract_text(content: str | Sequence[Any] | None) -> str:
        """Join the text parts of string-or-parts content, skipping non-text and thought parts."""
        if content is None:
            return ""
        if isinstance(content, str):
            return content

        chunks: list[str] = []
        for part in content:
            if isinstance(part, str):
                chunks.append(part)
                continue
            get = part.get if isinstance(part, Mapping) else lambda k, p=part: getattr(p, k, None)
            if get("thought"):
                continue
            part_type = get("type")
            if part_type is not None and part_type != "text":
                continue
            text = get("text")
            if isinstance(text, str):
                chunks.append(text)
        return "".join(chunks)

    @staticmethod
    def _parse_structured(text: str) -> dict[str, Any] | None:
        """Parse text as a JSON object, or return None."""
        try:
            value = json.loads(text)
        except ValueError:
            return None
        return value if isinstance(value, dict) else None

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={getattr(self, 'name', '?')!r})"
