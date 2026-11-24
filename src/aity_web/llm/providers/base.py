"""
Adapter contract for every LLM provider.

* BaseProviderAdapter - ABC with one abstract awaitable, complete().
* ProviderResponse    - what one complete() call returns. Short-lived, never
  persisted, unaware of users, sessions, prompts or caching. The service layer
  validates `structured` as LLMAnalysisPayload and builds the AnalysisResult.
"""

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
    """Result of one BaseProviderAdapter.complete() call.

    * text          - raw model output exactly as returned: the JSON string on
      structured calls (on every provider), plain prose
      otherwise. Logged server-side on a parse failure; never persisted.
    * structured    - the structured output as a dict; None when no
      response_schema was requested or the output could not be parsed.
      NOT validated here - the service layer does that.
    * usage         - token counts from the SDK's native usage object.
    * impacts       - EcoLogits `resp.impacts` (cloud adapters); always None on
      Ollama, whose estimate comes from the shim via the orchestrator (§7.5).
      Typed Any so this module never imports ecologits.
    * finish_reason - normalised stop reason; "length" means the output hit
      the token limit (service marks failed and retries once, ADR-008).
    * raw           - untouched SDK response, for debugging only.
    """

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
    """One adapter per provider SDK (gemini, openai, anthropic, ollama).

    Subclasses set the two class attributes and implement complete().
    Adapters are stateless with respect to users: the API key arrives per
    call (decrypted by ApiKeyService just before) and is never stored on
    the instance.
    """

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
        """Run one non-streaming completion and return a ProviderResponse.

        * api_key         - plaintext BYOK key; None for providers with
          requires_api_key=False. Use it for this call only.
        * messages        - user/assistant turns; no "system" role.
        * system_prompt   - passed in the provider's native slot.
        * response_schema - LLMAnalysisPayload JSON Schema; the adapter translates it to the native mechanism. None = free-text call, structured stays None.
        * kwargs          - provider sampling options. Common names adapters should have: `max_output_tokens`, `temperature`.

        Arguments are keyword-only so api_key / model can never be swapped.
        """

    # --- helpers for subclasses ---

    def _require_api_key(self, api_key: str | None) -> str:
        """Return the key, or fail before any network call if it is missing."""
        if self.requires_api_key and not api_key:
            raise ValueError(f"{self.name}: an API key is required for this provider")
        return api_key or ""

    @staticmethod
    def _extract_text(content: str | Sequence[Any] | None) -> str:
        """Unwrap string-or-parts content into one string.

        Accepts a plain string, or a list of parts that are dicts
        ({"type": "text", "text": ...}) or SDK objects with a `.text`
        attribute (Anthropic TextBlock, Gemini Part). Non-text parts
        (tool_use, images) and Gemini thought parts (thought=True) are
        skipped. Parts are concatenated without a separator.
        """
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
        try:
            value = json.loads(text)
        except ValueError:
            return None
        return value if isinstance(value, dict) else None

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={getattr(self, 'name', '?')!r})"
