"""
Types shared by the LLM layer and other modules of aity_web.

Lives in the llm package (not models) so adapters never import from
aity_web.models (ADR-002 §7.1); models.analysis_result imports TokenUsage
from here.
"""

from __future__ import annotations

from typing import Literal, get_args

from typing_extensions import TypedDict  # Pydantic needs this one on Python < 3.12

from pydantic import BaseModel, ConfigDict, Field, model_validator

ProviderName = Literal["gemini", "openai", "anthropic", "ollama"]
PROVIDER_NAMES: tuple[str, ...] = get_args(ProviderName)


# --- Message ---

Role = Literal["user", "assistant"]


class Message(TypedDict):
    """One chat turn passed to BaseProviderAdapter.complete(messages=...).

    Each adapter translates it to its SDK's shape.

    No "system" role: the system prompt is a separate argument of complete(),
    because Anthropic and Gemini take it outside the message list.
    """

    role: Role
    content: str


def user_message(content: str) -> Message:
    return {"role": "user", "content": content}


def assistant_message(content: str) -> Message:
    return {"role": "assistant", "content": content}


# --- TokenUsage ---

_Count = int | None


class TokenUsage(BaseModel):
    """Token counts of one LLM call, as reported by the provider.

    Every count is None when the provider does not report it; it is never
    guessed, apart from total_tokens (see below).

    * prompt_tokens  - input tokens, including the system prompt.
    * output_tokens  - ALL generated tokens, thinking included. This is the
      billed output count and what EcoLogits uses. OpenAI and Anthropic
      report it this way already; the Gemini adapter adds
      candidates_token_count + thoughts_token_count (same as the desktop
      build_token_usage()).
    * thought_tokens - the reasoning part of output_tokens; None when the
      provider does not split it out (e.g. Ollama with think=False).
    * total_tokens   - provider's total; filled with prompt + output when the
      provider gives no total (Anthropic, some Ollama responses).
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    prompt_tokens: _Count = Field(default=None, ge=0)
    output_tokens: _Count = Field(default=None, ge=0)
    thought_tokens: _Count = Field(default=None, ge=0)
    total_tokens: _Count = Field(default=None, ge=0)

    @model_validator(mode="before")
    @classmethod
    def _fill_total(cls, data: object) -> object:
        if (
            isinstance(data, dict)
            and data.get("total_tokens") is None
            and isinstance(data.get("prompt_tokens"), int)
            and isinstance(data.get("output_tokens"), int)
        ):
            data = {**data, "total_tokens": data["prompt_tokens"] + data["output_tokens"]}
        return data

    @property
    def is_empty(self) -> bool:
        """True when the provider reported no counts at all."""
        return all(
            v is None
            for v in (self.prompt_tokens, self.output_tokens, self.thought_tokens, self.total_tokens)
        )
