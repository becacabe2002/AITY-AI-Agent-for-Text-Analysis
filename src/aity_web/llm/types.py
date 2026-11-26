"""Types shared by the LLM layer and the rest of aity_web."""

from __future__ import annotations

from typing import Literal, get_args

from typing_extensions import TypedDict  # Pydantic needs this one on Python < 3.12

from pydantic import BaseModel, ConfigDict, Field, model_validator

ProviderName = Literal["gemini", "openai", "anthropic", "ollama"]
PROVIDER_NAMES: tuple[str, ...] = get_args(ProviderName)


# --- Message ---

Role = Literal["user", "assistant"]


class Message(TypedDict):
    """One chat turn passed to an adapter; there is no system role."""

    role: Role
    content: str


def user_message(content: str) -> Message:
    """Build a user message."""
    return {"role": "user", "content": content}


def assistant_message(content: str) -> Message:
    """Build an assistant message."""
    return {"role": "assistant", "content": content}


# --- TokenUsage ---

_Count = int | None


class TokenUsage(BaseModel):
    """Token counts of one LLM call as reported by the provider; None when not reported."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    prompt_tokens: _Count = Field(default=None, ge=0)
    output_tokens: _Count = Field(default=None, ge=0)
    thought_tokens: _Count = Field(default=None, ge=0)
    total_tokens: _Count = Field(default=None, ge=0)

    @model_validator(mode="before")
    @classmethod
    def _fill_total(cls, data: object) -> object:
        """Fill total_tokens from the prompt and output counts when the provider gives none."""
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
