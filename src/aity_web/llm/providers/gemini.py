"""Gemini adapter: google-genai async client with JSON Schema structured output.

* Structured output uses `response_json_schema` (full JSON Schema), not the
  legacy OpenAPI-subset `response_schema`.
* SDK exceptions (google.genai.errors.APIError) propagate unchanged; the
  service layer maps them to user-facing errors.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

from google import genai
from google.genai import types

from aity_web.llm.providers.base import BaseProviderAdapter, FinishReason, ProviderResponse
from aity_web.llm.types import Message, TokenUsage

#: Top-level schema keys Gemini does not document; removed from the copy sent.
_DROP_SCHEMA_KEYS: frozenset[str] = frozenset({"$schema"})

_FINISH_MAP: dict[str, FinishReason] = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
    "SAFETY": "content_filter",
    "RECITATION": "content_filter",
    "BLOCKLIST": "content_filter",
    "PROHIBITED_CONTENT": "content_filter",
    "SPII": "content_filter",
    "MALFORMED_FUNCTION_CALL": "tool_use",
    "UNEXPECTED_TOOL_CALL": "tool_use",
}

_ROLE_MAP: dict[str, str] = {"user": "user", "assistant": "model"}

_PASSTHROUGH: dict[str, str] = {
    "temperature": "temperature",
    "top_p": "top_p",
    "top_k": "top_k",
    "seed": "seed",
    "stop": "stop_sequences",
}
_THINKING: tuple[str, ...] = ("thinking_budget", "thinking_level")


class GeminiProvider(BaseProviderAdapter):
    name = "gemini"
    requires_api_key = True

    timeout_s: ClassVar[float] = 120.0
    retry_attempts: ClassVar[int] = 3              # SDK transport attempts (incl. the first)
    #: Thinking tokens count against this limit on Gemini 2.5+ models.
    default_max_output_tokens: ClassVar[int | None] = 8192

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
        key = self._require_api_key(api_key)
        config = self._build_config(system_prompt, response_schema, kwargs)
        contents = self._contents(messages)

        async with self._make_client(key) as client:
            raw = await client.models.generate_content(
                model=model, contents=contents, config=config
            )

        return self._to_response(raw, structured_call=response_schema is not None)

    # --- request ---

    def _make_client(self, api_key: str) -> Any:
        """Async client (`genai.Client(...).aio`), closed by `async with`."""
        http_options = types.HttpOptions(
            timeout=int(self.timeout_s * 1000),     # milliseconds
            retry_options=types.HttpRetryOptions(attempts=self.retry_attempts),
        )
        return genai.Client(api_key=api_key, http_options=http_options).aio

    @staticmethod
    def _contents(messages: Sequence[Message]) -> list[types.Content]:
        return [
            types.Content(role=_ROLE_MAP[m["role"]], parts=[types.Part(text=m["content"])])
            for m in messages
        ]

    def _build_config(
        self,
        system_prompt: str,
        response_schema: Mapping[str, Any] | None,
        options: Mapping[str, Any],
    ) -> types.GenerateContentConfig:
        unknown = set(options) - {"max_output_tokens", *_PASSTHROUGH, *_THINKING}
        if unknown:
            raise TypeError(f"{self.name}: unsupported options {sorted(unknown)}")

        config: dict[str, Any] = {}
        if system_prompt:
            config["system_instruction"] = system_prompt

        if response_schema is not None:
            config["response_mime_type"] = "application/json"
            config["response_json_schema"] = {
                k: v for k, v in response_schema.items() if k not in _DROP_SCHEMA_KEYS
            }

        limit = options.get("max_output_tokens")
        if limit is None:
            limit = self.default_max_output_tokens
        if limit is not None:
            config["max_output_tokens"] = limit

        for ours, theirs in _PASSTHROUGH.items():
            if options.get(ours) is not None:
                config[theirs] = options[ours]

        thinking = {k: options[k] for k in _THINKING if options.get(k) is not None}
        if thinking:                                 # unset = the model's own default
            config["thinking_config"] = types.ThinkingConfig(**thinking)

        return types.GenerateContentConfig(**config)

    # --- response ---

    def _to_response(self, raw: Any, *, structured_call: bool) -> ProviderResponse:
        candidates = raw.candidates or []

        if not candidates:                           # prompt blocked before generation
            feedback = getattr(raw, "prompt_feedback", None)
            blocked = feedback is not None and getattr(feedback, "block_reason", None)
            text, finish = "", ("content_filter" if blocked else "other")
        else:
            candidate = candidates[0]
            content = candidate.content
            text = self._extract_text(content.parts if content else None)  # skips thought parts
            finish = _FINISH_MAP.get(self._enum_name(candidate.finish_reason), "other")

        return ProviderResponse(
            text=text,
            structured=self._parse_structured(text) if structured_call else None,
            usage=self._usage(raw.usage_metadata),
            impacts=getattr(raw, "impacts", None),   # set by EcoLogits; absent when not initialised
            finish_reason=finish,
            raw=raw,
        )

    @staticmethod
    def _enum_name(value: Any) -> str:
        return str(getattr(value, "value", value) or "")

    @staticmethod
    def _usage(u: Any) -> TokenUsage:
        if u is None:
            return TokenUsage()
        candidates = u.candidates_token_count
        thoughts = u.thoughts_token_count
        output = (
            (candidates or 0) + (thoughts or 0)
            if isinstance(candidates, int) or isinstance(thoughts, int)
            else None
        )
        return TokenUsage(
            prompt_tokens=u.prompt_token_count,
            output_tokens=output,                     # candidates + thoughts = billed output
            thought_tokens=thoughts,
            total_tokens=u.total_token_count,
        )
