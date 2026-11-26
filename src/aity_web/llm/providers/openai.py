"""OpenAI adapter using chat completions with strict structured outputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

from openai import AsyncOpenAI

from aity_web.llm.providers.base import BaseProviderAdapter, FinishReason, ProviderResponse
from aity_web.llm.types import Message, TokenUsage

SCHEMA_NAME = "llm_analysis_payload"

_FINISH_MAP: dict[str, FinishReason] = {
    "stop": "stop",
    "length": "length",
    "content_filter": "content_filter",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
}

#: Sampling options forwarded unchanged when not None.
_PASSTHROUGH: tuple[str, ...] = ("temperature", "seed", "top_p", "stop", "reasoning_effort")

class OpenAIProvider(BaseProviderAdapter):
    """Adapter for OpenAI models."""

    name = "openai"
    requires_api_key = True

    timeout_s: ClassVar[float] = 120.0
    max_retries: ClassVar[int] = 2 # SDK transport retries
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
        """Call chat completions and return a ProviderResponse."""
        key = self._require_api_key(api_key)
        base_url = kwargs.pop("base_url", None) # None = api.openai.com
        request = self._build_request(model, messages, system_prompt, response_schema, kwargs)

        async with self._make_client(key, base_url) as client:
            raw = await client.chat.completions.create(**request)

        return self._to_response(raw, structured_call=response_schema is not None)

    # --- request ---

    def _make_client(self, api_key: str, base_url: str | None) -> AsyncOpenAI:
        """Create the async client for one call."""
        return AsyncOpenAI(
            api_key=api_key, base_url=base_url, timeout=self.timeout_s, max_retries=self.max_retries
        )

    def _build_request(
            self,
            model: str,
            messages: Sequence[Message],
            system_prompt: str,
            response_schema: Mapping[str, Any] | None,
            options: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Build the chat completions request, rejecting unsupported options."""
        unknown = set(options) - {"max_output_tokens", *_PASSTHROUGH}

        if unknown:
            raise TypeError(f"{self.name}: unsupported options {sorted(unknown)}")

        request: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                *({"role": m["role"], "content": m["content"]} for m in messages)
            ],
        }

        if response_schema is not None:
            request["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": SCHEMA_NAME, "strict": True, "schema": dict(response_schema)},
            }

        limit = options.get("max_output_tokens")
        if limit is None:
            limit = self.default_max_output_tokens
        if limit is not None:
            request["max_completion_tokens"] = limit

        for opt in _PASSTHROUGH:                    # reasoning models reject a non-default temperature,
            if options.get(opt) is not None:        # so only send what the caller actually set
                request[opt] = options[opt]
        
        return request

    # --- response ---

    def _to_response(self, raw: Any, *, structured_call: bool) -> ProviderResponse:
        """Convert the SDK response to a ProviderResponse."""
        choice = raw.choices[0]
        message = choice.message

        if message.refusal:                         # strict mode can refuse instead of answering
            text, structured = message.refusal, None
        else:
            text = message.content or ""
            structured = self._parse_structured(text) if structured_call else None

        return ProviderResponse(
            text=text,
            structured=structured,
            usage=self._usage(raw.usage),
            impacts=getattr(raw, "impacts", None),  # set by EcoLogits; absent when not initialised
            finish_reason=_FINISH_MAP.get(choice.finish_reason, "other"),
            raw=raw,
        )

    @staticmethod
    def _usage(u: Any) -> TokenUsage:
        """Convert OpenAI usage counts to TokenUsage."""
        if u is None:
            return TokenUsage()
        details = u.completion_tokens_details
        return TokenUsage(
            prompt_tokens=u.prompt_tokens,
            output_tokens=u.completion_tokens,      # already includes reasoning tokens
            thought_tokens=details.reasoning_tokens if details else None,
            total_tokens=u.total_tokens,
        )