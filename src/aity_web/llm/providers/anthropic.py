"""Anthropic adapter using the Messages API with JSON outputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

from anthropic import AsyncAnthropic

from aity_web.llm.providers.base import BaseProviderAdapter, FinishReason, ProviderResponse
from aity_web.llm.types import Message, TokenUsage

#: Top-level schema keys Anthropic does not document; removed from the copy sent.
_DROP_SCHEMA_KEYS: frozenset[str] = frozenset({"$schema"})

_FINISH_MAP: dict[str, FinishReason] = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "model_context_window_exceeded": "length",
    "refusal": "content_filter",
    "tool_use": "tool_use",
}                                                   # pause_turn -> "other"

#: Options forwarded when not None (AITY name -> API name).
_PASSTHROUGH: dict[str, str] = {
    "temperature": "temperature",
    "top_p": "top_p",
    "top_k": "top_k",
    "stop": "stop_sequences",
    "thinking": "thinking",
}


class AnthropicProvider(BaseProviderAdapter):
    """Adapter for Anthropic Claude models."""

    name = "anthropic"
    requires_api_key = True

    timeout_s: ClassVar[float] = 120.0
    max_retries: ClassVar[int] = 2                  # SDK transport retries
    default_max_output_tokens: ClassVar[int] = 8192  # max_tokens is required by the API

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
        """Call the Messages API and return a ProviderResponse."""
        key = self._require_api_key(api_key)
        request = self._build_request(model, messages, system_prompt, response_schema, kwargs)

        async with self._make_client(key) as client:
            raw = await client.messages.create(**request)

        return self._to_response(raw, structured_call=response_schema is not None)

    # --- request ---

    def _make_client(self, api_key: str) -> AsyncAnthropic:
        """Create the async client for one call."""
        return AsyncAnthropic(api_key=api_key, timeout=self.timeout_s, max_retries=self.max_retries)

    def _build_request(
        self,
        model: str,
        messages: Sequence[Message],
        system_prompt: str,
        response_schema: Mapping[str, Any] | None,
        options: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Build the Messages API request, rejecting unsupported options."""
        unknown = set(options) - {"max_output_tokens", *_PASSTHROUGH}
        if unknown:
            raise TypeError(f"{self.name}: unsupported options {sorted(unknown)}")

        request: dict[str, Any] = {
            "model": model,
            "max_tokens": options.get("max_output_tokens") or self.default_max_output_tokens,
            "messages": [{"role": m["role"], "content": m["content"]} for m in messages],
        }
        if system_prompt:
            request["system"] = system_prompt

        if response_schema is not None:
            schema = {k: v for k, v in response_schema.items() if k not in _DROP_SCHEMA_KEYS}
            request["output_config"] = {"format": {"type": "json_schema", "schema": schema}}

        for ours, theirs in _PASSTHROUGH.items():   # newer models reject temperature + top_p
            if options.get(ours) is not None:       # together, so only send what was set
                request[theirs] = options[ours]

        return request

    # --- response ---

    def _to_response(self, raw: Any, *, structured_call: bool) -> ProviderResponse:
        """Convert the SDK response to a ProviderResponse."""
        finish = _FINISH_MAP.get(raw.stop_reason or "", "other")
        text = self._extract_text(raw.content)      # text blocks only; thinking blocks skipped

        # A refusal may not follow the schema, so it is never parsed.
        structured = (
            self._parse_structured(text)
            if structured_call and finish != "content_filter"
            else None
        )

        return ProviderResponse(
            text=text,
            structured=structured,
            usage=self._usage(raw.usage),
            impacts=getattr(raw, "impacts", None),  # set by EcoLogits; absent when not initialised
            finish_reason=finish,
            raw=raw,
        )

    @staticmethod
    def _usage(u: Any) -> TokenUsage:
        """Convert Anthropic usage counts to TokenUsage."""
        if u is None:
            return TokenUsage()
        # input_tokens excludes prompt-cache reads/writes; add them so prompt_tokens is all input.
        prompt = (
            u.input_tokens
            + (getattr(u, "cache_creation_input_tokens", None) or 0)
            + (getattr(u, "cache_read_input_tokens", None) or 0)
        )
        details = getattr(u, "output_tokens_details", None)
        return TokenUsage(
            prompt_tokens=prompt,
            output_tokens=u.output_tokens,          # already includes thinking tokens
            thought_tokens=getattr(details, "thinking_tokens", None) if details else None,
        )                                           # total filled by TokenUsage
