"""Ollama adapter (AITY Built-in): no API key and no EcoLogits impacts."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, Literal

import httpx

from aity_web.llm.providers.base import BaseProviderAdapter, FinishReason, ProviderResponse
from aity_web.llm.types import Message, TokenUsage

DEFAULT_BASE_URL = "http://ollama:11434"   # internal Docker network only, never public

StructuredOutputMode = Literal["grammar", "prompt"]

_FINISH_MAP: dict[str, FinishReason] = {"stop": "stop", "length": "length"}  # load/unload -> "other"

#: Model families where thinking is forced off (empty content otherwise).
_NO_THINK_PREFIXES: tuple[str, ...] = ("qwen3",)

#: Sampling options copied into `options` when not None.
_OPTIONS: tuple[str, ...] = ("temperature", "seed", "top_p", "top_k", "stop")
#: Other kwargs this adapter accepts.
_ADAPTER_KWARGS: frozenset[str] = frozenset(
    {"max_output_tokens", "num_ctx", "keep_alive", "think", "structured_output_mode"}
)

_PROMPT_MODE_SUFFIX = (
    "\n\nRespond with a single JSON object, and nothing else, "
    "that conforms to this JSON Schema:\n{schema}"
)


class OllamaError(RuntimeError):
    """Ollama answered with an HTTP error; carries its {"error": ...} text."""

    def __init__(self, status_code: int, message: str) -> None:
        """Create the error from the HTTP status and Ollama's error text."""
        super().__init__(f"ollama HTTP {status_code}: {message}")
        self.status_code = status_code
        self.message = message


class OllamaProvider(BaseProviderAdapter):
    """Adapter for the shared Ollama server."""

    name = "ollama"
    requires_api_key = False

    #: CPU-only Pouta: an 8.5k-word document took ~456 s in the W1 benchmark.
    timeout: ClassVar[httpx.Timeout] = httpx.Timeout(600.0, connect=5.0)
    default_num_ctx: ClassVar[int] = 16384             # same as the benchmark
    default_max_output_tokens: ClassVar[int] = 8192        # -> options.num_predict
    default_structured_output_mode: ClassVar[StructuredOutputMode] = "grammar"

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        # Not user state: one shared server address. `transport` is for tests.
        """Create the adapter for one shared Ollama server; transport is for tests."""
        self.base_url = base_url.rstrip("/")
        self._transport = transport

    async def complete(
        self,
        *,
        api_key: str | None,                 # ignored: AITY Built-in needs no key
        model: str,
        messages: Sequence[Message],
        system_prompt: str,
        response_schema: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Call Ollama's chat endpoint and return a ProviderResponse."""
        request = self._build_request(model, messages, system_prompt, response_schema, kwargs)

        async with self._make_client() as client:
            resp = await client.post("/api/chat", json=request)
        if resp.is_error:
            raise OllamaError(resp.status_code, self._error_text(resp))

        return self._to_response(resp.json(), structured_call=response_schema is not None)

    # --- request ---

    def _make_client(self) -> httpx.AsyncClient:
        """Create the HTTP client for one call."""
        return httpx.AsyncClient(
            base_url=self.base_url, timeout=self.timeout, transport=self._transport
        )

    def _build_request(
        self,
        model: str,
        messages: Sequence[Message],
        system_prompt: str,
        response_schema: Mapping[str, Any] | None,
        kwargs: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Build the chat request, enforcing the schema by grammar or by prompt."""
        
        unknown = set(kwargs) - _ADAPTER_KWARGS - set(_OPTIONS)
        if unknown:
            raise TypeError(f"{self.name}: unsupported options {sorted(unknown)}")

        mode = kwargs.get("structured_output_mode") or self.default_structured_output_mode
        if mode not in ("grammar", "prompt"):
            raise ValueError(f"{self.name}: structured_output_mode must be 'grammar' or 'prompt'")

        system = system_prompt
        request: dict[str, Any] = {"model": model, "stream": False}

        if response_schema is not None:
            if mode == "grammar":            # token-level enforcement (decided 2026-11-11)
                request["format"] = dict(response_schema)
            else:                            # fallback: valid JSON only, schema in the prompt
                request["format"] = "json"
                system += _PROMPT_MODE_SUFFIX.format(schema=json.dumps(response_schema, indent=2))

        request["messages"] = [
            {"role": "system", "content": system},
            *({"role": m["role"], "content": m["content"]} for m in messages),
        ]

        think = self._think(model, kwargs.get("think"))
        if think is not None:
            request["think"] = think

        options: dict[str, Any] = {
            "num_ctx": kwargs.get("num_ctx") or self.default_num_ctx,
            "num_predict": kwargs.get("max_output_tokens") or self.default_max_output_tokens,
        }
        for opt in _OPTIONS:
            if kwargs.get(opt) is not None:
                options[opt] = kwargs[opt]
        request["options"] = options

        if kwargs.get("keep_alive") is not None:   # else the server's OLLAMA_KEEP_ALIVE applies
            request["keep_alive"] = kwargs["keep_alive"]

        return request

    def _think(self, model: str, requested: bool | None) -> bool | None:
        """Decide the think flag: forced off for Qwen3 models, otherwise as requested."""
        if model.lower().startswith(_NO_THINK_PREFIXES):
            if requested:
                raise ValueError(f"{self.name}: think=True is not allowed for {model!r}")
            return False
        return requested

    # --- response ---

    def _to_response(self, body: Mapping[str, Any], *, structured_call: bool) -> ProviderResponse:
        """Convert Ollama's reply to a ProviderResponse."""
        message = body.get("message") or {}
        text = self._extract_text(message.get("content"))   # `thinking` is never used

        return ProviderResponse(
            text=text,
            structured=self._parse_structured(text) if structured_call else None,
            usage=TokenUsage(
                prompt_tokens=body.get("prompt_eval_count"),
                output_tokens=body.get("eval_count"),
            ),
            impacts=None,
            finish_reason=_FINISH_MAP.get(body.get("done_reason") or "", "other"),
            raw=body,      # keeps total/load/prompt_eval/eval durations (ns) for debugging
        )

    @staticmethod
    def _error_text(resp: httpx.Response) -> str:
        """Extract the error text of a failed Ollama reply."""
        try:
            error = resp.json().get("error")
        except ValueError:
            error = None
        return error if isinstance(error, str) else resp.text[:500]
