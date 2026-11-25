"""
W2 STUB analysis service: one document, one provider call, no persistence.

    outcome = await analyze(text, LLMConfig(provider="gemini", model="gemini-3.5-flash-lite"),
                            api_key=key)

It wires the W2 pieces together in the order the real service will use:

    catalogue/registry -> prompt library -> [shim start] -> timed adapter.complete()
    -> [shim stop] -> parse + validate + normalise -> SustainabilityCollector

Replaced in W3 by the session-aware service, which adds what this stub leaves out:
* AnalysisResult rows (UUIDs, session/document/user, content_hash, timestamps)
  and the repository layer (ADR-004)
* ApiKeyService - here the caller passes the plaintext key
* the ADR-008 one-shot retry (length / broken JSON) and content-hash caching
* input truncation at the context window (ADR-007)
* the async job queue and per-provider concurrency ceiling (ADR-006)
* ProviderError(kind=...) mapping of SDK exceptions (ADR-002 §7.7)

Rules kept from the ADRs so W3 can lift them as-is:
* Config mistakes (unknown provider/model/variant, missing key, empty text)
  raise before any network call. Everything that goes wrong AFTER the call
  starts becomes status="failed" with a short sanitised `error`; the raw
  model output and exception text go to the server log only.
* latency_seconds times adapter.complete() only; the Ollama shim calls sit
  outside the timer (ADR-003 §7.3) and the shim is always stopped.
* Sustainability never fails an analysis (the collector never raises).
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Literal

import httpx
from pydantic import ValidationError

from aity_web.llm.providers.base import FinishReason, ProviderResponse
from aity_web.llm.prompts import RenderedPrompt, render_prompt
from aity_web.llm.registry import LLMProviderRegistry, ModelHandle, build_default_registry
from aity_web.llm.types import TokenUsage
from aity_web.models.analysis_payload import (
    LLMAnalysisPayload,
    llm_analysis_payload_schema,
    normalize_payload,
    payload_issues,
)
from aity_web.models.llm_config import LLMConfig
from aity_web.models.sustainability import SustainabilityMetrics
from aity_web.services.ollama_track_shim import OllamaTrackShim
from aity_web.services.sustainability import collect, init_ecologits, wall_clock

log = logging.getLogger(__name__)

AnalysisStatus = Literal["ok", "partial", "failed"]

_RAW_LOG_CHARS = 2000


class MissingApiKeyError(ValueError):
    """A BYOK provider was called without an API key."""


@dataclass(frozen=True, slots=True)
class AnalysisOutcome:
    """What the stub returns. NOT the persisted AnalysisResult (that is W3).

    * status         - "ok"; "partial" = valid payload with semantic issues
      (payload_issues); "failed" = no usable payload (see `error`)
    * payload        - normalised LLMAnalysisPayload; None when failed
    * issues         - semantic shortfalls behind a "partial" status
    * error          - short, sanitised reason; None unless failed
    * finish_reason  - from the provider; None if the call itself raised
    * prompt_profile - template profile actually rendered ("standard" | "small")
    """

    status: AnalysisStatus
    provider: str
    model: str
    prompt_variant: str
    prompt_profile: str
    sustainability: SustainabilityMetrics
    payload: LLMAnalysisPayload | None = None
    issues: tuple[str, ...] = ()
    error: str | None = None
    finish_reason: FinishReason | None = None
    usage: TokenUsage = field(default_factory=TokenUsage)

    @property
    def ok(self) -> bool:
        return self.status != "failed"


@lru_cache(maxsize=1)
def default_registry() -> LLMProviderRegistry:
    """Built on first use, so $AITY_OLLAMA_BASE_URL is read then (not at import)."""
    return build_default_registry()


async def analyze(
    text: str,
    config: LLMConfig,
    api_key: str | None = None,
    *,
    registry: LLMProviderRegistry | None = None,
    shim: OllamaTrackShim | None = None,
) -> AnalysisOutcome:
    """Analyse one document with the provider/model/prompt in `config`.

    `api_key` is the plaintext BYOK key for gemini/openai/anthropic; it is
    ignored for ollama. `registry` and `shim` are test seams (defaults: the
    four production adapters, and a shim client on $AITY_OLLAMA_SHIM_URL).

    Raises (before any provider call): UnknownProviderError,
    UnknownModelError, ModelNotActiveError (an AITY Built-in model other
    than the operator's active one), UnknownPromptVariantError,
    PromptVariantNotAvailableError, MissingApiKeyError, ValueError (empty text).
    """
    registry = registry or default_registry()
    handle = registry.create(config.provider, config.model)

    if handle.requires_api_key:
        if not api_key:
            raise MissingApiKeyError(f"{config.provider}: an API key is required")
    else:
        api_key = None  # AITY Built-in never receives a key

    prompt = render_prompt(config.prompt_variant, text, profile=handle.prompt_profile)
    init_ecologits()  # no-op after the first call; FastAPI startup takes this over in W5

    resp: ProviderResponse | None = None
    call_error: Exception | None = None
    track: dict[str, Any] | None = None

    if config.provider == "ollama":
        async with _shim_scope(shim) as active_shim:
            track_id = await active_shim.start()
            try:
                with wall_clock() as timer:
                    resp, call_error = await _call(handle, api_key, prompt)
            finally:
                track = await active_shim.stop(track_id, handle.model)
    else:
        with wall_clock() as timer:
            resp, call_error = await _call(handle, api_key, prompt)

    metrics = collect(config.provider, resp, timer.seconds or 0.0, ollama_track_result=track)
    base = dict(
        provider=config.provider,
        model=handle.model,
        prompt_variant=prompt.variant,
        prompt_profile=prompt.profile,
        sustainability=metrics,
    )

    if call_error is not None:
        return AnalysisOutcome(status="failed", error=_call_error_text(call_error), **base)

    assert resp is not None
    base.update(finish_reason=resp.finish_reason, usage=resp.usage)

    failure = _response_failure(resp)
    if failure:
        return AnalysisOutcome(status="failed", error=failure, **base)

    try:
        payload = normalize_payload(LLMAnalysisPayload.model_validate(resp.structured))
    except ValidationError as exc:
        log.warning(
            "%s/%s: output does not validate as LLMAnalysisPayload (%d errors): %s",
            config.provider, handle.model, exc.error_count(), resp.text[:_RAW_LOG_CHARS],
        )
        return AnalysisOutcome(
            status="failed",
            error=f"output does not match the analysis schema ({exc.error_count()} errors)",
            **base,
        )

    issues = tuple(payload_issues(payload))
    return AnalysisOutcome(
        status="partial" if issues else "ok", payload=payload, issues=issues, **base
    )


# --- helpers ---


async def _call(
    handle: ModelHandle, api_key: str | None, prompt: RenderedPrompt
) -> tuple[ProviderResponse | None, Exception | None]:
    """Run the adapter; an exception is returned, not raised, so the caller
    can still stop the shim and build a failed outcome with its latency."""
    try:
        resp = await handle.complete(
            api_key=api_key,
            messages=list(prompt.messages),
            system_prompt=prompt.system_prompt,
            response_schema=llm_analysis_payload_schema(),
        )
        return resp, None
    except Exception as exc:  # SDK errors propagate from adapters (ADR-002 §7.7)
        log.warning("%s/%s: provider call failed", handle.provider, handle.model, exc_info=True)
        return None, exc


def _call_error_text(exc: Exception) -> str:
    """Sanitised: the exception type only. SDK messages can echo request
    details, so the full text stays in the log (W3 adds ProviderError kinds)."""
    return f"provider call failed ({type(exc).__name__})"


def _response_failure(resp: ProviderResponse) -> str | None:
    """Reason a response cannot yield a payload, or None if it can."""
    if resp.finish_reason == "length":
        return "output hit the token limit"  # ADR-008 retry lands in W6
    if resp.finish_reason == "content_filter":
        return "the provider blocked the request or the output"
    if resp.structured is None:
        log.warning("unparseable model output (%d chars): %s", len(resp.text), resp.text[:_RAW_LOG_CHARS])
        return "output was not valid JSON"
    return None


@asynccontextmanager
async def _shim_scope(shim: OllamaTrackShim | None) -> AsyncIterator[OllamaTrackShim]:
    """Yield the injected shim, or a temporary one with its own httpx client."""
    if shim is not None:
        yield shim
        return
    async with httpx.AsyncClient() as client:
        yield OllamaTrackShim(client)


__all__ = ["AnalysisOutcome", "AnalysisStatus", "MissingApiKeyError", "analyze", "default_registry"]
