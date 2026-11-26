"""Analysis service: analyze() is the provider-only core; AnalysisService
.run_analysis() puts it on a stored document and saves the result (ADR-004 §7.4).

    result = await AnalysisService(uow_factory, storage, apikeys).run_analysis(
        user_id=uid, document_id=did, provider="gemini", model="gemini-3.5-flash-lite")

run_analysis() - two short transactions, the LLM call with none open:
  1. read: load the document (owner only), check the provider/model with the
     registry, read + clean its text, compute content_hash, get the user's
     API key from ApiKeyService (its own short read)
  2. analyze() - no transaction open; can take minutes on AITY Built-in
  3. write: map the AnalysisOutcome to an AnalysisResult (failed rows
     included), insert it, bump the session's updated_at, commit
The W5 job runner wraps run_analysis(); routes never call analyze() directly
(a review rule, no import-linter contract). W6 adds the cache lookup between
1 and 2 (ADR-008); truncation stays False until ADR-007.

analyze(text, config, api_key) - one document, one provider call, no
persistence. Its signature is unchanged since W2 so the unit tests and the
smoke suite keep calling it:

    catalogue/registry -> prompt library -> [shim start] -> timed adapter.complete()
    -> [shim stop] -> parse + validate + normalise -> SustainabilityCollector

Rules for analyze():
* Config mistakes (unknown provider/model/variant, missing key, empty text)
  raise before any network call. Everything that goes wrong AFTER the call
  starts becomes status="failed" with a short sanitised `error`; the raw
  model output and exception text go to the server log only.
* latency_seconds times adapter.complete() only; the Ollama shim calls sit
  outside the timer (ADR-003 §7.3) and the shim is always stopped.
* Sustainability never fails an analysis (the collector never raises).

Still to come: the ADR-008 one-shot retry (W6, inside analyze()), the
content-hash cache lookup (W6, inside run_analysis()), truncation (ADR-007),
the async job queue and concurrency ceiling (ADR-006, W5), and
ProviderError(kind=...) mapping of SDK exceptions (ADR-002 §7.7).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable, AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from typing import Any, Literal
from uuid import UUID, uuid4

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
from aity_web.models.analysis_result import AnalysisResult
from aity_web.models.document import Document
from aity_web.models.llm_config import DEFAULT_PROMPT_VARIANT, LLMConfig
from aity_web.models.sustainability import SustainabilityMetrics
from aity_web.repositories.base import ConflictError, NotFoundError
from aity_web.services._common import Clock, UowFactory, utc_now
from aity_web.services.apikey_service import ApiKeyService
from aity_web.services.cache_key import CACHE_KEY_VERSION, clean_text, content_hash, text_sha256
from aity_web.services.ollama_track_shim import OllamaTrackShim
from aity_web.services.sustainability import collect, init_ecologits, wall_clock
from aity_web.services.upload_storage import UploadStorage

log = logging.getLogger(__name__)

AnalysisStatus = Literal["ok", "partial", "failed"]

ANALYSIS_PROMPT_VARIANT = DEFAULT_PROMPT_VARIANT
"""The operator's prompt variant for analyses; requests never choose one."""

_RAW_LOG_CHARS = 2000


class MissingApiKeyError(ValueError):
    """A BYOK provider was called without an API key."""


class DocumentTextError(RuntimeError):
    """The document's stored text file is missing, unreadable, or no longer
    matches the text_sha256 recorded at upload. Raised before any provider
    call; the message is safe to show (no path, no text)."""


class DocumentGoneError(NotFoundError):
    """The document (or its session) was deleted while the analysis ran;
    the result was not saved."""


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
async def _shim_scope(shim: OllamaTrackShim | None) -> AsyncGenerator[OllamaTrackShim]:
    """Yield the injected shim, or a temporary one with its own httpx client."""
    if shim is not None:
        yield shim
        return
    async with httpx.AsyncClient() as client:
        yield OllamaTrackShim(client)


# --- run_analysis(): the stored-document path (ADR-004 §7.4) ---


def outcome_to_result(
    outcome: AnalysisOutcome,
    *,
    analysis_id: UUID,
    document: Document,
    content_hash: str,
    created_at: datetime,
    finished_at: datetime,
    cache_key_version: int = CACHE_KEY_VERSION,
) -> AnalysisResult:
    """Map what analyze() returned to the row that gets stored.

    status   payload   token_usage                         error
    ok       set       outcome.usage                       None
    partial  set       outcome.usage                       issues joined with "; "
    failed   None      usage if a response arrived, else   outcome.error
                       None (the call raised)
    """
    if outcome.status == "failed":
        token_usage = outcome.usage if outcome.finish_reason is not None else None
        error = outcome.error or "analysis failed"
    else:
        token_usage = outcome.usage
        error = "; ".join(outcome.issues) if outcome.status == "partial" else None
    return AnalysisResult(
        analysis_id=analysis_id,
        session_id=document.session_id,
        document_id=document.id,
        user_id=document.user_id,
        provider=outcome.provider,
        model=outcome.model,
        prompt_variant=outcome.prompt_variant,
        payload=outcome.payload if outcome.status != "failed" else None,
        token_usage=token_usage,
        sustainability=outcome.sustainability,
        content_hash=content_hash,
        cache_key_version=cache_key_version,
        truncated=False,  # ADR-007
        status=outcome.status,
        error=error,
        created_at=created_at,
        finished_at=finished_at,
    )


class AnalysisService:
    """Analyses of stored documents. One instance per process (W5 builds it
    with the real unit-of-work factory, UploadStorage and ApiKeyService)."""

    def __init__(
        self,
        uow_factory: UowFactory,
        storage: UploadStorage,
        apikeys: ApiKeyService,
        *,
        registry: LLMProviderRegistry | None = None,
        shim: OllamaTrackShim | None = None,
        clock: Clock = utc_now,
        new_id: Callable[[], UUID] = uuid4,
    ) -> None:
        self._uow_factory = uow_factory
        self._storage = storage
        self._apikeys = apikeys
        self._registry = registry
        self._shim = shim
        self._clock = clock
        self._new_id = new_id

    async def run_analysis(
        self, *, user_id: UUID, document_id: UUID, provider: str, model: str
    ) -> AnalysisResult:
        """Analyse one of the user's documents and store the result.

        Raises before any provider call (nothing is stored):
        * NotFoundError (document missing or not the user's)
        * UnknownProviderError
        * UnknownModelError
        * ModelNotActiveError (ADR-009, HTTP 422 in W5)
        * DocumentTextError
        * MissingApiKeyError
        * StoredKeyUnreadableError
        and the prompt errors of analyze().

        Raises after the call: 
        * DocumentGoneError if the document was deleted meanwhile (the result is dropped and logged).
        
        Every other outcome, failed included, is stored and returned."""
        registry = self._registry or default_registry()

        # --- 1. read: one short transaction for the row ---
        async with self._uow_factory() as uow:
            document = await uow.documents.get(user_id=user_id, document_id=document_id)

        handle = registry.create(provider, model)  # unknown / inactive model: refuse early
        config = LLMConfig(provider=provider, model=handle.model,
                           prompt_variant=ANALYSIS_PROMPT_VARIANT)
        text = await self._load_text(document)
        key = content_hash(text_sha256=document.text_sha256, provider=config.provider,
                           model=config.model, prompt_variant=config.prompt_variant)
        # W6: per-user cache lookup on (key, CACHE_KEY_VERSION) goes here (ADR-008)

        api_key: str | None = None
        if handle.requires_api_key:
            api_key = await self._apikeys.get_key(user_id=user_id, provider=provider)
            if not api_key:
                raise MissingApiKeyError(f"no {provider} API key is saved for this account")

        # --- 2. the LLM call, no transaction open ---
        created_at = self._clock()
        outcome = await analyze(text, config, api_key, registry=registry, shim=self._shim)
        del api_key  # the plaintext key lives no longer than the call
        finished_at = self._clock()

        result = outcome_to_result(
            outcome, analysis_id=self._new_id(), document=document,
            content_hash=key, created_at=created_at, finished_at=finished_at,
        )

        # --- 3. write: one short transaction ---
        try:
            async with self._uow_factory() as uow:
                await uow.analyses.add(user_id=user_id, result=result)
                await uow.sessions.touch(user_id=user_id, session_id=document.session_id)
                await uow.commit()
        except (NotFoundError, ConflictError) as exc:
            log.warning(
                "analysis %s of document %s dropped: the document was deleted during the call (%s)",
                result.analysis_id, document_id, type(exc).__name__,
            )
            raise DocumentGoneError(
                "the document was deleted while it was being analysed"
            ) from exc
        return result

    async def get(self, *, user_id: UUID, analysis_id: UUID) -> AnalysisResult:
        async with self._uow_factory() as uow:
            return await uow.analyses.get(user_id=user_id, analysis_id=analysis_id)

    async def list_for_document(self, *, user_id: UUID, document_id: UUID) -> list[AnalysisResult]:
        """Newest first. NotFoundError if the document is missing or not the user's."""
        async with self._uow_factory() as uow:
            return await uow.analyses.list_for_document(user_id=user_id, document_id=document_id)

    async def list_for_session(self, *, user_id: UUID, session_id: UUID) -> list[AnalysisResult]:
        """Newest first. NotFoundError if the session is missing or not the user's."""
        async with self._uow_factory() as uow:
            return await uow.analyses.list_for_session(user_id=user_id, session_id=session_id)

    async def delete(self, *, user_id: UUID, analysis_id: UUID) -> None:
        """Delete one result (no files involved)."""
        async with self._uow_factory() as uow:
            await uow.analyses.delete(user_id=user_id, analysis_id=analysis_id)
            await uow.commit()

    async def _load_text(self, document: Document) -> str:
        """The cleaned text, checked against the hash recorded at upload."""
        try:
            raw = await asyncio.to_thread(self._storage.read_text, document.text_path)
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            log.error("document %s: text file unreadable (%s)", document.id, type(exc).__name__)
            raise DocumentTextError("the document's text could not be read") from None
        text = clean_text(raw)
        if text_sha256(text) != document.text_sha256:
            log.error("document %s: text file does not match its upload hash", document.id)
            raise DocumentTextError("the document's stored text has changed since upload")
        return text


__all__ = [
    "ANALYSIS_PROMPT_VARIANT",
    "AnalysisOutcome",
    "AnalysisService",
    "AnalysisStatus",
    "DocumentGoneError",
    "DocumentTextError",
    "MissingApiKeyError",
    "analyze",
    "default_registry",
    "outcome_to_result",
]
