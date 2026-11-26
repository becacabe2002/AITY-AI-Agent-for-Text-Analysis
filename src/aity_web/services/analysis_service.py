"""Analysis service: analyze() makes one provider call; AnalysisService stores results for
documents.
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

from aity_web.llm.errors import ProviderError
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
"""Prompt variant used for analyses, set by the operator."""

_RAW_LOG_CHARS = 2000


class MissingApiKeyError(ValueError):
    """A BYOK provider was called without an API key."""


class DocumentTextError(RuntimeError):
    """The stored document text is missing, unreadable or changed since upload."""


class DocumentGoneError(NotFoundError):
    """The document was deleted while the analysis ran; the result was not saved."""


@dataclass(frozen=True, slots=True)
class AnalysisOutcome:
    """Outcome of one provider analysis, before it becomes a stored AnalysisResult."""

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
        """True unless the analysis failed."""
        return self.status != "failed"


@lru_cache(maxsize=1)
def default_registry() -> LLMProviderRegistry:
    """Return the shared provider registry, built on first use."""
    return build_default_registry()


async def analyze(
    text: str,
    config: LLMConfig,
    api_key: str | None = None,
    *,
    registry: LLMProviderRegistry | None = None,
    shim: OllamaTrackShim | None = None,
) -> AnalysisOutcome:
    """Analyse one document with the provider, model and prompt in config."""
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
    """Run the adapter and return its response or its exception instead of raising."""
    try:
        resp = await handle.complete(
            api_key=api_key,
            messages=list(prompt.messages),
            system_prompt=prompt.system_prompt,
            response_schema=llm_analysis_payload_schema(),
        )
        return resp, None
    except Exception as exc:  # ProviderError from ModelHandle, or an unexpected error
        log.warning("%s/%s: provider call failed: %s", handle.provider, handle.model, exc, exc_info=True)
        return None, exc


def _call_error_text(exc: Exception) -> str:
    """Return a sanitised error text for a failed provider call."""
    if isinstance(exc, ProviderError):
        return exc.user_message
    return f"provider call failed ({type(exc).__name__})"


def _response_failure(resp: ProviderResponse) -> str | None:
    """Return why a response cannot yield a payload, or None if it can."""
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
    """Map an AnalysisOutcome to the AnalysisResult row that gets stored."""
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
    """Runs and stores analyses of stored documents."""

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
        """Create the service with its unit-of-work factory, storage, key service and test seams."""
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
        """Analyse one of the user's documents and store the result, failed ones included."""
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
        """Return one of the user's results."""
        async with self._uow_factory() as uow:
            return await uow.analyses.get(user_id=user_id, analysis_id=analysis_id)

    async def list_for_document(self, *, user_id: UUID, document_id: UUID) -> list[AnalysisResult]:
        """List a document's results, newest first."""
        async with self._uow_factory() as uow:
            return await uow.analyses.list_for_document(user_id=user_id, document_id=document_id)

    async def list_for_session(self, *, user_id: UUID, session_id: UUID) -> list[AnalysisResult]:
        """List a session's results, newest first."""
        async with self._uow_factory() as uow:
            return await uow.analyses.list_for_session(user_id=user_id, session_id=session_id)

    async def delete(self, *, user_id: UUID, analysis_id: UUID) -> None:
        """Delete one result."""
        async with self._uow_factory() as uow:
            await uow.analyses.delete(user_id=user_id, analysis_id=analysis_id)
            await uow.commit()

    async def _load_text(self, document: Document) -> str:
        """Read the document's cleaned text and check it against the hash recorded at upload."""
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
