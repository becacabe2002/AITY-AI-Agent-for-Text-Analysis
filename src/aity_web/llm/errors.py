"""
ProviderError: one error type for every LLM provider (ADR-002 §7.7).

Adapters let SDK exceptions propagate unchanged. ModelHandle.complete()
(llm/registry.py) is the one place every provider call passes through; it
passes each exception to to_provider_error() and raises the ProviderError
it returns, chained to the original (`raise ... from exc`), so the SDK
details stay in the server log.

    kind           meaning                                     retryable
    auth           key invalid, revoked or lacking permission  no
    rate_limit     429; quota_exhausted=True for no credit     yes / no
                   (402, OpenAI insufficient_quota, Anthropic
                   billing / low credit balance)
    unavailable    5xx, overloaded, timeout, connection error  yes
    bad_request    any other 4xx (bad model id, too large...)  no

Only provider and transport errors are mapped. Anything else (TypeError for
an unsupported option, ValueError for a missing key) is a programming or
config error and is re-raised unchanged.

ProviderError's str() and user_message never contain SDK text: provider
messages can echo request details, so they stay on the chained exception.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any, Literal, get_args

import anthropic
import httpx
import openai
from google.genai import errors as genai_errors

from aity_web.llm.providers.ollama import OllamaError

ProviderErrorKind = Literal["auth", "rate_limit", "unavailable", "bad_request"]
PROVIDER_ERROR_KINDS: tuple[str, ...] = get_args(ProviderErrorKind)

#: Status codes that mean "try again later" (529 = Anthropic overloaded).
_UNAVAILABLE_STATUS: frozenset[int] = frozenset({408, 409, 529})

_MESSAGES: dict[str, str] = {
    "auth": "The provider rejected the API key. Check the key saved in your account settings.",
    "rate_limit": "The provider is rate-limiting requests for this key. Try again in a few minutes.",
    "quota": "The provider reports no remaining credit or quota for this key's account.",
    "unavailable": "The provider could not be reached or is temporarily unavailable. Try again later.",
    "timeout": "The provider did not answer in time. Try again later.",
    "bad_request": "The provider rejected the request.",
}


class ProviderError(RuntimeError):
    """A provider call failed; `kind` says what the user can do about it.

    * kind            - see the module table above
    * provider        - registry name (gemini, openai, anthropic, ollama)
    * status_code     - HTTP status, None for timeouts and connection errors
    * retryable       - True for rate_limit (unless quota_exhausted) and
      unavailable; the W5 job queue can use it for a delayed retry
    * retry_after_s   - the provider's Retry-After header, when sent
    * quota_exhausted - rate_limit caused by no credit/quota (OpenAI
      insufficient_quota, HTTP 402): waiting does not help
    * timed_out       - unavailable because the call timed out
    """

    def __init__(
        self,
        kind: ProviderErrorKind,
        provider: str,
        *,
        status_code: int | None = None,
        retry_after_s: float | None = None,
        quota_exhausted: bool = False,
        timed_out: bool = False,
    ) -> None:
        if kind not in PROVIDER_ERROR_KINDS:
            raise ValueError(f"kind must be one of {PROVIDER_ERROR_KINDS}, got {kind!r}")
        self.kind: ProviderErrorKind = kind
        self.provider = provider
        self.status_code = status_code
        self.retry_after_s = retry_after_s
        self.quota_exhausted = quota_exhausted
        self.timed_out = timed_out
        status = f" (HTTP {status_code})" if status_code is not None else ""
        detail = ", quota exhausted" if quota_exhausted else ", timed out" if timed_out else ""
        super().__init__(f"{provider}: {kind}{detail}{status}")

    @property
    def retryable(self) -> bool:
        if self.kind == "rate_limit":
            return not self.quota_exhausted
        return self.kind == "unavailable"

    @property
    def user_message(self) -> str:
        """Short, sanitised text for AnalysisResult.error and the UI."""
        if self.quota_exhausted:
            return _MESSAGES["quota"]
        if self.timed_out:
            return _MESSAGES["timeout"]
        message = _MESSAGES[self.kind]
        if self.kind == "bad_request" and self.status_code is not None:
            message = f"{message[:-1]} (HTTP {self.status_code})."
        return message


# --- mapping ---


def kind_for_status(status_code: int) -> ProviderErrorKind:
    """The one status-code rule shared by all four providers."""
    if status_code in (401, 403):
        return "auth"
    if status_code in (402, 429):
        return "rate_limit"
    if status_code >= 500 or status_code in _UNAVAILABLE_STATUS:
        return "unavailable"
    return "bad_request"


def to_provider_error(provider: str, exc: BaseException) -> ProviderError | None:
    """Map an exception raised by an adapter; None = not a provider error."""
    if isinstance(exc, ProviderError):
        return exc

    # OpenAI and Anthropic SDKs: same exception layout (both Stainless-generated)
    if isinstance(exc, (openai.APIStatusError, anthropic.APIStatusError)):
        return _from_status(
            provider,
            exc.status_code,
            headers=exc.response.headers,
            quota_exhausted=_openai_insufficient_quota(exc) or _anthropic_billing(exc),
        )
    if isinstance(exc, (openai.APIConnectionError, anthropic.APIConnectionError)):
        timed_out = isinstance(exc, (openai.APITimeoutError, anthropic.APITimeoutError))
        return ProviderError("unavailable", provider, timed_out=timed_out)

    # google-genai: APIError(code=HTTP status, status="INVALID_ARGUMENT", details=body)
    if isinstance(exc, genai_errors.APIError):
        if _gemini_invalid_key(exc):     # Gemini answers 400, not 401, for a bad key
            return ProviderError("auth", provider, status_code=exc.code)
        headers = exc.response.headers if isinstance(exc.response, httpx.Response) else None
        if isinstance(exc.code, int):
            return _from_status(provider, exc.code, headers=headers)
        return ProviderError("unavailable", provider)

    # Ollama (AITY Built-in)
    if isinstance(exc, OllamaError):
        return _from_status(provider, exc.status_code)

    # Transport errors outside the SDK wrappers: Ollama's httpx client, and
    # google-genai, which lets httpx exceptions through
    if isinstance(exc, (httpx.TimeoutException, TimeoutError, asyncio.TimeoutError)):
        return ProviderError("unavailable", provider, timed_out=True)
    if isinstance(exc, httpx.TransportError):
        return ProviderError("unavailable", provider)

    return None


def _from_status(
    provider: str,
    status_code: int,
    *,
    headers: Mapping[str, str] | None = None,
    quota_exhausted: bool = False,
) -> ProviderError:
    quota = quota_exhausted or status_code == 402
    kind: ProviderErrorKind = "rate_limit" if quota else kind_for_status(status_code)
    return ProviderError(
        kind,
        provider,
        status_code=status_code,
        retry_after_s=_retry_after(headers) if kind in ("rate_limit", "unavailable") else None,
        quota_exhausted=quota,
    )


def _retry_after(headers: Mapping[str, str] | None) -> float | None:
    """Retry-After in seconds; the HTTP-date form is ignored (no provider uses it)."""
    if not headers:
        return None
    value = headers.get("retry-after")
    try:
        seconds = float(value) if value is not None else None
    except ValueError:
        return None
    return seconds if seconds is not None and seconds >= 0 else None


def _error_body(exc: Any) -> Mapping[str, Any]:
    """The `error` object of an OpenAI/Anthropic error body, or {}."""
    body = getattr(exc, "body", None)
    if not isinstance(body, Mapping):
        return {}
    inner = body.get("error")
    return inner if isinstance(inner, Mapping) else body


def _openai_insufficient_quota(exc: Any) -> bool:
    # OpenAI sends 429 with code "insufficient_quota" when the account has no credit
    return getattr(exc, "code", None) == "insufficient_quota"


def _anthropic_billing(exc: Any) -> bool:
    # Anthropic: type "billing_error", or a 400 saying the credit balance is too low
    if not isinstance(exc, anthropic.APIStatusError):
        return False
    error = _error_body(exc)
    message = error.get("message")
    return error.get("type") == "billing_error" or (
        isinstance(message, str) and "credit balance is too low" in message.lower()
    )


def _gemini_invalid_key(exc: genai_errors.APIError) -> bool:
    # error.details[] holds an ErrorInfo with reason "API_KEY_INVALID"
    return "API_KEY_INVALID" in str(exc.details) or "API key not valid" in (exc.message or "")


__all__ = [
    "PROVIDER_ERROR_KINDS",
    "ProviderError",
    "ProviderErrorKind",
    "kind_for_status",
    "to_provider_error",
]
