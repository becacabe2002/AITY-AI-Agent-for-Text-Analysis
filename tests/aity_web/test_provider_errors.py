"""ProviderError mapping (ADR-002 §7.7): real SDK exception objects -> kind."""

from __future__ import annotations

import asyncio

import anthropic
import httpx
import openai
import pytest
from google.genai import errors as genai_errors

from aity_web.llm.catalog import load_catalog
from aity_web.llm.errors import (
    PROVIDER_ERROR_KINDS,
    ProviderError,
    kind_for_status,
    to_provider_error,
)
from aity_web.llm.providers.base import BaseProviderAdapter, ProviderResponse
from aity_web.llm.providers.ollama import OllamaError, OllamaProvider
from aity_web.llm.registry import OLLAMA_MODEL_ENV, LLMProviderRegistry, build_default_registry
from aity_web.llm.types import user_message

REQ = httpx.Request("POST", "https://api.example.test/v1/x")
SECRET = "sk-SECRET-123"


def _resp(status: int, headers: dict | None = None) -> httpx.Response:
    return httpx.Response(status, request=REQ, headers=headers or {})


def _openai(cls, status, body=None, headers=None):
    return cls(f"Error code: {status} - key {SECRET}", response=_resp(status, headers), body=body)


def _anthropic(cls, status, body=None, headers=None):
    return cls(f"Error code: {status} - key {SECRET}", response=_resp(status, headers), body=body)


@pytest.fixture(autouse=True)
def _no_operator_ollama_model(monkeypatch):
    monkeypatch.delenv(OLLAMA_MODEL_ENV, raising=False)


# --- status rule ---


@pytest.mark.parametrize(
    ("status", "kind"),
    [
        (400, "bad_request"), (401, "auth"), (402, "rate_limit"), (403, "auth"),
        (404, "bad_request"), (408, "unavailable"), (409, "unavailable"),
        (413, "bad_request"), (422, "bad_request"), (429, "rate_limit"),
        (500, "unavailable"), (503, "unavailable"), (529, "unavailable"),
    ],
)
def test_kind_for_status(status, kind):
    assert kind_for_status(status) == kind


# --- OpenAI ---


@pytest.mark.parametrize(
    ("cls", "status", "kind", "retryable"),
    [
        (openai.AuthenticationError, 401, "auth", False),
        (openai.PermissionDeniedError, 403, "auth", False),
        (openai.RateLimitError, 429, "rate_limit", True),
        (openai.BadRequestError, 400, "bad_request", False),
        (openai.NotFoundError, 404, "bad_request", False),
        (openai.InternalServerError, 500, "unavailable", True),
    ],
)
def test_openai_status_errors(cls, status, kind, retryable):
    err = to_provider_error("openai", _openai(cls, status))
    assert (err.kind, err.status_code, err.retryable, err.provider) == (kind, status, retryable, "openai")


def test_openai_insufficient_quota_is_not_retryable():
    exc = _openai(openai.RateLimitError, 429, body={"code": "insufficient_quota", "type": "insufficient_quota"})
    err = to_provider_error("openai", exc)
    assert err.kind == "rate_limit" and err.quota_exhausted and not err.retryable
    assert "credit or quota" in err.user_message


def test_retry_after_header_is_read():
    exc = _openai(openai.RateLimitError, 429, headers={"retry-after": "12"})
    assert to_provider_error("openai", exc).retry_after_s == 12.0
    exc = _openai(openai.RateLimitError, 429, headers={"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"})
    assert to_provider_error("openai", exc).retry_after_s is None


def test_openai_connection_and_timeout():
    err = to_provider_error("openai", openai.APIConnectionError(request=REQ))
    assert err.kind == "unavailable" and err.retryable and not err.timed_out and err.status_code is None
    err = to_provider_error("openai", openai.APITimeoutError(request=REQ))
    assert err.kind == "unavailable" and err.timed_out
    assert err.user_message == "The provider did not answer in time. Try again later."


# --- Anthropic ---


@pytest.mark.parametrize(
    ("cls", "status", "kind"),
    [
        (anthropic.AuthenticationError, 401, "auth"),
        (anthropic.RateLimitError, 429, "rate_limit"),
        (anthropic.RequestTooLargeError, 413, "bad_request"),
        (anthropic.OverloadedError, 529, "unavailable"),
        (anthropic.InternalServerError, 500, "unavailable"),
    ],
)
def test_anthropic_status_errors(cls, status, kind):
    assert to_provider_error("anthropic", _anthropic(cls, status)).kind == kind


def test_anthropic_low_credit_balance_is_quota():
    body = {"type": "error", "error": {"type": "invalid_request_error",
            "message": "Your credit balance is too low to access the Anthropic API."}}
    err = to_provider_error("anthropic", _anthropic(anthropic.BadRequestError, 400, body=body))
    assert err.kind == "rate_limit" and err.quota_exhausted and not err.retryable


def test_anthropic_billing_error_type_is_quota():
    body = {"type": "error", "error": {"type": "billing_error", "message": "..."}}
    err = to_provider_error("anthropic", _anthropic(anthropic.APIStatusError, 402, body=body))
    assert err.kind == "rate_limit" and err.quota_exhausted


def test_anthropic_timeout():
    err = to_provider_error("anthropic", anthropic.APITimeoutError(request=REQ))
    assert err.kind == "unavailable" and err.timed_out


# --- Gemini ---


def _gemini(code, status, message, details=None):
    body = {"error": {"code": code, "status": status, "message": message, "details": details or []}}
    cls = genai_errors.ServerError if code >= 500 else genai_errors.ClientError
    return cls(code, body)


def test_gemini_invalid_key_400_is_auth():
    exc = _gemini(400, "INVALID_ARGUMENT", "API key not valid. Please pass a valid API key.",
                  [{"@type": "type.googleapis.com/google.rpc.ErrorInfo", "reason": "API_KEY_INVALID"}])
    err = to_provider_error("gemini", exc)
    assert err.kind == "auth" and err.status_code == 400


@pytest.mark.parametrize(
    ("code", "status", "kind"),
    [
        (400, "INVALID_ARGUMENT", "bad_request"),
        (403, "PERMISSION_DENIED", "auth"),
        (404, "NOT_FOUND", "bad_request"),
        (429, "RESOURCE_EXHAUSTED", "rate_limit"),
        (500, "INTERNAL", "unavailable"),
        (503, "UNAVAILABLE", "unavailable"),
    ],
)
def test_gemini_status_errors(code, status, kind):
    assert to_provider_error("gemini", _gemini(code, status, "some message")).kind == kind


# --- Ollama and transport ---


@pytest.mark.parametrize(
    ("status", "kind"), [(400, "bad_request"), (404, "bad_request"), (500, "unavailable")]
)
def test_ollama_http_errors(status, kind):
    assert to_provider_error("ollama", OllamaError(status, "model 'x' not found")).kind == kind


@pytest.mark.parametrize(
    ("exc", "timed_out"),
    [
        (httpx.ConnectError("refused", request=REQ), False),
        (httpx.RemoteProtocolError("closed", request=REQ), False),
        (httpx.ReadTimeout("slow", request=REQ), True),
        (TimeoutError("slow"), True),
        (asyncio.TimeoutError(), True),
    ],
)
def test_transport_errors_are_unavailable(exc, timed_out):
    err = to_provider_error("ollama", exc)
    assert err.kind == "unavailable" and err.timed_out is timed_out and err.retryable


# --- what is NOT mapped ---


@pytest.mark.parametrize(
    "exc", [TypeError("unsupported options"), ValueError("an API key is required"), KeyError("x")]
)
def test_non_provider_errors_are_not_mapped(exc):
    assert to_provider_error("openai", exc) is None


def test_provider_error_passes_through():
    err = ProviderError("auth", "openai", status_code=401)
    assert to_provider_error("openai", err) is err


# --- ProviderError itself ---


def test_messages_never_contain_sdk_text():
    for exc in (
        _openai(openai.AuthenticationError, 401),
        _openai(openai.BadRequestError, 400),
        _anthropic(anthropic.RateLimitError, 429),
    ):
        err = to_provider_error("openai", exc)
        assert SECRET not in str(err) and SECRET not in err.user_message


def test_str_and_user_message():
    err = ProviderError("bad_request", "gemini", status_code=404)
    assert str(err) == "gemini: bad_request (HTTP 404)"
    assert err.user_message == "The provider rejected the request (HTTP 404)."
    assert str(ProviderError("unavailable", "ollama", timed_out=True)) == "ollama: unavailable, timed out"


def test_unknown_kind_rejected():
    with pytest.raises(ValueError):
        ProviderError("teapot", "openai")  # type: ignore[arg-type]
    assert set(PROVIDER_ERROR_KINDS) == {"auth", "rate_limit", "unavailable", "bad_request"}


# --- applied once, in ModelHandle.complete() ---


class RaisingAdapter(BaseProviderAdapter):
    name = "openai"

    def __init__(self, exc: BaseException) -> None:
        self.exc = exc

    async def complete(self, **kwargs) -> ProviderResponse:
        raise self.exc


def _handle(exc):
    registry = LLMProviderRegistry(load_catalog())
    registry.register(RaisingAdapter, factory=lambda: RaisingAdapter(exc))
    return registry.create("openai", "gpt-5.4-mini-2026-03-17")


@pytest.mark.asyncio
async def test_handle_maps_sdk_error_and_chains_original():
    original = _openai(openai.AuthenticationError, 401)
    with pytest.raises(ProviderError) as info:
        await _handle(original).complete(api_key="k", messages=[user_message("x")], system_prompt="s")
    assert info.value.kind == "auth"
    assert info.value.__cause__ is original


@pytest.mark.asyncio
async def test_handle_lets_programming_errors_through():
    with pytest.raises(TypeError, match="unsupported"):
        await _handle(TypeError("unsupported options")).complete(
            api_key="k", messages=[user_message("x")], system_prompt="s")


@pytest.mark.asyncio
async def test_real_ollama_adapter_error_is_mapped_through_the_registry():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "llama runner process has terminated"})

    registry = build_default_registry(load_catalog())
    adapter = registry.adapter("ollama")
    assert isinstance(adapter, OllamaProvider)
    adapter._transport = httpx.MockTransport(handler)
    handle = registry.create("ollama", "qwen3.5:4b")

    with pytest.raises(ProviderError) as info:
        await handle.complete(api_key=None, messages=[user_message("x")], system_prompt="s")
    assert (info.value.kind, info.value.status_code, info.value.provider) == ("unavailable", 500, "ollama")
    assert isinstance(info.value.__cause__, OllamaError)
