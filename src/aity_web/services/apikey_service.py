"""BYOK API keys for the three cloud providers - ADR-004 §9, ADR-005 (W4).

Only gemini, openai and anthropic take a user key; AITY Built-in (Ollama)
never needs one and is refused here.

What is stored: cipher.encrypt() of a compact JSON payload
    {"apiKey": "...", "provider": "openai"}
(the payload shape from the BYOK decision; "baseURL" and "extra" are reserved
for OpenAI-compatible overrides later and not written in v1). The provider
inside the payload is checked on every read, so a ciphertext copied onto
another provider's row is rejected.

The cipher is injected:
* W3: FakeCipher - reversible encoding, NOT encryption. Tests and
  development only.
* W4: a Fernet cipher reading the master key from /run/secrets/master.key
  (ADR-005), with the same two methods.

Plaintext keys only exist in this service and in the caller's local variable
(run_analysis passes it straight to the adapter). They are never logged,
never put in exception messages and never returned by list_keys().
"""

from __future__ import annotations

import base64
import binascii
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, runtime_checkable
from uuid import UUID

from aity_web.models.user_api_key import CLOUD_PROVIDER_NAMES, CloudProviderName
from aity_web.services._common import UowFactory

API_KEY_MAX = 512


# --- cipher ---


class CipherError(Exception):
    """A ciphertext could not be decrypted (corrupt, or another master key)."""


@runtime_checkable
class Cipher(Protocol):
    def encrypt(self, plaintext: bytes) -> bytes: ...

    def decrypt(self, ciphertext: bytes) -> bytes:
        """CipherError if the ciphertext is not valid for this cipher."""
        ...


class FakeCipher:
    """W3 stand-in for Fernet: a tagged base64 encoding. NOT ENCRYPTION -
    anyone with database access can read the keys. Never use it on Pouta."""

    _TAG = b"aity-fake-v1:"

    def encrypt(self, plaintext: bytes) -> bytes:
        return self._TAG + base64.urlsafe_b64encode(plaintext)

    def decrypt(self, ciphertext: bytes) -> bytes:
        if not ciphertext.startswith(self._TAG):
            raise CipherError("not a FakeCipher ciphertext")
        try:
            return base64.b64decode(ciphertext[len(self._TAG):], altchars=b"-_", validate=True)
        except (binascii.Error, ValueError):
            raise CipherError("not a FakeCipher ciphertext") from None


# --- errors ---


class ApiKeyError(ValueError):
    """Base class; messages never contain a key."""


class UnsupportedKeyProviderError(ApiKeyError):
    """Not one of the three cloud providers (e.g. "ollama")."""


class InvalidApiKeyError(ApiKeyError):
    """Empty, too long, or containing whitespace/non-ASCII characters."""


class StoredKeyUnreadableError(ApiKeyError):
    """The stored ciphertext cannot be decrypted or does not hold a key for
    this provider. The user has to enter the key again."""


# --- service ---


@dataclass(frozen=True)
class ApiKeyInfo:
    """What the API may show about a stored key: never the key itself."""

    provider: CloudProviderName
    created_at: datetime
    updated_at: datetime


class ApiKeyService:
    def __init__(self, uow_factory: UowFactory, cipher: Cipher) -> None:
        self._uow_factory = uow_factory
        self._cipher = cipher

    def __repr__(self) -> str:
        return f"ApiKeyService(cipher={type(self._cipher).__name__})"

    async def set_key(
        self, *, user_id: UUID, provider: str, api_key: str
    ) -> ApiKeyInfo:
        """Store or replace the user's key for a provider (surrounding
        whitespace is stripped). NotFoundError if the user is unknown."""
        provider = _check_provider(provider)
        key = _check_key(api_key)
        payload = json.dumps(
            {"apiKey": key, "provider": provider}, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
        ciphertext = self._cipher.encrypt(payload)
        async with self._uow_factory() as uow:
            row = await uow.credentials.upsert(
                user_id=user_id, provider=provider, ciphertext=ciphertext
            )
            await uow.commit()
        return ApiKeyInfo(provider=row.provider, created_at=row.created_at, updated_at=row.updated_at)

    async def get_key(self, *, user_id: UUID, provider: str) -> str | None:
        """The plaintext key, or None if the user has none for this provider.
        StoredKeyUnreadableError if the stored value cannot be decrypted."""
        provider = _check_provider(provider)
        async with self._uow_factory() as uow:
            row = await uow.credentials.get(user_id=user_id, provider=provider)
        if row is None:
            return None
        return self._open(row.ciphertext, provider)

    async def list_keys(self, *, user_id: UUID) -> list[ApiKeyInfo]:
        """Which providers have a stored key (no decryption, no plaintext)."""
        async with self._uow_factory() as uow:
            rows = await uow.credentials.list(user_id=user_id)
        return [
            ApiKeyInfo(provider=r.provider, created_at=r.created_at, updated_at=r.updated_at)
            for r in rows
        ]

    async def delete_key(self, *, user_id: UUID, provider: str) -> None:
        """NotFoundError if the user has no key for this provider."""
        provider = _check_provider(provider)
        async with self._uow_factory() as uow:
            await uow.credentials.delete(user_id=user_id, provider=provider)
            await uow.commit()

    def _open(self, ciphertext: bytes, provider: str) -> str:
        try:
            payload = json.loads(self._cipher.decrypt(ciphertext).decode("utf-8"))
        except (CipherError, UnicodeDecodeError, ValueError):
            raise StoredKeyUnreadableError(
                f"the stored {provider} key cannot be read; enter it again"
            ) from None
        key = payload.get("apiKey") if isinstance(payload, dict) else None
        if _provider_mismatch(payload, provider) or not isinstance(key, str) or not key:
            raise StoredKeyUnreadableError(
                f"the stored {provider} key cannot be read; enter it again"
            )
        return key


def _provider_mismatch(payload: object, provider: str) -> bool:
    return not isinstance(payload, dict) or payload.get("provider") != provider


def _check_provider(provider: str) -> CloudProviderName:
    if provider not in CLOUD_PROVIDER_NAMES:
        if provider == "ollama":
            raise UnsupportedKeyProviderError("AITY Built-in does not use an API key")
        raise UnsupportedKeyProviderError(
            f"API keys are only stored for {', '.join(CLOUD_PROVIDER_NAMES)}"
        )
    return provider  # type: ignore[return-value]


def _check_key(api_key: str) -> str:
    if not isinstance(api_key, str):
        raise InvalidApiKeyError("the API key must be text")
    key = api_key.strip()
    if not key:
        raise InvalidApiKeyError("the API key is empty")
    if len(key) > API_KEY_MAX:
        raise InvalidApiKeyError(f"the API key is longer than {API_KEY_MAX} characters")
    if any(not ("!" <= ch <= "~") for ch in key):  # printable ASCII, no spaces
        raise InvalidApiKeyError("the API key contains spaces or unexpected characters")
    return key


__all__ = [
    "API_KEY_MAX",
    "ApiKeyError",
    "ApiKeyInfo",
    "ApiKeyService",
    "Cipher",
    "CipherError",
    "FakeCipher",
    "InvalidApiKeyError",
    "StoredKeyUnreadableError",
    "UnsupportedKeyProviderError",
]
