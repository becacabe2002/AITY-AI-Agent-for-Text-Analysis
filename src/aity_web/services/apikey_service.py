"""Service for storing and reading users' encrypted cloud-provider API keys (BYOK)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from aity_web.models.user_api_key import CLOUD_PROVIDER_NAMES, CloudProviderName
from aity_web.services._common import UowFactory
from aity_web.services.crypto import Cipher, CipherError

API_KEY_MAX = 512


# --- errors ---


class ApiKeyError(ValueError):
    """Base class; messages never contain a key."""


class UnsupportedKeyProviderError(ApiKeyError):
    """Not one of the three cloud providers (e.g. "ollama")."""


class InvalidApiKeyError(ApiKeyError):
    """Empty, too long, or containing whitespace/non-ASCII characters."""


class StoredKeyUnreadableError(ApiKeyError):
    """The stored key cannot be decrypted; the user must enter it again."""


# --- service ---


@dataclass(frozen=True)
class ApiKeyInfo:
    """What the API may show about a stored key: never the key itself."""

    provider: CloudProviderName
    created_at: datetime
    updated_at: datetime


class ApiKeyService:
    """Stores, reads, lists and deletes a user's cloud-provider API keys."""

    def __init__(self, uow_factory: UowFactory, cipher: Cipher) -> None:
        """Create the service with a unit-of-work factory and a cipher."""
        self._uow_factory = uow_factory
        self._cipher = cipher

    def __repr__(self) -> str:
        return f"ApiKeyService(cipher={type(self._cipher).__name__})"

    async def set_key(
        self, *, user_id: UUID, provider: str, api_key: str
    ) -> ApiKeyInfo:
        """Store or replace the user's key for a provider and return its metadata."""
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
        """Return the plaintext key, or None if none is stored."""
        provider = _check_provider(provider)
        async with self._uow_factory() as uow:
            row = await uow.credentials.get(user_id=user_id, provider=provider)
        if row is None:
            return None
        return self._open(row.ciphertext, provider)

    async def list_keys(self, *, user_id: UUID) -> list[ApiKeyInfo]:
        """List which providers have a stored key, without decrypting anything."""
        async with self._uow_factory() as uow:
            rows = await uow.credentials.list(user_id=user_id)
        return [
            ApiKeyInfo(provider=r.provider, created_at=r.created_at, updated_at=r.updated_at)
            for r in rows
        ]

    async def delete_key(self, *, user_id: UUID, provider: str) -> None:
        """Delete the user's key for a provider."""
        provider = _check_provider(provider)
        async with self._uow_factory() as uow:
            await uow.credentials.delete(user_id=user_id, provider=provider)
            await uow.commit()

    def _open(self, ciphertext: bytes, provider: str) -> str:
        """Decrypt a stored payload, check its shape and provider, and return the key."""
        unreadable = StoredKeyUnreadableError(
            f"the stored {provider} key cannot be read; enter it again"
        )
        try:
            payload = json.loads(self._cipher.decrypt(ciphertext).decode("utf-8"))
        except (CipherError, UnicodeDecodeError, ValueError):
            raise unreadable from None
        if not _payload_ok(payload, provider):
            raise unreadable
        return payload["apiKey"]


def _payload_ok(payload: object, provider: str) -> bool:
    """True if the payload is {apiKey, provider, baseURL?, extra?} for this provider."""
    if not isinstance(payload, dict) or payload.get("provider") != provider:
        return False
    key = payload.get("apiKey")
    if not isinstance(key, str) or not key:
        return False
    if "baseURL" in payload and not isinstance(payload["baseURL"], str):
        return False
    return "extra" not in payload or isinstance(payload["extra"], dict)


def _check_provider(provider: str) -> CloudProviderName:
    """Check that the provider takes a user key and return it."""
    if provider not in CLOUD_PROVIDER_NAMES:
        if provider == "ollama":
            raise UnsupportedKeyProviderError("AITY Built-in does not use an API key")
        raise UnsupportedKeyProviderError(
            f"API keys are only stored for {', '.join(CLOUD_PROVIDER_NAMES)}"
        )
    return provider  # type: ignore[return-value]


def _check_key(api_key: str) -> str:
    """Validate and strip an API key."""
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
    "InvalidApiKeyError",
    "StoredKeyUnreadableError",
    "UnsupportedKeyProviderError",
]
