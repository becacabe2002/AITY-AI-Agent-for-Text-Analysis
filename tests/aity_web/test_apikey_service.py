"""apikey_service with the W3 FakeCipher and an injected test cipher."""

from __future__ import annotations

import json
from uuid import uuid4

import pytest

from _service_fixtures import StepClock, add_user
from aity_web.repositories import NotFoundError
from aity_web.repositories.memory import InMemoryStore
from aity_web.services.apikey_service import (
    API_KEY_MAX,
    ApiKeyService,
    Cipher,
    CipherError,
    FakeCipher,
    InvalidApiKeyError,
    StoredKeyUnreadableError,
    UnsupportedKeyProviderError,
)

KEY = "sk-test-0123456789abcdef"


class XorCipher:
    """A different cipher, to show the service only uses the two methods."""

    def encrypt(self, plaintext: bytes) -> bytes:
        return b"x:" + bytes(b ^ 0x5A for b in plaintext)

    def decrypt(self, ciphertext: bytes) -> bytes:
        if not ciphertext.startswith(b"x:"):
            raise CipherError("bad")
        return bytes(b ^ 0x5A for b in ciphertext[2:])


@pytest.fixture
def store():
    return InMemoryStore(clock=StepClock())


@pytest.fixture(params=[FakeCipher, XorCipher], ids=["fake", "xor"])
def cipher(request):
    return request.param()


@pytest.fixture
def service(store, cipher):
    return ApiKeyService(store.uow_factory, cipher)


async def stored_ciphertext(store, user_id, provider):
    async with store.uow_factory() as uow:
        return (await uow.credentials.get(user_id=user_id, provider=provider)).ciphertext


def test_ciphers_follow_the_protocol():
    assert isinstance(FakeCipher(), Cipher) and isinstance(XorCipher(), Cipher)


def test_fake_cipher_roundtrip_and_rejects_foreign_data():
    c = FakeCipher()
    assert c.decrypt(c.encrypt(b"abc")) == b"abc"
    for bad in (b"abc", b"aity-fake-v1:***"):
        with pytest.raises(CipherError):
            c.decrypt(bad)


@pytest.mark.asyncio
async def test_set_get_roundtrip(store, service, cipher):
    alice = await add_user(store)
    info = await service.set_key(user_id=alice.id, provider="openai", api_key=f"  {KEY}\n")
    assert info.provider == "openai"
    assert await service.get_key(user_id=alice.id, provider="openai") == KEY
    ct = await stored_ciphertext(store, alice.id, "openai")
    assert KEY.encode() not in ct
    assert json.loads(cipher.decrypt(ct)) == {"apiKey": KEY, "provider": "openai"}


@pytest.mark.asyncio
async def test_missing_key_is_none(store, service):
    alice = await add_user(store)
    assert await service.get_key(user_id=alice.id, provider="gemini") is None


@pytest.mark.asyncio
async def test_replace_keeps_created_at(store, service):
    alice = await add_user(store)
    first = await service.set_key(user_id=alice.id, provider="anthropic", api_key="old-key")
    second = await service.set_key(user_id=alice.id, provider="anthropic", api_key="new-key")
    assert second.created_at == first.created_at and second.updated_at > first.updated_at
    assert await service.get_key(user_id=alice.id, provider="anthropic") == "new-key"


@pytest.mark.asyncio
async def test_list_shows_providers_only(store, service):
    alice = await add_user(store)
    await service.set_key(user_id=alice.id, provider="openai", api_key=KEY)
    await service.set_key(user_id=alice.id, provider="anthropic", api_key="other-key")
    infos = await service.list_keys(user_id=alice.id)
    assert [i.provider for i in infos] == ["anthropic", "openai"]
    assert KEY not in repr(infos)


@pytest.mark.asyncio
async def test_delete(store, service):
    alice = await add_user(store)
    await service.set_key(user_id=alice.id, provider="gemini", api_key=KEY)
    await service.delete_key(user_id=alice.id, provider="gemini")
    assert await service.get_key(user_id=alice.id, provider="gemini") is None
    with pytest.raises(NotFoundError):
        await service.delete_key(user_id=alice.id, provider="gemini")


@pytest.mark.asyncio
async def test_keys_are_per_user(store, service):
    alice, bob = await add_user(store, "alice"), await add_user(store, "bob")
    await service.set_key(user_id=alice.id, provider="openai", api_key=KEY)
    assert await service.get_key(user_id=bob.id, provider="openai") is None
    assert await service.list_keys(user_id=bob.id) == []
    with pytest.raises(NotFoundError):
        await service.delete_key(user_id=bob.id, provider="openai")


@pytest.mark.asyncio
async def test_unknown_user(service):
    with pytest.raises(NotFoundError):
        await service.set_key(user_id=uuid4(), provider="openai", api_key=KEY)


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["ollama", "mistral", "OpenAI", ""])
async def test_only_cloud_providers(store, service, provider):
    alice = await add_user(store)
    with pytest.raises(UnsupportedKeyProviderError):
        await service.set_key(user_id=alice.id, provider=provider, api_key=KEY)
    with pytest.raises(UnsupportedKeyProviderError):
        await service.get_key(user_id=alice.id, provider=provider)
    with pytest.raises(UnsupportedKeyProviderError):
        await service.delete_key(user_id=alice.id, provider=provider)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad", ["", "   ", "sk with space", "sk-\u00e9", "sk-\x00", "a" * (API_KEY_MAX + 1), None]
)
async def test_invalid_keys(store, service, bad):
    alice = await add_user(store)
    with pytest.raises(InvalidApiKeyError) as info:
        await service.set_key(user_id=alice.id, provider="openai", api_key=bad)
    if isinstance(bad, str) and bad.strip():
        assert bad.strip() not in str(info.value)
    assert await service.list_keys(user_id=alice.id) == []


@pytest.mark.asyncio
async def test_ciphertext_moved_to_another_provider_is_refused(store, service):
    alice = await add_user(store)
    await service.set_key(user_id=alice.id, provider="openai", api_key=KEY)
    ct = await stored_ciphertext(store, alice.id, "openai")
    async with store.uow_factory() as uow:
        await uow.credentials.upsert(user_id=alice.id, provider="anthropic", ciphertext=ct)
        await uow.commit()
    with pytest.raises(StoredKeyUnreadableError) as info:
        await service.get_key(user_id=alice.id, provider="anthropic")
    assert KEY not in str(info.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [b"not json", b"[]", b'{"provider":"openai"}',
                                     b'{"apiKey":"","provider":"openai"}', b"\xff\xfe"])
async def test_unreadable_payloads(store, service, cipher, payload):
    alice = await add_user(store)
    async with store.uow_factory() as uow:
        await uow.credentials.upsert(user_id=alice.id, provider="openai", ciphertext=cipher.encrypt(payload))
        await uow.commit()
    with pytest.raises(StoredKeyUnreadableError):
        await service.get_key(user_id=alice.id, provider="openai")


@pytest.mark.asyncio
async def test_ciphertext_from_another_cipher_is_refused(store):
    alice = await add_user(store)
    await ApiKeyService(store.uow_factory, XorCipher()).set_key(
        user_id=alice.id, provider="gemini", api_key=KEY
    )
    with pytest.raises(StoredKeyUnreadableError):
        await ApiKeyService(store.uow_factory, FakeCipher()).get_key(user_id=alice.id, provider="gemini")


def test_repr_has_no_secrets(store):
    assert "FakeCipher" in repr(ApiKeyService(store.uow_factory, FakeCipher()))
