"""FernetCipher and the fail-fast secret file loaders (services/crypto.py)."""

from __future__ import annotations

import base64

import pytest
from cryptography.fernet import Fernet

from aity_web.services import crypto
from aity_web.services.crypto import (
    JWT_SECRET_FILE_ENV,
    JWT_SECRET_MIN_BYTES,
    MASTER_KEY_FILE_ENV,
    SECRET_FILE_MAX_BYTES,
    Cipher,
    CipherError,
    FernetCipher,
    SecretFileError,
    load_jwt_secret,
    load_master_key,
    load_secrets,
)

JWT = b"j" * JWT_SECRET_MIN_BYTES


@pytest.fixture(autouse=True)
def _no_env_overrides(monkeypatch):
    monkeypatch.delenv(MASTER_KEY_FILE_ENV, raising=False)
    monkeypatch.delenv(JWT_SECRET_FILE_ENV, raising=False)


@pytest.fixture
def key():
    return Fernet.generate_key()


def write(path, data: bytes):
    path.write_bytes(data)
    return path


# --- FernetCipher ---


def test_follows_the_cipher_protocol(key):
    assert isinstance(FernetCipher(key), Cipher)


def test_roundtrip_and_fresh_tokens(key):
    c = FernetCipher(key)
    t1, t2 = c.encrypt(b'{"apiKey":"sk-1"}'), c.encrypt(b'{"apiKey":"sk-1"}')
    assert c.decrypt(t1) == c.decrypt(t2) == b'{"apiKey":"sk-1"}'
    assert t1 != t2  # random IV per token
    assert b"sk-1" not in t1


def test_token_from_another_key_is_refused(key):
    token = FernetCipher(Fernet.generate_key()).encrypt(b"abc")
    with pytest.raises(CipherError):
        FernetCipher(key).decrypt(token)


def test_tampered_token_is_refused(key):
    c = FernetCipher(key)
    raw = bytearray(base64.urlsafe_b64decode(c.encrypt(b"abc")))
    raw[-40] ^= 0x01  # flip one ciphertext bit; the HMAC no longer matches
    with pytest.raises(CipherError):
        c.decrypt(base64.urlsafe_b64encode(bytes(raw)))


@pytest.mark.parametrize("bad", [b"", b"abc", b"aity-fake-v1:YWJj", b"\xff" * 80, None])
def test_garbage_is_refused(key, bad):
    with pytest.raises(CipherError):
        FernetCipher(key).decrypt(bad)


@pytest.mark.parametrize("bad", [b"", b"short", b"A" * 44])
def test_constructor_rejects_bad_keys(bad):
    with pytest.raises(ValueError):
        FernetCipher(bad)


def test_repr_has_no_key(key):
    assert repr(FernetCipher(key)) == "FernetCipher()"


# --- load_master_key ---


def test_master_key_from_path_strips_whitespace(tmp_path, key):
    path = write(tmp_path / "master.key", b"  " + key + b"\r\n")
    assert load_master_key(path) == key
    FernetCipher(load_master_key(path))  # usable as-is


def test_master_key_env_override(tmp_path, monkeypatch, key):
    monkeypatch.setenv(MASTER_KEY_FILE_ENV, str(write(tmp_path / "k", key)))
    assert load_master_key() == key


def test_master_key_argument_wins_over_env(tmp_path, monkeypatch, key):
    monkeypatch.setenv(MASTER_KEY_FILE_ENV, str(tmp_path / "missing"))
    assert load_master_key(write(tmp_path / "k", key)) == key


def test_master_key_default_path(tmp_path, monkeypatch, key):
    monkeypatch.setattr(crypto, "DEFAULT_MASTER_KEY_FILE", write(tmp_path / "default", key))
    monkeypatch.setenv(MASTER_KEY_FILE_ENV, "")  # empty counts as unset
    assert load_master_key() == key


def test_missing_master_key_names_the_path(tmp_path):
    path = tmp_path / "nope" / "master.key"
    with pytest.raises(SecretFileError, match="cannot be read") as info:
        load_master_key(path)
    assert str(path) in str(info.value)


def test_directory_is_not_a_key(tmp_path):
    with pytest.raises(SecretFileError, match="cannot be read"):
        load_master_key(tmp_path)


@pytest.mark.parametrize(
    "content",
    [
        b"",
        b" \n\t",
        b"not-a-key",
        base64.urlsafe_b64encode(b"x" * 31),  # wrong length
        base64.urlsafe_b64encode(b"x" * 33),
        base64.urlsafe_b64encode(b"x" * 32).rstrip(b"="),  # missing padding
        b"!" + base64.urlsafe_b64encode(b"x" * 32)[1:],  # character outside the alphabet
    ],
)
def test_malformed_master_key_is_refused_without_leaking(tmp_path, content):
    path = write(tmp_path / "master.key", content)
    with pytest.raises(SecretFileError) as info:
        load_master_key(path)
    if content.strip():
        assert content.strip().decode("latin-1") not in str(info.value)


def test_oversized_secret_file_is_refused(tmp_path):
    path = write(tmp_path / "master.key", b"A" * (SECRET_FILE_MAX_BYTES + 1))
    with pytest.raises(SecretFileError, match="larger than"):
        load_master_key(path)


# --- load_jwt_secret ---


def test_jwt_secret_from_path(tmp_path):
    assert load_jwt_secret(write(tmp_path / "jwt", JWT + b"\n")) == JWT


def test_jwt_secret_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv(JWT_SECRET_FILE_ENV, str(write(tmp_path / "jwt", JWT)))
    assert load_jwt_secret() == JWT


def test_jwt_secret_default_path(tmp_path, monkeypatch):
    monkeypatch.setattr(crypto, "DEFAULT_JWT_SECRET_FILE", write(tmp_path / "default", JWT))
    assert load_jwt_secret() == JWT


def test_short_jwt_secret_is_refused_without_leaking(tmp_path):
    short = b"s" * (JWT_SECRET_MIN_BYTES - 1)
    path = write(tmp_path / "jwt", short + b"   \n")  # whitespace does not count
    with pytest.raises(SecretFileError, match="shorter than 32 bytes") as info:
        load_jwt_secret(path)
    assert short.decode() not in str(info.value)


@pytest.mark.parametrize("content", [None, b"", b"\n"])
def test_missing_or_empty_jwt_secret(tmp_path, content):
    path = tmp_path / "jwt"
    if content is not None:
        write(path, content)
    with pytest.raises(SecretFileError):
        load_jwt_secret(path)


# --- load_secrets ---


def test_load_secrets(tmp_path, key):
    secrets = load_secrets(write(tmp_path / "k", key), write(tmp_path / "j", JWT))
    assert (secrets.master_key, secrets.jwt_secret) == (key, JWT)
    assert key.decode() not in repr(secrets) and JWT.decode() not in repr(secrets)


def test_load_secrets_fails_if_either_is_missing(tmp_path, key):
    with pytest.raises(SecretFileError, match="JWT secret"):
        load_secrets(write(tmp_path / "k", key), tmp_path / "missing")
    with pytest.raises(SecretFileError, match="master key"):
        load_secrets(tmp_path / "missing", write(tmp_path / "j", JWT))


def test_jwt_secret_must_differ_from_master_key(tmp_path, key):
    with pytest.raises(SecretFileError, match="different"):
        load_secrets(write(tmp_path / "k", key), write(tmp_path / "j", key))
