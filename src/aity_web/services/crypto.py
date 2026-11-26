"""Fernet cipher for stored API keys and fail-fast loaders for the secret files."""

from __future__ import annotations

import base64
import binascii
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from cryptography.fernet import Fernet, InvalidToken

MASTER_KEY_FILE_ENV = "AITY_MASTER_KEY_FILE"
DEFAULT_MASTER_KEY_FILE = Path("/run/secrets/master.key")
JWT_SECRET_FILE_ENV = "AITY_JWT_SECRET_FILE"
DEFAULT_JWT_SECRET_FILE = Path("/run/secrets/jwt_secret")
JWT_SECRET_MIN_BYTES = 32
SECRET_FILE_MAX_BYTES = 4096


# --- cipher ---


class CipherError(Exception):
    """A ciphertext could not be decrypted (corrupt, tampered, or another master key)."""


@runtime_checkable
class Cipher(Protocol): # only keep for tests
    """Encrypts and decrypts the stored key payload."""

    def encrypt(self, plaintext: bytes) -> bytes:
        """Encrypt bytes."""
        ...

    def decrypt(self, ciphertext: bytes) -> bytes:
        """Decrypt bytes; raise CipherError if they are not valid for this cipher."""
        ...


class FernetCipher:
    """Cipher backed by Fernet from the cryptography library, with one master key."""

    def __init__(self, key: bytes) -> None:
        """Create the cipher; raise ValueError if the key is not a valid Fernet key."""
        self._fernet = Fernet(key)

    def __repr__(self) -> str:
        return "FernetCipher()"

    def encrypt(self, plaintext: bytes) -> bytes:
        """Encrypt bytes into a Fernet token."""
        return self._fernet.encrypt(plaintext)

    def decrypt(self, ciphertext: bytes) -> bytes:
        """Decrypt a Fernet token; raise CipherError if it is invalid for this key."""
        try:
            return self._fernet.decrypt(ciphertext)
        except (InvalidToken, TypeError):
            raise CipherError("the ciphertext is not a valid token for this master key") from None


# --- secret files ---


class SecretFileError(RuntimeError):
    """A secret file is missing, unreadable or malformed; the message never holds its content."""


@dataclass(frozen=True)
class AppSecrets:
    """The secrets the web service needs at startup."""

    master_key: bytes = field(repr=False)
    jwt_secret: bytes = field(repr=False)


def load_master_key(path: str | os.PathLike[str] | None = None) -> bytes:
    """Read and validate the Fernet master key from the argument, the environment or the default path."""
    source = Path(path or os.environ.get(MASTER_KEY_FILE_ENV) or DEFAULT_MASTER_KEY_FILE)
    key = _read_secret_file(source, "master key")
    if not _is_fernet_key(key):
        raise SecretFileError(
            f"master key file {source} does not hold a valid Fernet key "
            "(32 url-safe base64-encoded bytes)"
        )
    return key


def load_jwt_secret(path: str | os.PathLike[str] | None = None) -> bytes:
    """Read and validate the JWT signing secret from the argument, the environment or the default path."""
    source = Path(path or os.environ.get(JWT_SECRET_FILE_ENV) or DEFAULT_JWT_SECRET_FILE)
    secret = _read_secret_file(source, "JWT secret")
    if len(secret) < JWT_SECRET_MIN_BYTES:
        raise SecretFileError(
            f"JWT secret file {source} is shorter than {JWT_SECRET_MIN_BYTES} bytes"
        )
    return secret


def load_secrets(
    master_key_file: str | os.PathLike[str] | None = None,
    jwt_secret_file: str | os.PathLike[str] | None = None,
) -> AppSecrets:
    """Load both startup secrets; raise SecretFileError if either is missing or malformed."""
    secrets = AppSecrets(
        master_key=load_master_key(master_key_file),
        jwt_secret=load_jwt_secret(jwt_secret_file),
    )
    if secrets.master_key == secrets.jwt_secret:
        raise SecretFileError("the JWT secret must be different from the master key")
    return secrets


def _read_secret_file(path: Path, what: str) -> bytes:
    """Read a small secret file and strip surrounding whitespace."""
    try:
        with path.open("rb") as fh:
            data = fh.read(SECRET_FILE_MAX_BYTES + 1)
    except OSError as exc:
        reason = exc.strerror or type(exc).__name__
        raise SecretFileError(f"{what} file {path} cannot be read: {reason}") from None
    if len(data) > SECRET_FILE_MAX_BYTES:
        raise SecretFileError(f"{what} file {path} is larger than {SECRET_FILE_MAX_BYTES} bytes")
    data = data.strip()
    if not data:
        raise SecretFileError(f"{what} file {path} is empty")
    return data


def _is_fernet_key(key: bytes) -> bool:
    """True if the key is strict url-safe base64 of exactly 32 bytes."""
    try:
        raw = base64.b64decode(key, altchars=b"-_", validate=True)
    except (binascii.Error, ValueError):
        return False
    return len(raw) == 32


__all__ = [
    "DEFAULT_JWT_SECRET_FILE",
    "DEFAULT_MASTER_KEY_FILE",
    "JWT_SECRET_FILE_ENV",
    "JWT_SECRET_MIN_BYTES",
    "MASTER_KEY_FILE_ENV",
    "SECRET_FILE_MAX_BYTES",
    "AppSecrets",
    "Cipher",
    "CipherError",
    "FernetCipher",
    "SecretFileError",
    "load_jwt_secret",
    "load_master_key",
    "load_secrets",
]
