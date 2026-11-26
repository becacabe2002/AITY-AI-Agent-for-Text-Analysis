"""Repository interfaces, errors and the unit-of-work contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Final, Literal
from uuid import UUID

from aity_web.models import (
    AnalysisResult,
    CloudProviderName,
    Document,
    Session,
    User,
    UserApiKey,
)

# --- errors ---


class RepositoryError(Exception):
    """Base class for repository errors."""


class NotFoundError(RepositoryError, LookupError):
    """The row does not exist, or it belongs to another user."""


class ConflictError(RepositoryError):
    """The write breaks a uniqueness or reference rule."""


# --- "leave unchanged" marker for partial updates ---


class _Unset(Enum):
    """Marker type for fields that an update must leave unchanged."""

    UNSET = "UNSET"

    def __repr__(self) -> str:
        return "UNSET"


UNSET: Final = _Unset.UNSET
"""Marker for a field that update() must leave as it is."""

Unset = Literal[_Unset.UNSET]

DEFAULT_PAGE_SIZE = 50


# --- repositories ---


class UserRepository(ABC):
    """Access to user accounts."""

    @abstractmethod
    async def add(self, *, username: str, password_hash: str) -> User:
        """Create a user; ConflictError if the username is taken."""

    @abstractmethod
    async def get(self, *, user_id: UUID) -> User:
        """Return the user; NotFoundError if there is none."""

    @abstractmethod
    async def get_by_username(self, *, username: str) -> User | None:
        """Return the user with this exact username, or None."""


class SessionRepository(ABC):
    """Access to a user's sessions."""

    @abstractmethod
    async def add(self, *, user_id: UUID, name: str, description: str | None = None) -> Session:
        """Create a session for the user."""

    @abstractmethod
    async def get(self, *, user_id: UUID, session_id: UUID) -> Session:
        """Return one of the user's sessions."""

    @abstractmethod
    async def list(
        self, *, user_id: UUID, limit: int = DEFAULT_PAGE_SIZE, offset: int = 0
    ) -> list[Session]:
        """List the user's sessions, most recently updated first."""

    @abstractmethod
    async def update(
        self,
        *,
        user_id: UUID,
        session_id: UUID,
        name: str | Unset = UNSET,
        description: str | None | Unset = UNSET,
    ) -> Session:
        """Change the name and/or description and bump updated_at."""

    @abstractmethod
    async def touch(self, *, user_id: UUID, session_id: UUID) -> Session:
        """Bump a session's updated_at to now."""

    @abstractmethod
    async def delete(self, *, user_id: UUID, session_id: UUID) -> list[str]:
        """Delete a session with its documents and analyses; return the text paths to remove."""


class DocumentRepository(ABC):
    """Access to a user's documents."""

    @abstractmethod
    async def add(self, *, user_id: UUID, document: Document) -> Document:
        """Insert a document built by the service."""

    @abstractmethod
    async def get(self, *, user_id: UUID, document_id: UUID) -> Document:
        """Return one of the user's documents."""

    @abstractmethod
    async def list_for_session(self, *, user_id: UUID, session_id: UUID) -> list[Document]:
        """List a session's documents, oldest upload first."""

    @abstractmethod
    async def delete(self, *, user_id: UUID, document_id: UUID) -> Document:
        """Delete a document and its analyses; return the deleted row."""


class AnalysisRepository(ABC):
    """Access to a user's analysis results."""

    @abstractmethod
    async def add(self, *, user_id: UUID, result: AnalysisResult) -> AnalysisResult:
        """Insert a result built by the service."""

    @abstractmethod
    async def get(self, *, user_id: UUID, analysis_id: UUID) -> AnalysisResult:
        """Return one of the user's results."""

    @abstractmethod
    async def list_for_document(self, *, user_id: UUID, document_id: UUID) -> list[AnalysisResult]:
        """List a document's results, newest first."""

    @abstractmethod
    async def list_for_session(self, *, user_id: UUID, session_id: UUID) -> list[AnalysisResult]:
        """List a session's results, newest first."""

    @abstractmethod
    async def delete(self, *, user_id: UUID, analysis_id: UUID) -> None:
        """Delete one result; copies served from it lose their link to it."""


class CredentialRepository(ABC):
    """Access to encrypted API keys (ciphertext only)."""

    @abstractmethod
    async def upsert(
        self, *, user_id: UUID, provider: CloudProviderName, ciphertext: bytes
    ) -> UserApiKey:
        """Insert or replace the stored ciphertext for a provider."""

    @abstractmethod
    async def get(self, *, user_id: UUID, provider: CloudProviderName) -> UserApiKey | None:
        """Return the stored key for a provider, or None."""

    @abstractmethod
    async def list(self, *, user_id: UUID) -> list[UserApiKey]:
        """List the user's stored keys, ordered by provider."""

    @abstractmethod
    async def delete(self, *, user_id: UUID, provider: CloudProviderName) -> None:
        """Delete the stored key for a provider."""


# --- unit of work ---


class UnitOfWork(ABC):
    """One transaction over all repositories, used as an async context manager."""

    users: UserRepository
    sessions: SessionRepository
    documents: DocumentRepository
    analyses: AnalysisRepository
    credentials: CredentialRepository

    @abstractmethod
    async def __aenter__(self) -> UnitOfWork: ...

    @abstractmethod
    async def __aexit__(self, exc_type, exc, tb) -> None:
        """Roll back anything not committed without swallowing the exception."""

    @abstractmethod
    async def commit(self) -> None:
        """Commit everything written in this unit of work."""

    @abstractmethod
    async def rollback(self) -> None:
        """Discard everything written since the last commit."""
