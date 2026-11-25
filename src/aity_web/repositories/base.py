"""Repository interfaces and the unit of work.

Rules every implementation follows:

* Every method of a user-owned row takes a keyword-only ``user_id`` and
  filters by it. A row that does not exist and a row owned by another user
  both raise NotFoundError, so callers cannot tell them apart.
  UserRepository is the only exception (it looks users up).
* Repositories return domain models (aity_web.models), never ORM objects.
* Repositories never commit. The service opens a UnitOfWork, calls the
  repositories and commits; leaving the block without commit() rolls back.
* Ids and timestamps of rows created by a repository (users, sessions,
  credentials) come from the repository and its clock. Documents and
  analysis results arrive fully built by the service.
* Deleting is cascading:
  session -> its documents and analyses; document -> its analyses;
  deleting an analysis clears served_from_analysis_id on rows copied from it.
  Files are not touched here: the service removes them after the commit.

Not in W3: find_cached() (W6, ADR-008), comparisons and syntheses (W8).
"""

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
    """The write breaks a uniqueness or reference rule (duplicate username,
    duplicate id, a parent deleted by a concurrent transaction)."""


# --- "leave unchanged" marker for partial updates ---


class _Unset(Enum):
    UNSET = "UNSET"

    def __repr__(self) -> str:
        return "UNSET"


UNSET: Final = _Unset.UNSET
"""Passed (or defaulted) for a field that update() must leave as it is.
None is a real value: description=None clears the description."""

Unset = Literal[_Unset.UNSET]

DEFAULT_PAGE_SIZE = 50


# --- repositories ---


class UserRepository(ABC):
    """Minimal in W3; registration and login arrive in W4."""

    @abstractmethod
    async def add(self, *, username: str, password_hash: str) -> User:
        """Create a user. ConflictError if the username is taken
        (usernames are case-sensitive)."""

    @abstractmethod
    async def get(self, *, user_id: UUID) -> User:
        """NotFoundError if there is no such user."""

    @abstractmethod
    async def get_by_username(self, *, username: str) -> User | None:
        """Exact, case-sensitive match; None if there is none."""


class SessionRepository(ABC):
    @abstractmethod
    async def add(self, *, user_id: UUID, name: str, description: str | None = None) -> Session:
        """Create a session; created_at = updated_at = now.
        NotFoundError if the user does not exist."""

    @abstractmethod
    async def get(self, *, user_id: UUID, session_id: UUID) -> Session:
        """NotFoundError if missing or not the user's."""

    @abstractmethod
    async def list(
        self, *, user_id: UUID, limit: int = DEFAULT_PAGE_SIZE, offset: int = 0
    ) -> list[Session]:
        """The user's sessions, most recently updated first (ties: id).
        limit >= 1 and offset >= 0, else ValueError."""

    @abstractmethod
    async def update(
        self,
        *,
        user_id: UUID,
        session_id: UUID,
        name: str | Unset = UNSET,
        description: str | None | Unset = UNSET,
    ) -> Session:
        """Change the name and/or description and bump updated_at.
        Fields left UNSET keep their value; description=None clears it.
        NotFoundError if missing or not the user's."""

    @abstractmethod
    async def touch(self, *, user_id: UUID, session_id: UUID) -> Session:
        """Bump updated_at to now (a document or analysis was added).
        NotFoundError if missing or not the user's."""

    @abstractmethod
    async def delete(self, *, user_id: UUID, session_id: UUID) -> list[str]:
        """Delete the session with its documents and analyses.
        Returns the deleted documents' text_paths for the service to remove
        after the commit. NotFoundError if missing or not the user's."""


class DocumentRepository(ABC):
    @abstractmethod
    async def add(self, *, user_id: UUID, document: Document) -> Document:
        """Insert a document built by the service.
        ValueError if document.user_id != user_id (a programming error).
        NotFoundError if its session is missing or not the user's.
        ConflictError if the id is already used."""

    @abstractmethod
    async def get(self, *, user_id: UUID, document_id: UUID) -> Document:
        """NotFoundError if missing or not the user's."""

    @abstractmethod
    async def list_for_session(self, *, user_id: UUID, session_id: UUID) -> list[Document]:
        """The session's documents, oldest upload first (ties: id).
        NotFoundError if the session is missing or not the user's."""

    @abstractmethod
    async def delete(self, *, user_id: UUID, document_id: UUID) -> Document:
        """Delete the document and its analyses; returns the deleted row so
        the service can remove its text file after the commit.
        NotFoundError if missing or not the user's."""


class AnalysisRepository(ABC):
    @abstractmethod
    async def add(self, *, user_id: UUID, result: AnalysisResult) -> AnalysisResult:
        """Insert a result built by the service.
        ValueError if result.user_id != user_id or result.session_id is not
        the document's session (programming errors).
        NotFoundError if the document, or the row named in
        served_from_analysis_id, is missing or not the user's.
        ConflictError if the analysis_id is already used."""

    @abstractmethod
    async def get(self, *, user_id: UUID, analysis_id: UUID) -> AnalysisResult:
        """NotFoundError if missing or not the user's."""

    @abstractmethod
    async def list_for_document(self, *, user_id: UUID, document_id: UUID) -> list[AnalysisResult]:
        """Newest first (ties: analysis_id). NotFoundError if the document
        is missing or not the user's."""

    @abstractmethod
    async def list_for_session(self, *, user_id: UUID, session_id: UUID) -> list[AnalysisResult]:
        """Newest first (ties: analysis_id). NotFoundError if the session
        is missing or not the user's."""

    @abstractmethod
    async def delete(self, *, user_id: UUID, analysis_id: UUID) -> None:
        """Delete one result. Rows copied from it keep their own payload and
        sustainability; their served_from_analysis_id becomes None.
        NotFoundError if missing or not the user's."""


class CredentialRepository(ABC):
    """Ciphertext only: ApiKeyService encrypts and decrypts (fake cipher in
    W3, Fernet in W4). One row per (user_id, provider)."""

    @abstractmethod
    async def upsert(
        self, *, user_id: UUID, provider: CloudProviderName, ciphertext: bytes
    ) -> UserApiKey:
        """Insert, or replace the ciphertext of the existing row (created_at
        kept, updated_at bumped). NotFoundError if the user does not exist."""

    @abstractmethod
    async def get(self, *, user_id: UUID, provider: CloudProviderName) -> UserApiKey | None:
        """None if the user has no key for this provider."""

    @abstractmethod
    async def list(self, *, user_id: UUID) -> list[UserApiKey]:
        """The user's keys, ordered by provider name."""

    @abstractmethod
    async def delete(self, *, user_id: UUID, provider: CloudProviderName) -> None:
        """NotFoundError if the user has no key for this provider."""


# --- unit of work ---


class UnitOfWork(ABC):
    """One transaction: one AsyncSession in the SQLAlchemy implementation.

        async with uow_factory() as uow:
            session = await uow.sessions.add(user_id=uid, name="SMS pilot")
            await uow.commit()

    Leaving the block without commit() - normally or by an exception -
    rolls back everything written since the last commit. After commit()
    the unit of work can keep being used for a further transaction.
    The repositories are only usable inside the ``async with`` block.
    """

    users: UserRepository
    sessions: SessionRepository
    documents: DocumentRepository
    analyses: AnalysisRepository
    credentials: CredentialRepository

    @abstractmethod
    async def __aenter__(self) -> UnitOfWork: ...

    @abstractmethod
    async def __aexit__(self, exc_type, exc, tb) -> None:
        """Roll back anything not committed; never swallows the exception."""

    @abstractmethod
    async def commit(self) -> None: ...

    @abstractmethod
    async def rollback(self) -> None:
        """Discard everything written since the last commit."""
