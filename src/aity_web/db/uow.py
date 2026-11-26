"""SQLAlchemy unit of work: one AsyncSession is one transaction."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from aity_web.models import CloudProviderName, UserApiKey
from aity_web.repositories.base import CredentialRepository, UnitOfWork
from aity_web.repositories.sqlalchemy import (
    SqlAlchemyAnalysisRepository,
    SqlAlchemyDocumentRepository,
    SqlAlchemySessionRepository,
    SqlAlchemyUserRepository,
)

Clock = Callable[[], datetime]


def utc_now() -> datetime:
    """Return the current time in UTC."""
    return datetime.now(timezone.utc)


class _CredentialsNotYetAvailable(CredentialRepository):
    """Placeholder credential repository that fails loudly until the key table exists."""

    def _fail(self):
        """Raise NotImplementedError for every credential operation."""
        raise NotImplementedError(
            "CredentialRepository has no PostgreSQL implementation until W4 (ADR-005)"
        )

    async def upsert(self, *, user_id, provider: CloudProviderName, ciphertext: bytes) -> UserApiKey:
        """Not available yet; always fails."""
        self._fail()

    async def get(self, *, user_id, provider: CloudProviderName) -> UserApiKey | None:
        """Not available yet; always fails."""
        self._fail()

    async def list(self, *, user_id) -> list[UserApiKey]:
        """Not available yet; always fails."""
        self._fail()

    async def delete(self, *, user_id, provider: CloudProviderName) -> None:
        """Not available yet; always fails."""
        self._fail()


class SqlAlchemyUnitOfWork(UnitOfWork):
    """Unit of work over one AsyncSession: explicit commit, rollback on exit."""

    def __init__(
        self, session_factory: async_sessionmaker[AsyncSession], *, clock: Clock = utc_now
    ) -> None:
        """Create the unit of work and its repositories; the session opens on entering the block."""
        self._session_factory = session_factory
        self.clock = clock
        self._session: AsyncSession | None = None
        self.users = SqlAlchemyUserRepository(self)
        self.sessions = SqlAlchemySessionRepository(self)
        self.documents = SqlAlchemyDocumentRepository(self)
        self.analyses = SqlAlchemyAnalysisRepository(self)
        self.credentials = _CredentialsNotYetAvailable()

    def _active_session(self) -> AsyncSession:
        """Return the open session, or raise if used outside the async with block."""
        if self._session is None:
            raise RuntimeError("repositories are only usable inside 'async with uow:'")
        return self._session

    async def __aenter__(self) -> SqlAlchemyUnitOfWork:
        if self._session is not None:
            raise RuntimeError("unit of work is already open")
        self._session = self._session_factory()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        session, self._session = self._session, None
        if session is None:
            return
        try:
            await session.rollback()  # no-op after commit(); drops anything uncommitted
        finally:
            await session.close()

    async def commit(self) -> None:
        """Commit the current transaction."""
        await self._active_session().commit()

    async def rollback(self) -> None:
        """Roll back the current transaction."""
        await self._active_session().rollback()


def make_uow_factory(
    session_factory: async_sessionmaker[AsyncSession], *, clock: Clock = utc_now
) -> Callable[[], SqlAlchemyUnitOfWork]:
    """Return the zero-argument factory that services use to get a new unit of work."""

    def factory() -> SqlAlchemyUnitOfWork:
        """Build one unit of work."""
        return SqlAlchemyUnitOfWork(session_factory, clock=clock)

    return factory
