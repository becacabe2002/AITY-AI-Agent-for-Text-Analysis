"""SqlAlchemyUnitOfWork: one AsyncSession = one unit of work (ADR-004 §7.3).

    uow_factory = make_uow_factory(make_session_factory(make_engine()))
    async with uow_factory() as uow:
        s = await uow.sessions.add(user_id=uid, name="SMS pilot")
        await uow.commit()

The session is opened on enter and closed on exit; anything not committed is
rolled back. After commit() the same unit of work starts a new transaction
on its next statement (SQLAlchemy autobegin).

Timestamps the repositories set (users, sessions) come from ``clock``, not
from the database, so tests can control them and the in-memory fakes match.
"""

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
    return datetime.now(timezone.utc)


class _CredentialsNotYetAvailable(CredentialRepository):
    """The user_api_keys table and its repository arrive in W4 (ADR-005).
    Until then every call fails loudly instead of silently storing nothing."""

    def _fail(self):
        raise NotImplementedError(
            "CredentialRepository has no PostgreSQL implementation until W4 (ADR-005)"
        )

    async def upsert(self, *, user_id, provider: CloudProviderName, ciphertext: bytes) -> UserApiKey:
        self._fail()

    async def get(self, *, user_id, provider: CloudProviderName) -> UserApiKey | None:
        self._fail()

    async def list(self, *, user_id) -> list[UserApiKey]:
        self._fail()

    async def delete(self, *, user_id, provider: CloudProviderName) -> None:
        self._fail()


class SqlAlchemyUnitOfWork(UnitOfWork):
    def __init__(
        self, session_factory: async_sessionmaker[AsyncSession], *, clock: Clock = utc_now
    ) -> None:
        self._session_factory = session_factory
        self.clock = clock
        self._session: AsyncSession | None = None
        self.users = SqlAlchemyUserRepository(self)
        self.sessions = SqlAlchemySessionRepository(self)
        self.documents = SqlAlchemyDocumentRepository(self)
        self.analyses = SqlAlchemyAnalysisRepository(self)
        self.credentials = _CredentialsNotYetAvailable()

    def _active_session(self) -> AsyncSession:
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
        await self._active_session().commit()

    async def rollback(self) -> None:
        await self._active_session().rollback()


def make_uow_factory(
    session_factory: async_sessionmaker[AsyncSession], *, clock: Clock = utc_now
) -> Callable[[], SqlAlchemyUnitOfWork]:
    """What services receive: a zero-argument callable returning a new unit of work."""

    def factory() -> SqlAlchemyUnitOfWork:
        return SqlAlchemyUnitOfWork(session_factory, clock=clock)

    return factory
