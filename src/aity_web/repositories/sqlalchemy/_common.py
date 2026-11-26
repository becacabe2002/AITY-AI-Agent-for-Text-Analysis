"""Shared helpers for the SQLAlchemy repositories."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import Executable, Insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from aity_web.repositories.base import ConflictError

if TYPE_CHECKING:
    from aity_web.db.uow import SqlAlchemyUnitOfWork

Clock = Callable[[], datetime]

# PostgreSQL SQLSTATEs that mean "the write conflicts with another row"
_UNIQUE_VIOLATION = "23505"
_FOREIGN_KEY_VIOLATION = "23503"


def sqlstate(exc: IntegrityError) -> str | None:
    """Return the PostgreSQL SQLSTATE code of an integrity error, if any."""
    orig = exc.orig
    return getattr(orig, "sqlstate", None) or getattr(orig, "pgcode", None)


class SqlRepo:
    """Base class of the SQLAlchemy repositories; the unit of work owns the session and clock."""

    def __init__(self, uow: SqlAlchemyUnitOfWork) -> None:
        """Bind the repository to its unit of work."""
        self._uow = uow

    @property
    def _db(self) -> AsyncSession:
        """The unit of work's active session."""
        return self._uow._active_session()

    def _now(self) -> datetime:
        """Return the current time from the unit of work's clock."""
        return self._uow.clock()

    async def _execute(self, statement: Executable) -> Any:
        """Run a statement on the active session."""
        return await self._db.execute(statement)

    async def _insert(self, statement: Insert, conflict: str) -> Any:
        """Run an INSERT in a savepoint; unique and foreign-key violations become ConflictError."""
        try:
            async with self._db.begin_nested():
                return await self._db.execute(statement)
        except IntegrityError as exc:
            if sqlstate(exc) in (_UNIQUE_VIOLATION, _FOREIGN_KEY_VIOLATION):
                raise ConflictError(conflict) from exc
            raise
