"""Shared plumbing for the SQLAlchemy repositories."""

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
    orig = exc.orig
    return getattr(orig, "sqlstate", None) or getattr(orig, "pgcode", None)


class SqlRepo:
    """Base class: the unit of work owns the AsyncSession and the clock."""

    def __init__(self, uow: SqlAlchemyUnitOfWork) -> None:
        self._uow = uow

    @property
    def _db(self) -> AsyncSession:
        return self._uow._active_session()

    def _now(self) -> datetime:
        return self._uow.clock()

    async def _execute(self, statement: Executable) -> Any:
        return await self._db.execute(statement)

    async def _insert(self, statement: Insert, conflict: str) -> Any:
        """Run an INSERT inside a SAVEPOINT, so a refused row does not abort
        the whole transaction. Unique and foreign-key violations become
        ConflictError(conflict); any other integrity error (a CHECK the
        domain model should have caught) is re-raised unchanged."""
        try:
            async with self._db.begin_nested():
                return await self._db.execute(statement)
        except IntegrityError as exc:
            if sqlstate(exc) in (_UNIQUE_VIOLATION, _FOREIGN_KEY_VIOLATION):
                raise ConflictError(conflict) from exc
            raise
