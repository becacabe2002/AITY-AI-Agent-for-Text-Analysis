from __future__ import annotations

from uuid import UUID, uuid4

from sqlalchemy import delete, insert, select, update

from aity_web.db.tables import documents, sessions, users
from aity_web.models import Session
from aity_web.repositories.base import (
    DEFAULT_PAGE_SIZE,
    UNSET,
    NotFoundError,
    SessionRepository,
    Unset,
)
from aity_web.repositories.sqlalchemy._common import SqlRepo


class SqlAlchemySessionRepository(SqlRepo, SessionRepository):
    async def _owned(self, user_id: UUID, session_id: UUID) -> Session:
        row = (
            await self._execute(
                select(sessions).where(sessions.c.id == session_id, sessions.c.user_id == user_id)
            )
        ).one_or_none()
        if row is None:
            raise NotFoundError(f"session {session_id} not found")
        return Session.model_validate(row)

    async def _save(self, new: Session) -> Session:
        await self._execute(
            update(sessions)
            .where(sessions.c.id == new.id, sessions.c.user_id == new.user_id)
            .values(name=new.name, description=new.description, updated_at=new.updated_at)
        )
        return new

    async def add(self, *, user_id: UUID, name: str, description: str | None = None) -> Session:
        now = self._now()
        row = Session(
            id=uuid4(), user_id=user_id, name=name, description=description,
            created_at=now, updated_at=now,
        )
        exists = (await self._execute(select(users.c.id).where(users.c.id == user_id))).first()
        if exists is None:
            raise NotFoundError(f"user {user_id} not found")
        # a user deleted by a concurrent transaction -> ConflictError
        await self._insert(insert(sessions).values(**row.model_dump()), f"user {user_id} is gone")
        return row

    async def get(self, *, user_id: UUID, session_id: UUID) -> Session:
        return await self._owned(user_id, session_id)

    async def list(
        self, *, user_id: UUID, limit: int = DEFAULT_PAGE_SIZE, offset: int = 0
    ) -> list[Session]:
        if limit < 1 or offset < 0:
            raise ValueError("limit must be >= 1 and offset >= 0")
        result = await self._execute(
            select(sessions)
            .where(sessions.c.user_id == user_id)
            .order_by(sessions.c.updated_at.desc(), sessions.c.id)
            .limit(limit)
            .offset(offset)
        )
        return [Session.model_validate(r) for r in result]

    async def update(
        self,
        *,
        user_id: UUID,
        session_id: UUID,
        name: str | Unset = UNSET,
        description: str | None | Unset = UNSET,
    ) -> Session:
        row = await self._owned(user_id, session_id)
        data = row.model_dump()
        if name is not UNSET:
            data["name"] = name
        if description is not UNSET:
            data["description"] = description
        data["updated_at"] = self._now()
        return await self._save(Session.model_validate(data))  # re-validates name/description

    async def touch(self, *, user_id: UUID, session_id: UUID) -> Session:
        row = await self._owned(user_id, session_id)
        return await self._save(Session.model_validate({**row.model_dump(), "updated_at": self._now()}))

    async def delete(self, *, user_id: UUID, session_id: UUID) -> list[str]:
        await self._owned(user_id, session_id)
        paths = await self._execute(
            select(documents.c.text_path)
            .where(documents.c.session_id == session_id, documents.c.user_id == user_id)
            .order_by(documents.c.uploaded_at, documents.c.id)
        )
        text_paths = list(paths.scalars())
        # documents and analyses go by ON DELETE CASCADE
        await self._execute(
            delete(sessions).where(sessions.c.id == session_id, sessions.c.user_id == user_id)
        )
        return text_paths
