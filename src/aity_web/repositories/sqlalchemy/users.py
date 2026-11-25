from __future__ import annotations

from uuid import UUID, uuid4

from sqlalchemy import insert, select

from aity_web.db.tables import users
from aity_web.models import User
from aity_web.repositories.base import NotFoundError, UserRepository
from aity_web.repositories.sqlalchemy._common import SqlRepo


class SqlAlchemyUserRepository(SqlRepo, UserRepository):
    async def add(self, *, username: str, password_hash: str) -> User:
        # Validate first: a bad username is a ValueError, never a CHECK failure.
        user = User(id=uuid4(), username=username, password_hash=password_hash, created_at=self._now())
        await self._insert(
            insert(users).values(**user.model_dump()), f"username {username!r} is taken"
        )
        return user

    async def get(self, *, user_id: UUID) -> User:
        row = (await self._execute(select(users).where(users.c.id == user_id))).one_or_none()
        if row is None:
            raise NotFoundError(f"user {user_id} not found")
        return User.model_validate(row)

    async def get_by_username(self, *, username: str) -> User | None:
        row = (
            await self._execute(select(users).where(users.c.username == username))
        ).one_or_none()
        return None if row is None else User.model_validate(row)
