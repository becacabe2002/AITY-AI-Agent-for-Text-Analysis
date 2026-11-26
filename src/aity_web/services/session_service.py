"""Sessions: create, list, read, rename/describe, delete - ADR-004 §7.7, §7.8.

A session only groups a user's documents; it holds no LLM configuration.
Every call takes the caller's user_id; another user's session raises
NotFoundError exactly like a missing one (the W5 routes return 404).

Delete order: collect the text paths, delete the row (the database cascades
to documents and analyses), commit, then remove the text files and the empty
session folder. File removal runs after the commit and never raises.
"""

from __future__ import annotations

import asyncio
import logging
from uuid import UUID

from aity_web.models import Session
from aity_web.repositories.base import DEFAULT_PAGE_SIZE, UNSET, Unset
from aity_web.services._common import UowFactory
from aity_web.services.upload_storage import UploadStorage

log = logging.getLogger(__name__)


class SessionService:
    def __init__(self, uow_factory: UowFactory, storage: UploadStorage) -> None:
        self._uow_factory = uow_factory
        self._storage = storage

    async def create(
        self, *, user_id: UUID, name: str, description: str | None = None
    ) -> Session:
        """Name 1-200 characters (stripped), description up to 2000 (blank -> None);
        pydantic.ValidationError otherwise. NotFoundError if the user is unknown."""
        async with self._uow_factory() as uow:
            session = await uow.sessions.add(user_id=user_id, name=name, description=description)
            await uow.commit()
        return session

    async def list(
        self, *, user_id: UUID, limit: int = DEFAULT_PAGE_SIZE, offset: int = 0
    ) -> list[Session]:
        """Most recently updated first."""
        async with self._uow_factory() as uow:
            return await uow.sessions.list(user_id=user_id, limit=limit, offset=offset)

    async def get(self, *, user_id: UUID, session_id: UUID) -> Session:
        async with self._uow_factory() as uow:
            return await uow.sessions.get(user_id=user_id, session_id=session_id)

    async def update(
        self,
        *,
        user_id: UUID,
        session_id: UUID,
        name: str | Unset = UNSET,
        description: str | None | Unset = UNSET,
    ) -> Session:
        """Edit the name and/or description. Fields left UNSET keep their value;
        description=None (or blank) clears it. Calling with neither field is a
        no-op that returns the session unchanged (updated_at not bumped)."""
        async with self._uow_factory() as uow:
            if name is UNSET and description is UNSET:
                return await uow.sessions.get(user_id=user_id, session_id=session_id)
            session = await uow.sessions.update(
                user_id=user_id, session_id=session_id, name=name, description=description
            )
            await uow.commit()
        return session

    async def delete(self, *, user_id: UUID, session_id: UUID) -> None:
        """Delete the session, its documents, their text files and their analyses."""
        async with self._uow_factory() as uow:
            text_paths = await uow.sessions.delete(user_id=user_id, session_id=session_id)
            await uow.commit()
        await asyncio.to_thread(self._remove_files, user_id, session_id, text_paths)

    def _remove_files(self, user_id: UUID, session_id: UUID, text_paths: list[str]) -> None:
        failed = sum(not self._storage.delete(p) for p in text_paths)
        self._storage.remove_session_dir(user_id=user_id, session_id=session_id)
        if failed:
            log.warning("session %s deleted; %d text file(s) left on disk", session_id, failed)


__all__ = ["SessionService"]
