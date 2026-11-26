"""Service for uploading, listing, reading and deleting documents."""

from __future__ import annotations

import asyncio
import logging
import unicodedata
from collections.abc import Callable
from pathlib import PureWindowsPath
from uuid import UUID, uuid4

from aity_web.models import Document
from aity_web.models.document import FILENAME_MAX
from aity_web.services._common import Clock, UowFactory, utc_now
from aity_web.services.text_extraction import UnsupportedFileTypeError, extract
from aity_web.services.upload_storage import UploadStorage, text_path_for, upload_timestamp

log = logging.getLogger(__name__)


def clean_filename(filename: str) -> str:
    """Return a safe display name for an uploaded file; it is never used to build a path."""
    name = PureWindowsPath(filename or "").name  # splits on both / and \
    name = "".join(ch for ch in name if unicodedata.category(ch)[0] != "C").strip()
    if not name:
        raise UnsupportedFileTypeError("The file has no name.")
    if len(name) > FILENAME_MAX:
        suffix = PureWindowsPath(name).suffix[:16]
        name = name[: FILENAME_MAX - len(suffix)] + suffix
    return name


class DocumentService:
    """Document operations: upload, list, read and delete."""

    def __init__(
        self,
        uow_factory: UowFactory,
        storage: UploadStorage,
        *,
        clock: Clock = utc_now,
        new_id: Callable[[], UUID] = uuid4,
    ) -> None:
        """Create the service with its unit-of-work factory, storage, clock and id source."""
        self._uow_factory = uow_factory
        self._storage = storage
        self._clock = clock
        self._new_id = new_id

    async def upload(
        self, *, user_id: UUID, session_id: UUID, filename: str, data: bytes
    ) -> Document:
        """Store an uploaded PDF, DOCX or TXT in a session and return the new document."""
        original = clean_filename(filename)
        extracted = await asyncio.to_thread(extract, original, data)

        document_id = self._new_id()
        uploaded_at = upload_timestamp(self._clock())
        text_path = text_path_for(
            user_id=user_id, session_id=session_id,
            document_id=document_id, uploaded_at=uploaded_at,
        )
        document = Document(
            id=document_id,
            session_id=session_id,
            user_id=user_id,
            original_filename=original,
            source_type=extracted.source_type,
            text_path=text_path,
            text_sha256=extracted.text_sha256,
            char_count=extracted.char_count,
            uploaded_at=uploaded_at,
        )

        async with self._uow_factory() as uow:
            await uow.sessions.get(user_id=user_id, session_id=session_id)
            await asyncio.to_thread(self._storage.write, text_path, extracted.stored_bytes)
            try:
                await uow.documents.add(user_id=user_id, document=document)
                await uow.sessions.touch(user_id=user_id, session_id=session_id)
                await uow.commit()
            except BaseException:
                self._storage.delete(text_path)
                raise
        return document

    async def list(self, *, user_id: UUID, session_id: UUID) -> list[Document]:
        """List a session's documents, oldest upload first."""
        async with self._uow_factory() as uow:
            return await uow.documents.list_for_session(user_id=user_id, session_id=session_id)

    async def get(self, *, user_id: UUID, document_id: UUID) -> Document:
        """Return one of the user's documents."""
        async with self._uow_factory() as uow:
            return await uow.documents.get(user_id=user_id, document_id=document_id)

    async def delete(self, *, user_id: UUID, document_id: UUID) -> None:
        """Delete the document, its analyses and its text file."""
        async with self._uow_factory() as uow:
            document = await uow.documents.delete(user_id=user_id, document_id=document_id)
            await uow.commit()
        if not await asyncio.to_thread(self._storage.delete, document.text_path):
            log.warning("document %s deleted; its text file is left on disk", document_id)


__all__ = ["DocumentService", "clean_filename"]
