"""PostgreSQL implementation of the document repository."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import delete, insert, select

from aity_web.db.tables import documents, sessions
from aity_web.models import Document
from aity_web.repositories.base import DocumentRepository, NotFoundError
from aity_web.repositories.sqlalchemy._common import SqlRepo


class SqlAlchemyDocumentRepository(SqlRepo, DocumentRepository):
    """SQLAlchemy implementation of DocumentRepository."""

    async def _require_session(self, user_id: UUID, session_id: UUID) -> None:
        """Raise NotFoundError unless the session exists and belongs to the user."""
        found = (
            await self._execute(
                select(sessions.c.id).where(sessions.c.id == session_id, sessions.c.user_id == user_id)
            )
        ).first()
        if found is None:
            raise NotFoundError(f"session {session_id} not found")

    async def _owned(self, user_id: UUID, document_id: UUID) -> Document:
        """Return the user's document or raise NotFoundError."""
        row = (
            await self._execute(
                select(documents).where(documents.c.id == document_id, documents.c.user_id == user_id)
            )
        ).one_or_none()
        if row is None:
            raise NotFoundError(f"document {document_id} not found")
        return Document.model_validate(row)

    async def add(self, *, user_id: UUID, document: Document) -> Document:
        """Insert a document built by the service."""
        if document.user_id != user_id:
            raise ValueError("document.user_id does not match user_id")
        await self._require_session(user_id, document.session_id)
        await self._insert(
            insert(documents).values(**document.model_dump()),
            f"document id {document.id} or its text_path already exists, or its session is gone",
        )
        return document

    async def get(self, *, user_id: UUID, document_id: UUID) -> Document:
        """Return one of the user's documents."""
        return await self._owned(user_id, document_id)

    async def list_for_session(self, *, user_id: UUID, session_id: UUID) -> list[Document]:
        """List a session's documents, oldest upload first."""
        await self._require_session(user_id, session_id)
        result = await self._execute(
            select(documents)
            .where(documents.c.session_id == session_id, documents.c.user_id == user_id)
            .order_by(documents.c.uploaded_at, documents.c.id)
        )
        return [Document.model_validate(r) for r in result]

    async def delete(self, *, user_id: UUID, document_id: UUID) -> Document:
        """Delete a document and its analyses; return the deleted row."""
        row = await self._owned(user_id, document_id)
        # analyses go by ON DELETE CASCADE
        await self._execute(
            delete(documents).where(documents.c.id == document_id, documents.c.user_id == user_id)
        )
        return row
