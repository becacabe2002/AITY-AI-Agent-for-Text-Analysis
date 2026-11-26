"""PostgreSQL implementation of the analysis repository."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from sqlalchemy import delete, insert, select

from aity_web.db.tables import analysis_results, documents, sessions
from aity_web.models import AnalysisResult
from aity_web.repositories.base import AnalysisRepository, NotFoundError
from aity_web.repositories.sqlalchemy._common import SqlRepo

_JSONB_FIELDS = ("payload", "token_usage", "sustainability")


def to_row(result: AnalysisResult) -> dict[str, Any]:
    """Convert a result to column values, with the JSON blobs as plain dicts."""
    row = result.model_dump(exclude=set(_JSONB_FIELDS))
    for name in _JSONB_FIELDS:
        value = getattr(result, name)
        row[name] = None if value is None else value.model_dump(mode="json")
    return row


def from_row(row: Any) -> AnalysisResult:
    # schema_version is Literal[1]: a row written by a newer shape is refused
    """Build a result from a database row; an unknown schema version is refused."""
    return AnalysisResult.model_validate(dict(row._mapping))


class SqlAlchemyAnalysisRepository(SqlRepo, AnalysisRepository):
    """SQLAlchemy implementation of AnalysisRepository."""

    async def _owned(self, user_id: UUID, analysis_id: UUID) -> AnalysisResult:
        """Return the user's result or raise NotFoundError."""
        row = (
            await self._execute(
                select(analysis_results).where(
                    analysis_results.c.analysis_id == analysis_id,
                    analysis_results.c.user_id == user_id,
                )
            )
        ).one_or_none()
        if row is None:
            raise NotFoundError(f"analysis {analysis_id} not found")
        return from_row(row)

    async def _list(self, *conditions) -> list[AnalysisResult]:
        """List results matching the conditions, newest first."""
        result = await self._execute(
            select(analysis_results)
            .where(*conditions)
            .order_by(analysis_results.c.created_at.desc(), analysis_results.c.analysis_id)
        )
        return [from_row(r) for r in result]

    async def add(self, *, user_id: UUID, result: AnalysisResult) -> AnalysisResult:
        """Insert a result built by the service."""
        if result.user_id != user_id:
            raise ValueError("result.user_id does not match user_id")
        doc_session = (
            await self._execute(
                select(documents.c.session_id).where(
                    documents.c.id == result.document_id, documents.c.user_id == user_id
                )
            )
        ).scalar_one_or_none()
        if doc_session is None:
            raise NotFoundError(f"document {result.document_id} not found")
        if doc_session != result.session_id:
            raise ValueError("result.session_id is not the document's session")
        if result.served_from_analysis_id is not None:
            source = (
                await self._execute(
                    select(analysis_results.c.analysis_id).where(
                        analysis_results.c.analysis_id == result.served_from_analysis_id,
                        analysis_results.c.user_id == user_id,
                    )
                )
            ).first()
            if source is None:
                raise NotFoundError(f"analysis {result.served_from_analysis_id} not found")
        await self._insert(
            insert(analysis_results).values(**to_row(result)),
            f"analysis id {result.analysis_id} already exists, or its document is gone",
        )
        return result

    async def get(self, *, user_id: UUID, analysis_id: UUID) -> AnalysisResult:
        """Return one of the user's results."""
        return await self._owned(user_id, analysis_id)

    async def list_for_document(self, *, user_id: UUID, document_id: UUID) -> list[AnalysisResult]:
        """List a document's results, newest first."""
        found = (
            await self._execute(
                select(documents.c.id).where(documents.c.id == document_id, documents.c.user_id == user_id)
            )
        ).first()
        if found is None:
            raise NotFoundError(f"document {document_id} not found")
        return await self._list(
            analysis_results.c.document_id == document_id, analysis_results.c.user_id == user_id
        )

    async def list_for_session(self, *, user_id: UUID, session_id: UUID) -> list[AnalysisResult]:
        """List a session's results, newest first."""
        found = (
            await self._execute(
                select(sessions.c.id).where(sessions.c.id == session_id, sessions.c.user_id == user_id)
            )
        ).first()
        if found is None:
            raise NotFoundError(f"session {session_id} not found")
        return await self._list(
            analysis_results.c.session_id == session_id, analysis_results.c.user_id == user_id
        )

    async def delete(self, *, user_id: UUID, analysis_id: UUID) -> None:
        """Delete one result; copies served from it lose their link to it."""
        deleted = (
            await self._execute(
                delete(analysis_results)
                .where(
                    analysis_results.c.analysis_id == analysis_id,
                    analysis_results.c.user_id == user_id,
                )
                .returning(analysis_results.c.analysis_id)
            )
        ).first()
        if deleted is None:
            raise NotFoundError(f"analysis {analysis_id} not found")
        # copies served from it: served_from_analysis_id -> NULL (ON DELETE SET NULL)
