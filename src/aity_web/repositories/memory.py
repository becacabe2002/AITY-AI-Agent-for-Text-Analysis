"""In-memory repositories and unit of work - used ONLY in tests.

They follow the same contract as the SQLAlchemy implementations, proven by
the shared suite in tests/aity_web/test_repository_contract.py.

Transactions: each unit of work works on a private copy of the store taken
when the block is entered (or after the last commit / rollback). commit()
publishes that copy; leaving without commit() drops it. Two units of work
open at once do not see each other's uncommitted writes, and the last
commit wins (there is no locking: enough for single-task tests).

Usage in a test:

    store = InMemoryStore(clock=FakeClock())
    uow_factory = store.uow_factory          # what services take
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import UUID, uuid4

from aity_web.models import (
    AnalysisResult,
    CloudProviderName,
    Document,
    Session,
    User,
    UserApiKey,
)
from aity_web.repositories.base import (
    DEFAULT_PAGE_SIZE,
    UNSET,
    AnalysisRepository,
    ConflictError,
    CredentialRepository,
    DocumentRepository,
    NotFoundError,
    SessionRepository,
    UnitOfWork,
    Unset,
    UserRepository,
)

Clock = Callable[[], datetime]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class _State:
    users: dict[UUID, User] = field(default_factory=dict)
    sessions: dict[UUID, Session] = field(default_factory=dict)
    documents: dict[UUID, Document] = field(default_factory=dict)
    analyses: dict[UUID, AnalysisResult] = field(default_factory=dict)
    credentials: dict[tuple[UUID, str], UserApiKey] = field(default_factory=dict)

    def copy(self) -> _State:
        # Domain models are frozen, so copying the dicts is enough.
        return _State(
            users=dict(self.users),
            sessions=dict(self.sessions),
            documents=dict(self.documents),
            analyses=dict(self.analyses),
            credentials=dict(self.credentials),
        )


class InMemoryStore:
    """The committed data shared by every unit of work built from it."""

    def __init__(self, *, clock: Clock = utc_now) -> None:
        self.clock = clock
        self._state = _State()

    def uow_factory(self) -> InMemoryUnitOfWork:
        return InMemoryUnitOfWork(self)

    def _snapshot(self) -> _State:
        return self._state.copy()

    def _publish(self, state: _State) -> None:
        self._state = state.copy()


class _Repo:
    def __init__(self, uow: InMemoryUnitOfWork) -> None:
        self._uow = uow

    @property
    def _s(self) -> _State:
        return self._uow._active_state()

    def _now(self) -> datetime:
        return self._uow._store.clock()

    # ownership helpers: missing and not-owned look the same
    def _owned_session(self, user_id: UUID, session_id: UUID) -> Session:
        row = self._s.sessions.get(session_id)
        if row is None or row.user_id != user_id:
            raise NotFoundError(f"session {session_id} not found")
        return row

    def _owned_document(self, user_id: UUID, document_id: UUID) -> Document:
        row = self._s.documents.get(document_id)
        if row is None or row.user_id != user_id:
            raise NotFoundError(f"document {document_id} not found")
        return row

    def _owned_analysis(self, user_id: UUID, analysis_id: UUID) -> AnalysisResult:
        row = self._s.analyses.get(analysis_id)
        if row is None or row.user_id != user_id:
            raise NotFoundError(f"analysis {analysis_id} not found")
        return row

    def _require_user(self, user_id: UUID) -> None:
        if user_id not in self._s.users:
            raise NotFoundError(f"user {user_id} not found")


class InMemoryUserRepository(_Repo, UserRepository):
    async def add(self, *, username: str, password_hash: str) -> User:
        if any(u.username == username for u in self._s.users.values()):
            raise ConflictError(f"username {username!r} is taken")
        user = User(id=uuid4(), username=username, password_hash=password_hash, created_at=self._now())
        self._s.users[user.id] = user
        return user

    async def get(self, *, user_id: UUID) -> User:
        user = self._s.users.get(user_id)
        if user is None:
            raise NotFoundError(f"user {user_id} not found")
        return user

    async def get_by_username(self, *, username: str) -> User | None:
        return next((u for u in self._s.users.values() if u.username == username), None)


class InMemorySessionRepository(_Repo, SessionRepository):
    async def add(self, *, user_id: UUID, name: str, description: str | None = None) -> Session:
        self._require_user(user_id)
        now = self._now()
        row = Session(
            id=uuid4(), user_id=user_id, name=name, description=description,
            created_at=now, updated_at=now,
        )
        self._s.sessions[row.id] = row
        return row

    async def get(self, *, user_id: UUID, session_id: UUID) -> Session:
        return self._owned_session(user_id, session_id)

    async def list(
        self, *, user_id: UUID, limit: int = DEFAULT_PAGE_SIZE, offset: int = 0
    ) -> list[Session]:
        if limit < 1 or offset < 0:
            raise ValueError("limit must be >= 1 and offset >= 0")
        rows = [s for s in self._s.sessions.values() if s.user_id == user_id]
        rows.sort(key=lambda s: str(s.id))
        rows.sort(key=lambda s: s.updated_at, reverse=True)
        return rows[offset : offset + limit]

    async def update(
        self,
        *,
        user_id: UUID,
        session_id: UUID,
        name: str | Unset = UNSET,
        description: str | None | Unset = UNSET,
    ) -> Session:
        row = self._owned_session(user_id, session_id)
        data = row.model_dump()
        if name is not UNSET:
            data["name"] = name
        if description is not UNSET:
            data["description"] = description
        data["updated_at"] = self._now()
        new = Session.model_validate(data)  # re-validates name/description
        self._s.sessions[session_id] = new
        return new

    async def touch(self, *, user_id: UUID, session_id: UUID) -> Session:
        row = self._owned_session(user_id, session_id)
        new = Session.model_validate({**row.model_dump(), "updated_at": self._now()})
        self._s.sessions[session_id] = new
        return new

    async def delete(self, *, user_id: UUID, session_id: UUID) -> list[str]:
        self._owned_session(user_id, session_id)
        docs = sorted(
            (d for d in self._s.documents.values() if d.session_id == session_id),
            key=lambda d: (d.uploaded_at, str(d.id)),
        )
        for a in [a for a in self._s.analyses.values() if a.session_id == session_id]:
            _delete_analysis(self._s, a.analysis_id)
        for d in docs:
            del self._s.documents[d.id]
        del self._s.sessions[session_id]
        return [d.text_path for d in docs]


class InMemoryDocumentRepository(_Repo, DocumentRepository):
    async def add(self, *, user_id: UUID, document: Document) -> Document:
        if document.user_id != user_id:
            raise ValueError("document.user_id does not match user_id")
        self._owned_session(user_id, document.session_id)
        if document.id in self._s.documents:
            raise ConflictError(f"document id {document.id} already exists")
        self._s.documents[document.id] = document
        return document

    async def get(self, *, user_id: UUID, document_id: UUID) -> Document:
        return self._owned_document(user_id, document_id)

    async def list_for_session(self, *, user_id: UUID, session_id: UUID) -> list[Document]:
        self._owned_session(user_id, session_id)
        rows = [d for d in self._s.documents.values() if d.session_id == session_id]
        return sorted(rows, key=lambda d: (d.uploaded_at, str(d.id)))

    async def delete(self, *, user_id: UUID, document_id: UUID) -> Document:
        row = self._owned_document(user_id, document_id)
        for a in [a for a in self._s.analyses.values() if a.document_id == document_id]:
            _delete_analysis(self._s, a.analysis_id)
        del self._s.documents[document_id]
        return row


class InMemoryAnalysisRepository(_Repo, AnalysisRepository):
    async def add(self, *, user_id: UUID, result: AnalysisResult) -> AnalysisResult:
        if result.user_id != user_id:
            raise ValueError("result.user_id does not match user_id")
        doc = self._owned_document(user_id, result.document_id)
        if doc.session_id != result.session_id:
            raise ValueError("result.session_id is not the document's session")
        if result.served_from_analysis_id is not None:
            self._owned_analysis(user_id, result.served_from_analysis_id)
        if result.analysis_id in self._s.analyses:
            raise ConflictError(f"analysis id {result.analysis_id} already exists")
        self._s.analyses[result.analysis_id] = result
        return result

    async def get(self, *, user_id: UUID, analysis_id: UUID) -> AnalysisResult:
        return self._owned_analysis(user_id, analysis_id)

    async def list_for_document(self, *, user_id: UUID, document_id: UUID) -> list[AnalysisResult]:
        self._owned_document(user_id, document_id)
        return _newest_first(a for a in self._s.analyses.values() if a.document_id == document_id)

    async def list_for_session(self, *, user_id: UUID, session_id: UUID) -> list[AnalysisResult]:
        self._owned_session(user_id, session_id)
        return _newest_first(a for a in self._s.analyses.values() if a.session_id == session_id)

    async def delete(self, *, user_id: UUID, analysis_id: UUID) -> None:
        self._owned_analysis(user_id, analysis_id)
        _delete_analysis(self._s, analysis_id)


class InMemoryCredentialRepository(_Repo, CredentialRepository):
    async def upsert(
        self, *, user_id: UUID, provider: CloudProviderName, ciphertext: bytes
    ) -> UserApiKey:
        self._require_user(user_id)
        now = self._now()
        old = self._s.credentials.get((user_id, provider))
        row = UserApiKey(
            user_id=user_id, provider=provider, ciphertext=ciphertext,
            created_at=old.created_at if old else now, updated_at=now,
        )
        self._s.credentials[(user_id, row.provider)] = row
        return row

    async def get(self, *, user_id: UUID, provider: CloudProviderName) -> UserApiKey | None:
        return self._s.credentials.get((user_id, provider))

    async def list(self, *, user_id: UUID) -> list[UserApiKey]:
        rows = [k for (uid, _), k in self._s.credentials.items() if uid == user_id]
        return sorted(rows, key=lambda k: k.provider)

    async def delete(self, *, user_id: UUID, provider: CloudProviderName) -> None:
        if self._s.credentials.pop((user_id, provider), None) is None:
            raise NotFoundError(f"no {provider} key for user {user_id}")


def _newest_first(rows) -> list[AnalysisResult]:
    out = sorted(rows, key=lambda a: str(a.analysis_id))
    out.sort(key=lambda a: a.created_at, reverse=True)
    return out


def _delete_analysis(state: _State, analysis_id: UUID) -> None:
    """Delete one result; copies served from it lose the link (ON DELETE SET NULL)."""
    del state.analyses[analysis_id]
    for aid, a in list(state.analyses.items()):
        if a.served_from_analysis_id == analysis_id:
            state.analyses[aid] = AnalysisResult.model_validate(
                {**a.model_dump(), "served_from_analysis_id": None}
            )


class InMemoryUnitOfWork(UnitOfWork):
    def __init__(self, store: InMemoryStore) -> None:
        self._store = store
        self._state: _State | None = None
        self.users = InMemoryUserRepository(self)
        self.sessions = InMemorySessionRepository(self)
        self.documents = InMemoryDocumentRepository(self)
        self.analyses = InMemoryAnalysisRepository(self)
        self.credentials = InMemoryCredentialRepository(self)

    def _active_state(self) -> _State:
        if self._state is None:
            raise RuntimeError("repositories are only usable inside 'async with uow:'")
        return self._state

    async def __aenter__(self) -> InMemoryUnitOfWork:
        if self._state is not None:
            raise RuntimeError("unit of work is already open")
        self._state = self._store._snapshot()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self._state = None  # uncommitted writes are dropped

    async def commit(self) -> None:
        self._store._publish(self._active_state())
        self._state = self._store._snapshot()

    async def rollback(self) -> None:
        self._active_state()
        self._state = self._store._snapshot()
