"""Repository contract suite (ADR-004 §7.9).

Every test runs against each backend in BACKENDS. W3 Group A: the in-memory
fakes. W3 Group B adds "postgres" (opt-in, AITY_DB_TESTS=1), so the same
tests prove the fakes behave like the database.

A backend fixture provides a fresh, empty store, a UnitOfWork factory and
the controllable clock the repositories use for the timestamps they set.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable
from uuid import UUID, uuid4

import pytest
import pytest_asyncio

from aity_web.models import AnalysisResult, Document, SustainabilityMetrics
from aity_web.repositories import UNSET, ConflictError, NotFoundError, UnitOfWork
from aity_web.repositories.memory import InMemoryStore

pytestmark = pytest.mark.asyncio

T0 = datetime(2026, 11, 25, 10, 0, tzinfo=timezone.utc)


class FakeClock:
    """Each call returns a time one second later than the previous one."""

    def __init__(self, start: datetime = T0) -> None:
        self.now = start

    def __call__(self) -> datetime:
        self.now += timedelta(seconds=1)
        return self.now


@dataclass
class Backend:
    name: str
    uow: Callable[[], UnitOfWork]
    clock: FakeClock


def _memory_backend() -> Backend:
    clock = FakeClock()
    return Backend("memory", InMemoryStore(clock=clock).uow_factory, clock)


BACKENDS = {"memory": _memory_backend}


@pytest.fixture(params=sorted(BACKENDS))
def backend(request) -> Backend:
    return BACKENDS[request.param]()


# --- builders --------------------------------------------------------------------


async def make_user(be: Backend, username: str = "mintu") -> UUID:
    async with be.uow() as uow:
        user = await uow.users.add(username=username, password_hash="$2b$12$hash")
        await uow.commit()
    return user.id


async def make_session(be: Backend, user_id: UUID, name: str = "SMS pilot") -> UUID:
    async with be.uow() as uow:
        s = await uow.sessions.add(user_id=user_id, name=name)
        await uow.commit()
    return s.id


def build_document(user_id: UUID, session_id: UUID, *, uploaded_at: datetime | None = None) -> Document:
    doc_id = uuid4()
    return Document(
        id=doc_id, session_id=session_id, user_id=user_id,
        original_filename="qsl-survey.pdf", source_type="pdf",
        text_path=f"{user_id}/{session_id}/{doc_id}_{user_id}_20261125T100000Z.txt",
        text_sha256="a" * 64, char_count=1234, uploaded_at=uploaded_at or T0,
    )


async def make_document(be: Backend, user_id: UUID, session_id: UUID, **kw) -> Document:
    doc = build_document(user_id, session_id, **kw)
    async with be.uow() as uow:
        await uow.documents.add(user_id=user_id, document=doc)
        await uow.commit()
    return doc


def build_analysis(doc: Document, *, created_at: datetime | None = None, **kw) -> AnalysisResult:
    created = created_at or T0
    data = dict(
        analysis_id=uuid4(), session_id=doc.session_id, document_id=doc.id, user_id=doc.user_id,
        provider="gemini", model="gemini-3.5-flash-lite", prompt_variant="default_analysis",
        payload={
            "summary": "A short summary.",
            "keywords": ["quantum software"],
            "topics": [{"label": "SE", "description": "Why."}],
        },
        token_usage={"prompt_tokens": 100, "output_tokens": 50},
        sustainability=SustainabilityMetrics.unavailable(latency_seconds=2.0),
        content_hash="b" * 64, cache_key_version=1, status="ok",
        created_at=created, finished_at=created + timedelta(seconds=2),
    )
    data.update(kw)
    return AnalysisResult(**data)


async def make_analysis(be: Backend, doc: Document, **kw) -> AnalysisResult:
    result = build_analysis(doc, **kw)
    async with be.uow() as uow:
        await uow.analyses.add(user_id=doc.user_id, result=result)
        await uow.commit()
    return result


@pytest_asyncio.fixture
async def world(backend):
    """Two users; Alice has one session with one document and one analysis."""
    alice = await make_user(backend, "alice")
    bob = await make_user(backend, "bob")
    sid = await make_session(backend, alice)
    doc = await make_document(backend, alice, sid)
    res = await make_analysis(backend, doc)
    return backend, alice, bob, sid, doc, res


# --- unit of work -------------------------------------------------------------------


async def test_uncommitted_writes_are_rolled_back_on_exit(backend):
    async with backend.uow() as uow:
        await uow.users.add(username="ghost", password_hash="h")
    async with backend.uow() as uow:
        assert await uow.users.get_by_username(username="ghost") is None


async def test_exception_rolls_back_and_propagates(backend):
    with pytest.raises(RuntimeError, match="boom"):
        async with backend.uow() as uow:
            await uow.users.add(username="ghost", password_hash="h")
            raise RuntimeError("boom")
    async with backend.uow() as uow:
        assert await uow.users.get_by_username(username="ghost") is None


async def test_commit_is_visible_to_a_new_unit_of_work(backend):
    uid = await make_user(backend, "mintu")
    async with backend.uow() as uow:
        assert (await uow.users.get(user_id=uid)).username == "mintu"


async def test_explicit_rollback_discards_since_last_commit(backend):
    async with backend.uow() as uow:
        await uow.users.add(username="kept", password_hash="h")
        await uow.commit()
        await uow.users.add(username="dropped", password_hash="h")
        await uow.rollback()
        assert await uow.users.get_by_username(username="dropped") is None
        assert await uow.users.get_by_username(username="kept") is not None


async def test_uncommitted_writes_invisible_to_another_unit_of_work(backend):
    async with backend.uow() as writer:
        await writer.users.add(username="pending", password_hash="h")
        async with backend.uow() as reader:
            assert await reader.users.get_by_username(username="pending") is None
        await writer.commit()


async def test_repositories_unusable_outside_the_block(backend):
    uow = backend.uow()
    with pytest.raises(RuntimeError):
        await uow.users.get_by_username(username="x")


# --- users ---------------------------------------------------------------------------


async def test_user_add_get_and_lookup(backend):
    uid = await make_user(backend, "Mintu42")
    async with backend.uow() as uow:
        u = await uow.users.get(user_id=uid)
        assert u.username == "Mintu42" and u.created_at.tzinfo is not None
        assert (await uow.users.get_by_username(username="Mintu42")).id == uid
        assert await uow.users.get_by_username(username="mintu42") is None  # case-sensitive


async def test_duplicate_username_conflicts_but_case_differs_ok(backend):
    await make_user(backend, "alice")
    async with backend.uow() as uow:
        with pytest.raises(ConflictError):
            await uow.users.add(username="alice", password_hash="h")
    await make_user(backend, "Alice")


async def test_missing_user_not_found(backend):
    async with backend.uow() as uow:
        with pytest.raises(NotFoundError):
            await uow.users.get(user_id=uuid4())


# --- sessions ---------------------------------------------------------------------------


async def test_session_add_and_get(backend):
    uid = await make_user(backend)
    async with backend.uow() as uow:
        s = await uow.sessions.add(user_id=uid, name="  SMS pilot ", description="  ")
        await uow.commit()
    async with backend.uow() as uow:
        got = await uow.sessions.get(user_id=uid, session_id=s.id)
    assert got == s
    assert got.name == "SMS pilot" and got.description is None
    assert got.created_at == got.updated_at


async def test_session_add_for_unknown_user_not_found(backend):
    async with backend.uow() as uow:
        with pytest.raises(NotFoundError):
            await uow.sessions.add(user_id=uuid4(), name="x")


async def test_session_list_order_paging_and_isolation(backend):
    alice, bob = await make_user(backend, "alice"), await make_user(backend, "bob")
    first = await make_session(backend, alice, "first")
    second = await make_session(backend, alice, "second")
    await make_session(backend, bob, "bobs")
    async with backend.uow() as uow:
        assert [s.id for s in await uow.sessions.list(user_id=alice)] == [second, first]
        await uow.sessions.touch(user_id=alice, session_id=first)
        assert [s.id for s in await uow.sessions.list(user_id=alice)] == [first, second]
        assert [s.id for s in await uow.sessions.list(user_id=alice, limit=1, offset=1)] == [second]
        assert await uow.sessions.list(user_id=uuid4()) == []
        with pytest.raises(ValueError):
            await uow.sessions.list(user_id=alice, limit=0)
        with pytest.raises(ValueError):
            await uow.sessions.list(user_id=alice, offset=-1)


async def test_session_update_name_description_and_unset(backend):
    uid = await make_user(backend)
    async with backend.uow() as uow:
        s = await uow.sessions.add(user_id=uid, name="old", description="desc")
        renamed = await uow.sessions.update(user_id=uid, session_id=s.id, name="new")
        assert (renamed.name, renamed.description) == ("new", "desc")
        assert renamed.updated_at > s.updated_at and renamed.created_at == s.created_at
        cleared = await uow.sessions.update(user_id=uid, session_id=s.id, description=None)
        assert (cleared.name, cleared.description) == ("new", None)
        same = await uow.sessions.update(user_id=uid, session_id=s.id, name=UNSET, description=UNSET)
        assert (same.name, same.description) == ("new", None)
        with pytest.raises(ValueError):  # pydantic ValidationError is a ValueError
            await uow.sessions.update(user_id=uid, session_id=s.id, name="   ")
        await uow.commit()
    async with backend.uow() as uow:
        assert (await uow.sessions.get(user_id=uid, session_id=s.id)).name == "new"


async def test_session_touch_bumps_updated_at_only(backend):
    uid = await make_user(backend)
    sid = await make_session(backend, uid)
    async with backend.uow() as uow:
        before = await uow.sessions.get(user_id=uid, session_id=sid)
        after = await uow.sessions.touch(user_id=uid, session_id=sid)
    assert after.updated_at > before.updated_at
    assert (after.name, after.description, after.created_at) == (before.name, before.description, before.created_at)


async def test_session_delete_cascades_and_returns_text_paths(world):
    be, alice, _, sid, doc, res = world
    doc2 = await make_document(be, alice, sid, uploaded_at=T0 + timedelta(minutes=1))
    other_sid = await make_session(be, alice, "kept")
    kept_doc = await make_document(be, alice, other_sid)
    async with be.uow() as uow:
        paths = await uow.sessions.delete(user_id=alice, session_id=sid)
        await uow.commit()
    assert paths == [doc.text_path, doc2.text_path]
    async with be.uow() as uow:
        for call in (
            uow.sessions.get(user_id=alice, session_id=sid),
            uow.documents.get(user_id=alice, document_id=doc.id),
            uow.analyses.get(user_id=alice, analysis_id=res.analysis_id),
        ):
            with pytest.raises(NotFoundError):
                await call
        assert (await uow.documents.get(user_id=alice, document_id=kept_doc.id)) == kept_doc


async def test_session_empty_delete_returns_no_paths(backend):
    uid = await make_user(backend)
    sid = await make_session(backend, uid)
    async with backend.uow() as uow:
        assert await uow.sessions.delete(user_id=uid, session_id=sid) == []


# --- documents -----------------------------------------------------------------------------


async def test_document_add_get_list(world):
    be, alice, _, sid, doc, _ = world
    later = await make_document(be, alice, sid, uploaded_at=T0 + timedelta(minutes=5))
    async with be.uow() as uow:
        assert await uow.documents.get(user_id=alice, document_id=doc.id) == doc
        assert [d.id for d in await uow.documents.list_for_session(user_id=alice, session_id=sid)] == [doc.id, later.id]


async def test_document_add_checks(world):
    be, alice, bob, sid, doc, _ = world
    async with be.uow() as uow:
        with pytest.raises(ValueError):  # user_id mismatch: programming error
            await uow.documents.add(user_id=bob, document=build_document(alice, sid))
        with pytest.raises(NotFoundError):  # bob's document in alice's session
            await uow.documents.add(user_id=bob, document=build_document(bob, sid))
        with pytest.raises(NotFoundError):  # session does not exist
            await uow.documents.add(user_id=alice, document=build_document(alice, uuid4()))
        with pytest.raises(ConflictError):  # id reused
            await uow.documents.add(user_id=alice, document=doc)


async def test_document_delete_cascades_analyses(world):
    be, alice, _, sid, doc, res = world
    async with be.uow() as uow:
        deleted = await uow.documents.delete(user_id=alice, document_id=doc.id)
        await uow.commit()
    assert deleted == doc
    async with be.uow() as uow:
        with pytest.raises(NotFoundError):
            await uow.analyses.get(user_id=alice, analysis_id=res.analysis_id)
        assert await uow.documents.list_for_session(user_id=alice, session_id=sid) == []
        await uow.sessions.get(user_id=alice, session_id=sid)  # session kept


# --- analyses -------------------------------------------------------------------------------


async def test_analysis_get_and_lists_newest_first(world):
    be, alice, _, sid, doc, res = world
    newer = await make_analysis(be, doc, created_at=T0 + timedelta(hours=1), status="partial",
                                error="11 keywords (max 10)")
    failed = await make_analysis(be, doc, created_at=T0 + timedelta(hours=2), status="failed",
                                 payload=None, token_usage=None, error="TimeoutError")
    async with be.uow() as uow:
        assert await uow.analyses.get(user_id=alice, analysis_id=res.analysis_id) == res
        assert await uow.analyses.get(user_id=alice, analysis_id=failed.analysis_id) == failed
        expected = [failed.analysis_id, newer.analysis_id, res.analysis_id]
        assert [a.analysis_id for a in await uow.analyses.list_for_document(user_id=alice, document_id=doc.id)] == expected
        assert [a.analysis_id for a in await uow.analyses.list_for_session(user_id=alice, session_id=sid)] == expected


async def test_analysis_add_checks(world):
    be, alice, bob, sid, doc, res = world
    other_sid = await make_session(be, alice, "other")
    async with be.uow() as uow:
        with pytest.raises(ValueError):  # user_id mismatch
            await uow.analyses.add(user_id=bob, result=build_analysis(doc))
        with pytest.raises(ValueError):  # wrong session for the document
            await uow.analyses.add(user_id=alice, result=build_analysis(doc, session_id=other_sid))
        bobs_view = build_analysis(doc, user_id=bob)
        with pytest.raises(NotFoundError):  # bob cannot attach to alice's document
            await uow.analyses.add(user_id=bob, result=bobs_view)
        with pytest.raises(NotFoundError):  # cache source does not exist
            await uow.analyses.add(user_id=alice, result=build_analysis(doc, served_from_analysis_id=uuid4()))
        with pytest.raises(ConflictError):
            await uow.analyses.add(user_id=alice, result=res)


async def test_analysis_delete_clears_served_from_on_copies(world):
    be, alice, _, sid, doc, res = world
    copy = await make_analysis(be, doc, served_from_analysis_id=res.analysis_id,
                               created_at=T0 + timedelta(minutes=1))
    async with be.uow() as uow:
        await uow.analyses.delete(user_id=alice, analysis_id=res.analysis_id)
        await uow.commit()
    async with be.uow() as uow:
        with pytest.raises(NotFoundError):
            await uow.analyses.get(user_id=alice, analysis_id=res.analysis_id)
        kept = await uow.analyses.get(user_id=alice, analysis_id=copy.analysis_id)
    assert kept.served_from_analysis_id is None
    assert kept.payload == copy.payload and kept.sustainability == copy.sustainability


# --- credentials -----------------------------------------------------------------------------


async def test_credential_upsert_get_list_delete(backend):
    uid = await make_user(backend)
    async with backend.uow() as uow:
        assert await uow.credentials.get(user_id=uid, provider="gemini") is None
        first = await uow.credentials.upsert(user_id=uid, provider="gemini", ciphertext=b"c1")
        second = await uow.credentials.upsert(user_id=uid, provider="gemini", ciphertext=b"c2")
        assert second.ciphertext == b"c2"
        assert second.created_at == first.created_at and second.updated_at > first.updated_at
        await uow.credentials.upsert(user_id=uid, provider="anthropic", ciphertext=b"c3")
        await uow.commit()
    async with backend.uow() as uow:
        assert [k.provider for k in await uow.credentials.list(user_id=uid)] == ["anthropic", "gemini"]
        await uow.credentials.delete(user_id=uid, provider="gemini")
        assert await uow.credentials.get(user_id=uid, provider="gemini") is None
        with pytest.raises(NotFoundError):
            await uow.credentials.delete(user_id=uid, provider="gemini")


async def test_credential_for_unknown_user_not_found(backend):
    async with backend.uow() as uow:
        with pytest.raises(NotFoundError):
            await uow.credentials.upsert(user_id=uuid4(), provider="openai", ciphertext=b"c")


async def test_credentials_are_per_user(backend):
    alice, bob = await make_user(backend, "alice"), await make_user(backend, "bob")
    async with backend.uow() as uow:
        await uow.credentials.upsert(user_id=alice, provider="openai", ciphertext=b"a")
        assert await uow.credentials.get(user_id=bob, provider="openai") is None
        assert await uow.credentials.list(user_id=bob) == []
        with pytest.raises(NotFoundError):
            await uow.credentials.delete(user_id=bob, provider="openai")


# --- ownership: another user's ids behave exactly like missing ids ----------------------------

OWNERSHIP_CALLS = {
    "sessions.get": lambda u, uid, w: u.sessions.get(user_id=uid, session_id=w["sid"]),
    "sessions.update": lambda u, uid, w: u.sessions.update(user_id=uid, session_id=w["sid"], name="x"),
    "sessions.touch": lambda u, uid, w: u.sessions.touch(user_id=uid, session_id=w["sid"]),
    "sessions.delete": lambda u, uid, w: u.sessions.delete(user_id=uid, session_id=w["sid"]),
    "documents.get": lambda u, uid, w: u.documents.get(user_id=uid, document_id=w["did"]),
    "documents.list_for_session": lambda u, uid, w: u.documents.list_for_session(user_id=uid, session_id=w["sid"]),
    "documents.delete": lambda u, uid, w: u.documents.delete(user_id=uid, document_id=w["did"]),
    "analyses.get": lambda u, uid, w: u.analyses.get(user_id=uid, analysis_id=w["aid"]),
    "analyses.list_for_document": lambda u, uid, w: u.analyses.list_for_document(user_id=uid, document_id=w["did"]),
    "analyses.list_for_session": lambda u, uid, w: u.analyses.list_for_session(user_id=uid, session_id=w["sid"]),
    "analyses.delete": lambda u, uid, w: u.analyses.delete(user_id=uid, analysis_id=w["aid"]),
}


@pytest.mark.parametrize("call", sorted(OWNERSHIP_CALLS))
@pytest.mark.parametrize("who", ["other_user", "missing_ids"])
async def test_not_found_for_other_users_and_missing_rows(world, call, who):
    be, alice, bob, sid, doc, res = world
    if who == "other_user":
        uid, ids = bob, {"sid": sid, "did": doc.id, "aid": res.analysis_id}
    else:
        uid, ids = alice, {"sid": uuid4(), "did": uuid4(), "aid": uuid4()}
    async with be.uow() as uow:
        with pytest.raises(NotFoundError):
            await OWNERSHIP_CALLS[call](uow, uid, ids)
    # nothing of alice's was touched
    async with be.uow() as uow:
        assert await uow.sessions.get(user_id=alice, session_id=sid)
        assert await uow.documents.get(user_id=alice, document_id=doc.id) == doc
        assert await uow.analyses.get(user_id=alice, analysis_id=res.analysis_id) == res
