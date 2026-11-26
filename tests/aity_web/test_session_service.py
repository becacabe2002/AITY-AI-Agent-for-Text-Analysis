"""session_service on the in-memory unit of work (ADR-004 §7.7)."""

from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import ValidationError

from _service_fixtures import StepClock, add_user
from aity_web.repositories import UNSET, NotFoundError
from aity_web.repositories.memory import InMemoryStore
from aity_web.services.document_service import DocumentService
from aity_web.services.session_service import SessionService
from aity_web.services.upload_storage import UploadStorage
from test_repository_contract import build_analysis

pytestmark = pytest.mark.asyncio


@pytest.fixture
def store():
    return InMemoryStore(clock=StepClock())


@pytest.fixture
def storage(tmp_path):
    return UploadStorage(tmp_path)


@pytest.fixture
def sessions(store, storage):
    return SessionService(store.uow_factory, storage)


@pytest.fixture
def documents(store, storage):
    return DocumentService(store.uow_factory, storage, clock=StepClock())


async def test_create_list_get(store, sessions):
    alice = await add_user(store)
    s1 = await sessions.create(user_id=alice.id, name="  SMS pilot ", description="  ")
    s2 = await sessions.create(user_id=alice.id, name="Second", description="notes")
    assert s1.name == "SMS pilot" and s1.description is None
    assert [s.id for s in await sessions.list(user_id=alice.id)] == [s2.id, s1.id]
    assert await sessions.get(user_id=alice.id, session_id=s1.id) == s1


async def test_create_validates(store, sessions):
    alice = await add_user(store)
    with pytest.raises(ValidationError):
        await sessions.create(user_id=alice.id, name="   ")
    with pytest.raises(ValidationError):
        await sessions.create(user_id=alice.id, name="x" * 201)
    assert await sessions.list(user_id=alice.id) == []


async def test_create_for_unknown_user(sessions):
    with pytest.raises(NotFoundError):
        await sessions.create(user_id=uuid4(), name="x")


async def test_update_name_and_description(store, sessions):
    alice = await add_user(store)
    s = await sessions.create(user_id=alice.id, name="Old", description="keep me")
    renamed = await sessions.update(user_id=alice.id, session_id=s.id, name="New")
    assert (renamed.name, renamed.description) == ("New", "keep me")
    assert renamed.updated_at > s.updated_at
    cleared = await sessions.update(user_id=alice.id, session_id=s.id, description=None)
    assert (cleared.name, cleared.description) == ("New", None)
    assert await sessions.get(user_id=alice.id, session_id=s.id) == cleared


async def test_update_with_nothing_is_a_noop(store, sessions):
    alice = await add_user(store)
    s = await sessions.create(user_id=alice.id, name="Same")
    assert await sessions.update(user_id=alice.id, session_id=s.id, name=UNSET) == s


async def test_update_invalid_name_changes_nothing(store, sessions):
    alice = await add_user(store)
    s = await sessions.create(user_id=alice.id, name="Valid")
    with pytest.raises(ValidationError):
        await sessions.update(user_id=alice.id, session_id=s.id, name="")
    assert await sessions.get(user_id=alice.id, session_id=s.id) == s


async def test_other_users_session_is_not_found(store, sessions):
    alice, bob = await add_user(store, "alice"), await add_user(store, "bob")
    s = await sessions.create(user_id=alice.id, name="Private")
    with pytest.raises(NotFoundError):
        await sessions.get(user_id=bob.id, session_id=s.id)
    with pytest.raises(NotFoundError):
        await sessions.update(user_id=bob.id, session_id=s.id, name="mine now")
    with pytest.raises(NotFoundError):
        await sessions.delete(user_id=bob.id, session_id=s.id)
    assert await sessions.list(user_id=bob.id) == []
    assert (await sessions.get(user_id=alice.id, session_id=s.id)).name == "Private"


async def test_delete_cascades_rows_files_and_folder(store, storage, sessions, documents, tmp_path):
    alice = await add_user(store)
    s = await sessions.create(user_id=alice.id, name="Doomed")
    keep = await sessions.create(user_id=alice.id, name="Kept")
    d1 = await documents.upload(user_id=alice.id, session_id=s.id, filename="a.txt", data=b"one")
    d2 = await documents.upload(user_id=alice.id, session_id=s.id, filename="b.txt", data=b"two")
    d3 = await documents.upload(user_id=alice.id, session_id=keep.id, filename="c.txt", data=b"3")
    analysis = build_analysis(d1)
    async with store.uow_factory() as uow:
        await uow.analyses.add(user_id=alice.id, result=analysis)
        await uow.commit()

    await sessions.delete(user_id=alice.id, session_id=s.id)

    with pytest.raises(NotFoundError):
        await sessions.get(user_id=alice.id, session_id=s.id)
    async with store.uow_factory() as uow:
        for d in (d1, d2):
            with pytest.raises(NotFoundError):
                await uow.documents.get(user_id=alice.id, document_id=d.id)
        with pytest.raises(NotFoundError):
            await uow.analyses.get(user_id=alice.id, analysis_id=analysis.analysis_id)
        assert await uow.documents.get(user_id=alice.id, document_id=d3.id) == d3
    assert not (tmp_path / str(alice.id) / str(s.id)).exists()
    assert storage.read_bytes(d3.text_path) == b"3"


async def test_delete_survives_file_errors(store, storage, sessions, documents, monkeypatch, caplog):
    alice = await add_user(store)
    s = await sessions.create(user_id=alice.id, name="x")
    d = await documents.upload(user_id=alice.id, session_id=s.id, filename="a.txt", data=b"one")
    monkeypatch.setattr(storage, "delete", lambda rel: False)

    await sessions.delete(user_id=alice.id, session_id=s.id)  # does not raise

    with pytest.raises(NotFoundError):
        await sessions.get(user_id=alice.id, session_id=s.id)
    assert storage.read_bytes(d.text_path) == b"one"  # left behind, logged
    assert "left on disk" in caplog.text


async def test_delete_empty_session(store, sessions):
    alice = await add_user(store)
    s = await sessions.create(user_id=alice.id, name="empty")
    await sessions.delete(user_id=alice.id, session_id=s.id)
    assert await sessions.list(user_id=alice.id) == []
