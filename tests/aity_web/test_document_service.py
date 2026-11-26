"""document_service on the in-memory unit of work (ADR-004 §7.5, §7.7)."""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from _service_fixtures import StepClock, add_user
from _upload_fixtures import make_docx, make_pdf
from aity_web.repositories import ConflictError, NotFoundError
from aity_web.repositories.memory import InMemoryStore
from aity_web.services.cache_key import text_sha256
from aity_web.services.document_service import DocumentService, clean_filename
from aity_web.services.text_extraction import (
    DocumentRejectedError,
    NoTextLayerError,
    UnsupportedFileTypeError,
)
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
def clock():
    return StepClock()


@pytest.fixture
def service(store, storage, clock):
    return DocumentService(store.uow_factory, storage, clock=clock)


async def make_session(store, user_id, name="s"):
    async with store.uow_factory() as uow:
        s = await uow.sessions.add(user_id=user_id, name=name)
        await uow.commit()
    return s


def files(tmp_path):
    return sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file())


async def test_upload_txt(store, storage, service, clock, tmp_path):
    alice = await add_user(store)
    s = await make_session(store, alice.id)
    data = b"\xef\xbb\xbfLine one  \r\n\r\n\r\nLine two\r\n"

    doc = await service.upload(user_id=alice.id, session_id=s.id, filename="C:\\fakepath\\notes.txt", data=data)

    assert doc.original_filename == "notes.txt"
    assert doc.source_type == "txt"
    assert doc.uploaded_at == clock.now and doc.uploaded_at.microsecond == 0
    assert doc.text_path == (
        f"{alice.id}/{s.id}/{doc.id}_{alice.id}_{clock.now:%Y%m%dT%H%M%SZ}.txt"
    )
    assert doc.char_count == len("Line one\n\nLine two")
    assert doc.text_sha256 == text_sha256("Line one\n\nLine two")
    assert storage.read_bytes(doc.text_path) == data  # exact copy
    assert files(tmp_path) == [doc.text_path]
    async with store.uow_factory() as uow:
        assert await uow.documents.get(user_id=alice.id, document_id=doc.id) == doc
        assert (await uow.sessions.get(user_id=alice.id, session_id=s.id)).updated_at > s.updated_at


async def test_upload_pdf_and_docx_store_extracted_text(store, storage, service):
    alice = await add_user(store)
    s = await make_session(store, alice.id)
    pdf = await service.upload(user_id=alice.id, session_id=s.id, filename="p.PDF",
                               data=make_pdf(["Page one", "Page two"]))
    docx = await service.upload(user_id=alice.id, session_id=s.id, filename="d.docx",
                                data=make_docx(lambda d: d.add_paragraph("Hello docx")))
    assert pdf.source_type == "pdf"
    assert storage.read_text(pdf.text_path) == "Page one\n\nPage two"
    assert docx.source_type == "docx"
    assert storage.read_text(docx.text_path) == "Hello docx"
    assert [d.id for d in await service.list(user_id=alice.id, session_id=s.id)] == [pdf.id, docx.id]


async def test_same_content_twice_is_two_documents(store, service):
    alice = await add_user(store)
    s = await make_session(store, alice.id)
    a = await service.upload(user_id=alice.id, session_id=s.id, filename="a.txt", data=b"same")
    b = await service.upload(user_id=alice.id, session_id=s.id, filename="a.txt", data=b"same")
    assert a.id != b.id and a.text_path != b.text_path and a.text_sha256 == b.text_sha256


@pytest.mark.parametrize(
    "filename, data, error",
    [
        ("a.doc", b"x", UnsupportedFileTypeError),
        ("scan.pdf", make_pdf([""]), NoTextLayerError),
        ("empty.txt", b"   ", DocumentRejectedError),
        ("", b"x", UnsupportedFileTypeError),
    ],
)
async def test_rejected_upload_writes_nothing(store, service, tmp_path, filename, data, error):
    alice = await add_user(store)
    s = await make_session(store, alice.id)
    with pytest.raises(error):
        await service.upload(user_id=alice.id, session_id=s.id, filename=filename, data=data)
    assert files(tmp_path) == []
    assert await service.list(user_id=alice.id, session_id=s.id) == []


async def test_upload_into_foreign_or_missing_session(store, service, tmp_path):
    alice, bob = await add_user(store, "alice"), await add_user(store, "bob")
    s = await make_session(store, alice.id)
    for user_id, session_id in [(bob.id, s.id), (alice.id, uuid4())]:
        with pytest.raises(NotFoundError):
            await service.upload(user_id=user_id, session_id=session_id, filename="a.txt", data=b"x")
    assert files(tmp_path) == []


async def test_failed_commit_removes_the_file(store, storage, clock, tmp_path):
    alice = await add_user(store)
    s = await make_session(store, alice.id)

    def failing_factory():
        uow = store.uow_factory()

        async def commit():
            raise RuntimeError("database went away")

        uow.commit = commit
        return uow

    service = DocumentService(failing_factory, storage, clock=clock)
    with pytest.raises(RuntimeError):
        await service.upload(user_id=alice.id, session_id=s.id, filename="a.txt", data=b"x")
    assert files(tmp_path) == []
    async with store.uow_factory() as uow:
        assert await uow.documents.list_for_session(user_id=alice.id, session_id=s.id) == []


async def test_duplicate_id_conflict_removes_the_file(store, storage, clock, tmp_path):
    alice = await add_user(store)
    s = await make_session(store, alice.id)
    fixed = UUID("44444444-4444-4444-4444-444444444444")
    service = DocumentService(store.uow_factory, storage, clock=clock, new_id=lambda: fixed)
    first = await service.upload(user_id=alice.id, session_id=s.id, filename="a.txt", data=b"x")
    with pytest.raises(ConflictError):
        await service.upload(user_id=alice.id, session_id=s.id, filename="b.txt", data=b"y")
    assert files(tmp_path) == [first.text_path]


async def test_get_and_list_are_owner_only(store, service):
    alice, bob = await add_user(store, "alice"), await add_user(store, "bob")
    s = await make_session(store, alice.id)
    d = await service.upload(user_id=alice.id, session_id=s.id, filename="a.txt", data=b"x")
    assert await service.get(user_id=alice.id, document_id=d.id) == d
    with pytest.raises(NotFoundError):
        await service.get(user_id=bob.id, document_id=d.id)
    with pytest.raises(NotFoundError):
        await service.list(user_id=bob.id, session_id=s.id)


async def test_delete_removes_row_analyses_and_file(store, storage, service, tmp_path):
    alice, bob = await add_user(store, "alice"), await add_user(store, "bob")
    s = await make_session(store, alice.id)
    d = await service.upload(user_id=alice.id, session_id=s.id, filename="a.txt", data=b"x")
    other = await service.upload(user_id=alice.id, session_id=s.id, filename="b.txt", data=b"y")
    analysis = build_analysis(d)
    async with store.uow_factory() as uow:
        await uow.analyses.add(user_id=alice.id, result=analysis)
        await uow.commit()

    with pytest.raises(NotFoundError):
        await service.delete(user_id=bob.id, document_id=d.id)
    await service.delete(user_id=alice.id, document_id=d.id)

    with pytest.raises(NotFoundError):
        await service.get(user_id=alice.id, document_id=d.id)
    async with store.uow_factory() as uow:
        with pytest.raises(NotFoundError):
            await uow.analyses.get(user_id=alice.id, analysis_id=analysis.analysis_id)
    assert files(tmp_path) == [other.text_path]
    with pytest.raises(NotFoundError):
        await service.delete(user_id=alice.id, document_id=d.id)


async def test_delete_survives_file_errors(store, storage, service, monkeypatch, caplog):
    alice = await add_user(store)
    s = await make_session(store, alice.id)
    d = await service.upload(user_id=alice.id, session_id=s.id, filename="a.txt", data=b"x")
    monkeypatch.setattr(storage, "delete", lambda rel: False)
    await service.delete(user_id=alice.id, document_id=d.id)
    with pytest.raises(NotFoundError):
        await service.get(user_id=alice.id, document_id=d.id)
    assert "left on disk" in caplog.text


# --- clean_filename (sync) ---


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("report.pdf", "report.pdf"),
        ("C:\\Users\\me\\report.pdf", "report.pdf"),
        ("/home/me/report.pdf", "report.pdf"),
        ("  spaced name.txt  ", "spaced name.txt"),
        ("bad\x00\x1fname\n.txt", "badname.txt"),
        ("Lähde – raportti.docx", "Lähde – raportti.docx"),
    ],
)
async def test_clean_filename(raw, expected):
    assert clean_filename(raw) == expected


async def test_clean_filename_truncates_keeping_extension():
    name = clean_filename("a" * 300 + ".docx")
    assert len(name) == 255 and name.endswith(".docx")
