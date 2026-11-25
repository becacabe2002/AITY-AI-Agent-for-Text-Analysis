"""upload_storage: path builder, AITY_UPLOADS_DIR, safe write/read/delete."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import UUID

import pytest

from aity_web.models import Document
from aity_web.services.upload_storage import (
    UPLOADS_DIR_ENV,
    UnsafeTextPathError,
    UploadStorage,
    UploadsDirNotConfiguredError,
    text_path_for,
    upload_timestamp,
)

UID = UUID("11111111-1111-1111-1111-111111111111")
SID = UUID("22222222-2222-2222-2222-222222222222")
DID = UUID("33333333-3333-3333-3333-333333333333")


def test_text_path_layout():
    at = datetime(2026, 11, 25, 12, 15, 0, 987654, tzinfo=timezone(timedelta(hours=2)))
    assert text_path_for(user_id=UID, session_id=SID, document_id=DID, uploaded_at=at) == (
        f"{UID}/{SID}/{DID}_{UID}_20261125T101500Z.txt"
    )


def test_text_path_is_accepted_by_document_model():
    at = upload_timestamp(datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc))
    path = text_path_for(user_id=UID, session_id=SID, document_id=DID, uploaded_at=at)
    Document(
        id=DID, session_id=SID, user_id=UID, original_filename="a.txt", source_type="txt",
        text_path=path, text_sha256="0" * 64, char_count=1, uploaded_at=at,
    )


def test_upload_timestamp_is_utc_whole_seconds():
    at = datetime(2026, 1, 1, 0, 0, 1, 500, tzinfo=timezone(timedelta(hours=-5)))
    assert upload_timestamp(at) == datetime(2026, 1, 1, 5, 0, 1, tzinfo=timezone.utc)


def test_naive_timestamp_rejected():
    with pytest.raises(ValueError):
        upload_timestamp(datetime(2026, 1, 1))


def test_root_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv(UPLOADS_DIR_ENV, str(tmp_path))
    assert UploadStorage().root == tmp_path.resolve()


def test_root_argument_wins(tmp_path, monkeypatch):
    monkeypatch.setenv(UPLOADS_DIR_ENV, "/somewhere/else")
    assert UploadStorage(tmp_path).root == tmp_path.resolve()


@pytest.mark.parametrize("value", [None, ""])
def test_missing_root(monkeypatch, value):
    if value is None:
        monkeypatch.delenv(UPLOADS_DIR_ENV, raising=False)
    else:
        monkeypatch.setenv(UPLOADS_DIR_ENV, value)
    with pytest.raises(UploadsDirNotConfiguredError):
        UploadStorage()


def test_write_read_delete_roundtrip(tmp_path):
    st = UploadStorage(tmp_path)
    rel = f"{UID}/{SID}/{DID}_x.txt"
    st.write(rel, b"\xef\xbb\xbfhello")
    assert (tmp_path / str(UID) / str(SID) / f"{DID}_x.txt").read_bytes() == b"\xef\xbb\xbfhello"
    assert st.read_text(rel) == "hello"  # BOM dropped
    assert not list(tmp_path.rglob("*.part"))
    assert st.delete(rel) is True
    assert st.delete(rel) is True  # missing is fine
    assert st.remove_session_dir(user_id=UID, session_id=SID) is True
    assert not (tmp_path / str(UID) / str(SID)).exists()
    assert (tmp_path / str(UID)).exists()  # user folder kept


def test_write_refuses_overwrite(tmp_path):
    st = UploadStorage(tmp_path)
    st.write("a/b/c.txt", b"1")
    with pytest.raises(FileExistsError):
        st.write("a/b/c.txt", b"2")
    assert st.read_bytes("a/b/c.txt") == b"1"


@pytest.mark.parametrize(
    "rel", ["", "/etc/passwd", "../x.txt", "a/../../x.txt", "a\\b.txt", "a//b.txt", "./a.txt"]
)
def test_unsafe_paths_refused(tmp_path, rel):
    st = UploadStorage(tmp_path)
    with pytest.raises(UnsafeTextPathError):
        st.resolve(rel)
    assert st.delete(rel) is False


def test_symlink_escape_refused(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "root"
    root.mkdir()
    (root / "link").symlink_to(outside, target_is_directory=True)
    with pytest.raises(UnsafeTextPathError):
        UploadStorage(root).resolve("link/x.txt")


def test_remove_session_dir_keeps_non_empty_folder(tmp_path):
    st = UploadStorage(tmp_path)
    st.write(f"{UID}/{SID}/left.txt", b"x")
    (tmp_path / str(UID) / str(SID) / "crash.txt.part").write_bytes(b"partial")
    assert st.remove_session_dir(user_id=UID, session_id=SID) is False
    assert not (tmp_path / str(UID) / str(SID) / "crash.txt.part").exists()
    assert st.remove_session_dir(user_id=UUID(int=9), session_id=SID) is True  # missing is fine


def test_failed_write_leaves_no_part_file(tmp_path, monkeypatch):
    st = UploadStorage(tmp_path)
    import aity_web.services.upload_storage as mod

    def boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(mod.os, "replace", boom)
    with pytest.raises(OSError):
        st.write("a/b/c.txt", b"data")
    assert not list(tmp_path.rglob("*.part"))
    assert not (tmp_path / "a" / "b" / "c.txt").exists()
