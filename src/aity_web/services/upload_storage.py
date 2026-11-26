"""File storage for the extracted text of uploaded documents."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from uuid import UUID

from aity_web.services.text_extraction import decode_txt

log = logging.getLogger(__name__)

UPLOADS_DIR_ENV = "AITY_UPLOADS_DIR"
TIMESTAMP_FORMAT = "%Y%m%dT%H%M%SZ"


class UploadsDirNotConfiguredError(RuntimeError):
    """AITY_UPLOADS_DIR is not set and no root was passed."""


class UnsafeTextPathError(ValueError):
    """A text path that would leave the uploads root."""


def upload_timestamp(now: datetime) -> datetime:
    """Return the upload time as UTC with whole seconds."""
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("uploaded_at must be timezone-aware")
    return now.astimezone(timezone.utc).replace(microsecond=0)


def text_path_for(
    *, user_id: UUID, session_id: UUID, document_id: UUID, uploaded_at: datetime
) -> str:
    """The relative POSIX path of a document's text file (see module docstring)."""
    stamp = upload_timestamp(uploaded_at).strftime(TIMESTAMP_FORMAT)
    name = f"{document_id}_{user_id}_{stamp}.txt"
    return PurePosixPath(str(user_id), str(session_id), name).as_posix()


def session_dir_for(*, user_id: UUID, session_id: UUID) -> str:
    """Return the relative folder of a user's session."""
    return PurePosixPath(str(user_id), str(session_id)).as_posix()


class UploadStorage:
    """File operations on the uploads root. One instance per process."""

    def __init__(self, root: str | os.PathLike[str] | None = None) -> None:
        """Create the storage on a root folder or on $AITY_UPLOADS_DIR."""
        if root is None:
            root = os.environ.get(UPLOADS_DIR_ENV) or None
        if root is None:
            raise UploadsDirNotConfiguredError(
                f"{UPLOADS_DIR_ENV} is not set (see .env.example)"
            )
        self.root = Path(root).resolve()

    def __repr__(self) -> str:
        return f"UploadStorage(root={str(self.root)!r})"

    # --- paths ---

    def resolve(self, rel: str) -> Path:
        """Absolute path of a relative text_path; refuses anything outside the root."""
        pure = PurePosixPath(rel)
        if (
            not rel
            or "\\" in rel
            or pure.is_absolute()
            or ".." in pure.parts
            or pure.as_posix() != rel
        ):
            raise UnsafeTextPathError("not a clean relative POSIX path")
        path = self.root.joinpath(*pure.parts)
        if not path.resolve().is_relative_to(self.root):  # symlinks
            raise UnsafeTextPathError("path leaves the uploads root")
        return path

    # --- write / read ---

    def write(self, rel: str, data: bytes) -> None:
        """Write a file atomically via a .part file; never overwrite an existing one."""
        target = self.resolve(rel)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError(rel)
        part = target.with_name(target.name + ".part")
        try:
            with open(part, "wb") as fh:
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(part, target)
        except BaseException:
            _quiet_unlink(part)
            raise

    def read_bytes(self, rel: str) -> bytes:
        """Read a stored file as bytes."""
        return self.resolve(rel).read_bytes()

    def read_text(self, rel: str) -> str:
        """Read a stored file as UTF-8 text."""
        return decode_txt(self.read_bytes(rel))

    # --- delete (never raises) ---

    def delete(self, rel: str) -> bool:
        """Remove one text file; return False and log if it cannot be removed."""
        try:
            path = self.resolve(rel)
        except UnsafeTextPathError:
            log.error("refusing to delete an unsafe text_path")
            return False
        try:
            path.unlink(missing_ok=True)
            return True
        except OSError as exc:
            log.error("could not delete text file %s: %s", rel, type(exc).__name__)
            return False

    def remove_session_dir(self, *, user_id: UUID, session_id: UUID) -> bool:
        """Remove a session's folder if it is empty; return False and log otherwise."""
        path = self.resolve(session_dir_for(user_id=user_id, session_id=session_id))
        if not path.exists():
            return True
        for leftover in path.glob("*.part"):
            _quiet_unlink(leftover)
        try:
            path.rmdir()
            return True
        except OSError as exc:  # not empty (a failed delete) or a permission problem
            log.error("could not remove session folder %s: %s", path.name, type(exc).__name__)
            return False


def _quiet_unlink(path: Path) -> None:
    """Delete a file, logging instead of raising."""
    try:
        path.unlink(missing_ok=True)
    except OSError:
        log.error("could not remove %s", path.name)


__all__ = [
    "TIMESTAMP_FORMAT",
    "UPLOADS_DIR_ENV",
    "UnsafeTextPathError",
    "UploadStorage",
    "UploadsDirNotConfiguredError",
    "session_dir_for",
    "text_path_for",
    "upload_timestamp",
]
