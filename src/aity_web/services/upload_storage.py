"""Where document text files live - ADR-004 §7.5 / §7.7.

Layout under AITY_UPLOADS_DIR (development: a local folder such as
./var/uploads; Pouta: /srv/aity/uploads, bind-mounted from the Cinder volume):

    {user_id}/{session_id}/{document_id}_{user_id}_{uploaded_at:%Y%m%dT%H%M%SZ}.txt

Only UUIDs and a timestamp go into a path, so no user text reaches the
filesystem; the original filename is stored as metadata on the row. The
database stores the path relative to the root (Document.text_path).

Writes go to a ".part" file that is renamed into place, so a crash never
leaves a half-written text file under its final name. Deletes never raise:
they run after a database commit, and a failed file delete must not undo a
committed delete (it is logged; ADR-004 accepts the leftover file).
"""

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
    """A text_path that would leave the uploads root (never expected from
    validated Document rows; checked anyway before touching the disk)."""


def upload_timestamp(now: datetime) -> datetime:
    """uploaded_at as stored: UTC, whole seconds, so the row and the
    timestamp in the file name are the same instant."""
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
    return PurePosixPath(str(user_id), str(session_id)).as_posix()


class UploadStorage:
    """File operations on the uploads root. One instance per process."""

    def __init__(self, root: str | os.PathLike[str] | None = None) -> None:
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
        """Write via '<name>.part' + atomic rename. Creates the folders.
        Raises FileExistsError rather than overwrite (document ids are unique)."""
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
        return self.resolve(rel).read_bytes()

    def read_text(self, rel: str) -> str:
        """The stored text (UTF-8, a BOM from a TXT upload is dropped).
        Callers apply clean_text() before hashing or sending it to a model."""
        return decode_txt(self.read_bytes(rel))

    # --- delete (never raises) ---

    def delete(self, rel: str) -> bool:
        """Remove one text file. Missing is fine. Returns False (and logs)
        if it could not be removed."""
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
        """Remove a session's folder if it is empty (leftover '.part' files
        from a crash are removed first); the user's folder is kept.
        Returns False (and logs) if the folder is still there afterwards."""
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
