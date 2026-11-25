"""Turn an uploaded PDF, DOCX or TXT into text - ADR-004 §7.5.

* PDF  - pypdf, the text of each page, pages joined by a blank line.
* DOCX - python-docx, paragraphs and table cells in document order
         (a table row becomes one line, its cells separated by a tab).
* TXT  - decoded as UTF-8 (a BOM is accepted); the stored file is an exact
         copy of the uploaded bytes.

extract() works on bytes held in memory: the original PDF or DOCX is never
written to disk. It is synchronous and CPU-bound (a large PDF can take a
while), so async callers run it in a worker thread (asyncio.to_thread).

Every rejection is a DocumentRejectedError subclass whose message is safe to
show to the user (no file content, no library internals). The W5 upload
route maps them to HTTP 415 (unsupported type) / 422 (the rest).
The upload size limit is set with the route in W5.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from pathlib import PureWindowsPath
from typing import get_args

from aity_web.models.document import SourceType
from aity_web.services.cache_key import clean_text, text_sha256

log = logging.getLogger(__name__)

SOURCE_TYPES: tuple[str, ...] = get_args(SourceType)


# --- errors ---


class DocumentRejectedError(ValueError):
    """The upload cannot become a document. The message is user-safe."""


class UnsupportedFileTypeError(DocumentRejectedError):
    """Not a .pdf, .docx or .txt file, or the content does not match it."""


class UnreadableDocumentError(DocumentRejectedError):
    """The file is damaged or not what its extension says."""


class EncryptedPdfError(DocumentRejectedError):
    """A password-protected PDF."""


class NoTextLayerError(DocumentRejectedError):
    """A PDF without extractable text (a scan); OCR is not in v1."""


class NotUtf8Error(DocumentRejectedError):
    """A .txt file that is not valid UTF-8."""


class EmptyDocumentError(DocumentRejectedError):
    """Nothing left after clean_text()."""


# --- result ---


@dataclass(frozen=True, repr=False)
class ExtractedText:
    """
    * source_type - pdf / docx / txt
    * stored_bytes - what goes into the .txt file: the extracted text in
      UTF-8 for PDF/DOCX, the uploaded bytes unchanged for TXT
    * cleaned     - clean_text() of the text (what a model will see)
    * text_sha256 - SHA-256 of `cleaned`
    * char_count  - len(cleaned), at least 1
    """

    source_type: SourceType
    stored_bytes: bytes
    cleaned: str
    text_sha256: str

    @property
    def char_count(self) -> int:
        return len(self.cleaned)

    def __repr__(self) -> str:  # never print document text into logs
        return f"ExtractedText(source_type={self.source_type!r}, char_count={self.char_count})"


# --- public API ---


def source_type_for(filename: str) -> SourceType:
    """pdf / docx / txt from the file extension (case-insensitive).
    UnsupportedFileTypeError for anything else."""
    suffix = PureWindowsPath(filename).suffix.lower().lstrip(".")  # handles / and \
    if suffix not in SOURCE_TYPES:
        raise UnsupportedFileTypeError(
            "Only PDF, DOCX and TXT files can be uploaded."
        )
    return suffix  # type: ignore[return-value]


def extract(filename: str, data: bytes) -> ExtractedText:
    """Extract the text of an upload. See the module docstring for the rules."""
    source_type = source_type_for(filename)
    if source_type == "pdf":
        raw = _pdf_text(data)
        stored = raw.encode("utf-8")
    elif source_type == "docx":
        raw = _docx_text(data)
        stored = raw.encode("utf-8")
    else:
        raw = decode_txt(data)
        stored = bytes(data)

    cleaned = clean_text(raw)
    if not cleaned:
        if source_type == "pdf":
            raise NoTextLayerError(
                "This PDF has no text layer (for example a scanned document). "
                "Text recognition (OCR) is not supported."
            )
        raise EmptyDocumentError("The file contains no text.")
    return ExtractedText(
        source_type=source_type,
        stored_bytes=stored,
        cleaned=cleaned,
        text_sha256=text_sha256(cleaned),
    )


def decode_txt(data: bytes) -> str:
    """UTF-8 with an optional BOM. Also used to read stored .txt files back."""
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise NotUtf8Error("The text file is not UTF-8 encoded.") from None


# --- PDF ---


def _pdf_text(data: bytes) -> str:
    from pypdf import PdfReader
    from pypdf.errors import PyPdfError

    if not data.lstrip()[:5] == b"%PDF-":
        raise UnreadableDocumentError("The file is not a valid PDF.")
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            # PDFs with only an owner password (copy/print restrictions)
            # open with an empty user password; anything else is refused.
            try:
                opened = reader.decrypt("")
            except Exception:  # wrong password, or an unsupported cipher
                opened = 0
            if not opened:
                raise EncryptedPdfError(
                    "This PDF is password-protected. Remove the password and upload it again."
                )
        pages = [page.extract_text() or "" for page in reader.pages]
    except DocumentRejectedError:
        raise
    except (PyPdfError, ValueError, KeyError, TypeError, AttributeError, IndexError, OSError) as exc:
        log.info("PDF extraction failed: %s", type(exc).__name__)
        raise UnreadableDocumentError("The PDF could not be read; it may be damaged.") from None
    return "\n\n".join(pages)


# --- DOCX ---


def _docx_text(data: bytes) -> str:
    import zipfile

    import docx
    from docx.opc.exceptions import PackageNotFoundError

    if data[:4] != b"PK\x03\x04":
        raise UnreadableDocumentError("The file is not a valid DOCX document.")
    try:
        document = docx.Document(io.BytesIO(data))
        lines = list(_block_lines(document))
    except (PackageNotFoundError, zipfile.BadZipFile, KeyError, ValueError) as exc:
        log.info("DOCX extraction failed: %s", type(exc).__name__)
        raise UnreadableDocumentError(
            "The DOCX file could not be read; it may be damaged."
        ) from None
    return "\n".join(lines)


def _block_lines(container):
    """Paragraphs and tables of a document body or table cell, in order."""
    from docx.table import Table

    for block in container.iter_inner_content():
        if isinstance(block, Table):
            yield from _table_lines(block)
        else:
            yield block.text


def _table_lines(table):
    # Merged cells are returned once per grid position they cover (across
    # columns and rows); each underlying cell is emitted only the first time.
    # References are kept, so the identity check is reliable.
    seen: list = []
    for row in table.rows:
        cells = []
        for cell in row.cells:
            if any(cell._tc is tc for tc in seen):
                continue
            seen.append(cell._tc)
            cells.append("\n".join(_block_lines(cell)).strip())
        if cells:
            yield "\t".join(cells)


__all__ = [
    "DocumentRejectedError",
    "EmptyDocumentError",
    "EncryptedPdfError",
    "ExtractedText",
    "NoTextLayerError",
    "NotUtf8Error",
    "SOURCE_TYPES",
    "UnreadableDocumentError",
    "UnsupportedFileTypeError",
    "decode_txt",
    "extract",
    "source_type_for",
]
