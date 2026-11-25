"""text_extraction and clean_text (ADR-004 §7.5, §7.6)."""

from __future__ import annotations

import hashlib

import pytest

from aity_web.services.cache_key import clean_text, text_sha256
from aity_web.services.text_extraction import (
    EmptyDocumentError,
    EncryptedPdfError,
    NoTextLayerError,
    NotUtf8Error,
    UnreadableDocumentError,
    UnsupportedFileTypeError,
    extract,
    source_type_for,
)

from _upload_fixtures import encrypt_pdf, make_docx, make_pdf

# --- clean_text ---


def test_clean_text_rules():
    raw = "  \r\nCafe\u0301 \t\r\nline two  \r\r\r\n\n\nend\t \n\n"
    assert clean_text(raw) == "Caf\u00e9\nline two\n\nend"


def test_clean_text_keeps_single_blank_lines_and_inner_spaces():
    assert clean_text("a  b\n\nc") == "a  b\n\nc"


def test_clean_text_whitespace_only_is_empty():
    assert clean_text(" \n\t\r\n ") == ""


def test_text_sha256_is_utf8_sha256():
    assert text_sha256("é") == hashlib.sha256("é".encode()).hexdigest()


# --- source type ---


@pytest.mark.parametrize(
    "name, expected",
    [("a.pdf", "pdf"), ("A.PDF", "pdf"), ("x.Docx", "docx"), ("notes.txt", "txt"),
     ("C:\\fakepath\\r.pdf", "pdf"), ("dir/sub/r.txt", "txt")],
)
def test_source_type_for(name, expected):
    assert source_type_for(name) == expected


@pytest.mark.parametrize("name", ["a.doc", "a.md", "a", "pdf", "a.pdf.exe", ".txt"])
def test_unsupported_types(name):
    with pytest.raises(UnsupportedFileTypeError):
        source_type_for(name)


# --- TXT ---


def test_txt_is_stored_as_exact_copy():
    data = "Title\r\n\r\n\r\nBody  \r\n".encode()
    out = extract("n.txt", data)
    assert out.source_type == "txt"
    assert out.stored_bytes == data
    assert out.cleaned == "Title\n\nBody"
    assert out.char_count == len("Title\n\nBody")
    assert out.text_sha256 == text_sha256("Title\n\nBody")


def test_txt_with_bom_is_accepted_and_kept():
    data = b"\xef\xbb\xbfhello"
    out = extract("n.txt", data)
    assert out.stored_bytes == data
    assert out.cleaned == "hello"


def test_txt_not_utf8_rejected():
    with pytest.raises(NotUtf8Error):
        extract("n.txt", "caf\u00e9".encode("latin-1"))


def test_empty_txt_rejected():
    with pytest.raises(EmptyDocumentError):
        extract("n.txt", b" \n\r\n ")


def test_repr_hides_text():
    out = extract("n.txt", b"secret content")
    assert "secret" not in repr(out)


# --- PDF ---


def test_pdf_pages_joined_by_blank_line():
    out = extract("p.pdf", make_pdf(["First page", "Second page"]))
    assert out.source_type == "pdf"
    assert out.cleaned == "First page\n\nSecond page"
    assert out.stored_bytes.decode("utf-8") == "First page\n\nSecond page"


def test_pdf_without_text_layer_rejected():
    with pytest.raises(NoTextLayerError):
        extract("scan.pdf", make_pdf(["", ""]))


def test_password_protected_pdf_rejected():
    data = encrypt_pdf(make_pdf(["secret"]), user_password="pw", owner_password="owner")
    with pytest.raises(EncryptedPdfError):
        extract("locked.pdf", data)


def test_owner_password_only_pdf_is_accepted():
    data = encrypt_pdf(make_pdf(["restricted copy"]), user_password="", owner_password="owner")
    assert extract("restricted.pdf", data).cleaned == "restricted copy"


@pytest.mark.parametrize("data", [b"not a pdf", b"%PDF-1.4\n garbage without objects", b""])
def test_broken_pdf_rejected(data):
    with pytest.raises(UnreadableDocumentError):
        extract("broken.pdf", data)


# --- DOCX ---


def test_docx_paragraphs_and_tables_in_order():
    def build(d):
        d.add_heading("Report", level=1)
        d.add_paragraph("Intro paragraph.")
        t = d.add_table(rows=2, cols=3)
        t.cell(0, 0).text, t.cell(0, 1).text, t.cell(0, 2).text = "A", "B", "C"
        t.cell(1, 0).merge(t.cell(1, 1)).text = "merged"
        t.cell(1, 2).text = "D"
        d.add_paragraph("Closing paragraph.")

    out = extract("r.docx", make_docx(build))
    assert out.source_type == "docx"
    assert out.cleaned == "Report\nIntro paragraph.\nA\tB\tC\nmerged\tD\nClosing paragraph."


def test_docx_vertical_merge_not_repeated():
    def build(d):
        t = d.add_table(rows=2, cols=2)
        t.cell(0, 0).merge(t.cell(1, 0)).text = "tall"
        t.cell(0, 1).text, t.cell(1, 1).text = "x", "y"

    assert extract("t.docx", make_docx(build)).cleaned == "tall\tx\ny"


def test_empty_docx_rejected():
    with pytest.raises(EmptyDocumentError):
        extract("e.docx", make_docx(lambda d: d.add_paragraph("   ")))


@pytest.mark.parametrize("data", [b"plain text", b"PK\x03\x04 not really a zip"])
def test_broken_docx_rejected(data):
    with pytest.raises(UnreadableDocumentError):
        extract("b.docx", data)


def test_pdf_named_as_docx_rejected():
    with pytest.raises(UnreadableDocumentError):
        extract("wrong.docx", make_pdf(["x"]))


def test_rejection_messages_do_not_leak_content():
    with pytest.raises(NotUtf8Error) as info:
        extract("n.txt", b"\xff secret")
    assert "secret" not in str(info.value)
