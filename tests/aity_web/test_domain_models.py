"""Domain models (ADR-004 §7.1): User, Session, Document, AnalysisResult, UserApiKey."""

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from pydantic import ValidationError

from aity_web.llm.types import PROVIDER_NAMES
from aity_web.models import (
    ANALYSIS_RESULT_SCHEMA_VERSION,
    CLOUD_PROVIDER_NAMES,
    AnalysisResult,
    Document,
    LLMConfig,
    Session,
    SustainabilityMetrics,
    User,
    UserApiKey,
)

T0 = datetime(2026, 11, 25, 10, 15, tzinfo=timezone.utc)
T1 = T0 + timedelta(seconds=2)
HEX = "a" * 64
UID, SID, DID = uuid4(), uuid4(), uuid4()

PAYLOAD = {
    "summary": "A short summary.",
    "keywords": ["quantum software", "libraries"],
    "topics": [{"label": "Software Engineering", "description": "Why it is here."}],
}
USAGE = {"prompt_tokens": 100, "output_tokens": 50}
SUSTAIN = SustainabilityMetrics(
    energy_kwh=1e-5, co2e_kg=4e-6, estimate_status="ok",
    estimation_source="ecologits", latency_seconds=2.0,
)


def frozen_check(model, field, value):
    with pytest.raises(ValidationError):
        setattr(model, field, value)


# --- User -----------------------------------------------------------------------

def user(**kw):
    return User(**{"id": UID, "username": "Mintu42", "password_hash": "$2b$12$x",
                   "created_at": T0, **kw})


def test_user_valid_and_hash_hidden_from_repr():
    u = user()
    assert u.username == "Mintu42"
    assert "$2b$" not in repr(u)
    frozen_check(u, "username", "other")


@pytest.mark.parametrize("name", ["", "mintu ngo", "mintu_ngo", "tù", "a" * 65])
def test_user_rejects_bad_username(name):
    with pytest.raises(ValidationError):
        user(username=name)


def test_user_rejects_naive_datetime():
    with pytest.raises(ValidationError):
        user(created_at=datetime(2026, 11, 25, 10, 15))


def test_datetimes_normalised_to_utc():
    helsinki = timezone(timedelta(hours=2))
    u = user(created_at=datetime(2026, 11, 25, 12, 15, tzinfo=helsinki))
    assert u.created_at == T0
    assert u.created_at.utcoffset() == timedelta(0)


# --- Session ----------------------------------------------------------------------

def session(**kw):
    return Session(**{"id": SID, "user_id": UID, "name": " SMS pilot ",
                      "created_at": T0, "updated_at": T0, **kw})


def test_session_strips_and_defaults():
    s = session()
    assert s.name == "SMS pilot"
    assert s.description is None


def test_session_has_no_llm_config():
    assert not {"provider", "model", "prompt_variant", "llm_config"} & set(Session.model_fields)
    with pytest.raises(ValidationError):
        session(provider="openai")


@pytest.mark.parametrize("desc, expected", [("  ", None), (" notes ", "notes"), (None, None)])
def test_session_description(desc, expected):
    assert session(description=desc).description == expected


@pytest.mark.parametrize(
    "kw",
    [{"name": "   "}, {"name": "x" * 201}, {"description": "x" * 2001},
     {"updated_at": T0 - timedelta(seconds=1)}],
)
def test_session_rejects_invalid(kw):
    with pytest.raises(ValidationError):
        session(**kw)


# --- Document ---------------------------------------------------------------------

def text_path(user_id=UID, session_id=SID, doc_id=DID):
    return f"{user_id}/{session_id}/{doc_id}_{user_id}_20261125T101500Z.txt"


def document(**kw):
    return Document(**{"id": DID, "session_id": SID, "user_id": UID,
                       "original_filename": "qsl-survey.pdf", "source_type": "pdf",
                       "text_path": text_path(), "text_sha256": HEX, "char_count": 1234,
                       "uploaded_at": T0, **kw})


def test_document_valid():
    d = document()
    assert d.source_type == "pdf"
    frozen_check(d, "text_path", "x.txt")


@pytest.mark.parametrize(
    "path",
    [
        f"/{text_path()}",                                 # absolute
        text_path().replace("/", "\\"),                     # Windows separators
        f"{UID}/{SID}/../{DID}_x.txt",                      # traversal
        text_path()[:-4] + ".pdf",                          # not .txt
        text_path(user_id=uuid4()),                         # another user's folder
        text_path(session_id=uuid4()),                      # another session's folder
        text_path(doc_id=uuid4()),                          # another document's name
        f"{UID}/{SID}/sub/{DID}_x.txt",                     # nested folder
        f"{UID}//{SID}/{DID}_x.txt",                        # not normalised
    ],
)
def test_document_rejects_unsafe_or_foreign_path(path):
    with pytest.raises(ValidationError):
        document(text_path=path)


@pytest.mark.parametrize(
    "kw",
    [{"source_type": "doc"}, {"text_sha256": "A" * 64}, {"text_sha256": "a" * 63},
     {"char_count": 0}, {"original_filename": "x" * 256}, {"original_filename": " "}],
)
def test_document_rejects_invalid(kw):
    with pytest.raises(ValidationError):
        document(**kw)


# --- UserApiKey -------------------------------------------------------------------

def api_key(**kw):
    return UserApiKey(**{"user_id": UID, "provider": "openai", "ciphertext": b"gAAAAA-secret",
                         "created_at": T0, "updated_at": T1, **kw})


def test_api_key_valid_and_ciphertext_hidden_from_repr():
    k = api_key()
    assert "secret" not in repr(k)


def test_cloud_providers_are_the_byok_ones():
    assert set(CLOUD_PROVIDER_NAMES) == {
        p for p in PROVIDER_NAMES if LLMConfig(provider=p, model="m").needs_api_key
    }


@pytest.mark.parametrize(
    "kw", [{"provider": "ollama"}, {"ciphertext": b""}, {"updated_at": T0 - timedelta(1)}]
)
def test_api_key_rejects_invalid(kw):
    with pytest.raises(ValidationError):
        api_key(**kw)


# --- AnalysisResult ---------------------------------------------------------------

def result(**kw):
    data = {
        "analysis_id": uuid4(), "session_id": SID, "document_id": DID, "user_id": UID,
        "provider": "gemini", "model": "gemini-3.5-flash-lite",
        "prompt_variant": "default_analysis",
        "payload": PAYLOAD, "token_usage": USAGE, "sustainability": SUSTAIN,
        "content_hash": HEX, "cache_key_version": 1,
        "status": "ok", "created_at": T0, "finished_at": T1,
    }
    data.update(kw)
    return AnalysisResult(**data)


def test_ok_result_from_plain_dicts():
    # As a repository builds it from a row: JSONB columns arrive as dicts.
    r = result()
    assert r.schema_version == ANALYSIS_RESULT_SCHEMA_VERSION == 1
    assert r.payload.topics[0].label == "Software Engineering"
    assert r.token_usage.total_tokens == 150
    assert r.config == LLMConfig(provider="gemini", model="gemini-3.5-flash-lite")
    assert not r.is_cache_copy
    assert not r.truncated
    frozen_check(r, "status", "failed")


def test_json_round_trip():
    r = result(status="partial", error="keywords: 11 (expected 5–10)")
    assert AnalysisResult.model_validate_json(r.model_dump_json()) == r


def test_from_attributes():
    class Row:  # stands in for an ORM object
        pass

    row = Row()
    for k, v in result().model_dump().items():
        setattr(row, k, v)
    assert AnalysisResult.model_validate(row).payload.summary == "A short summary."


def test_failed_rows_have_no_payload_and_may_lack_usage():
    r = result(status="failed", payload=None, token_usage=None, error="TimeoutError",
               sustainability=SustainabilityMetrics.unavailable(120.0, ["no response"]))
    assert r.payload is None and r.token_usage is None


def test_failed_row_may_keep_usage():
    result(status="failed", payload=None, error="finish_reason=length")


def test_cache_copy_row():
    original = result()
    copy = result(session_id=uuid4(), served_from_analysis_id=original.analysis_id)
    assert copy.is_cache_copy


@pytest.mark.parametrize(
    "kw",
    [
        {"status": "ok", "error": "should not be here"},
        {"status": "ok", "payload": None},
        {"status": "ok", "token_usage": None},
        {"status": "partial"},                                     # partial needs its issues
        {"status": "partial", "error": "  "},
        {"status": "failed", "error": "x"},                        # payload still set
        {"status": "failed", "payload": None},                     # error missing
        {"status": "partial", "error": "x", "served_from_analysis_id": uuid4()},
        {"truncated": True},                                       # warning missing
        {"truncation_warning": "dropped the tail"},                # flag missing
        {"finished_at": T0 - timedelta(seconds=1)},
        {"schema_version": 2},                                     # unknown version refused
        {"cache_key_version": 0},
        {"content_hash": "not-a-hash"},
        {"provider": "berts"},
        {"model": "  "},
        {"prompt_variant": ""},
        {"cache_hit": True},                                       # API-only field
        {"payload": {**PAYLOAD, "notes": "x"}},                    # strict payload
    ],
)
def test_result_rejects_inconsistent_rows(kw):
    with pytest.raises(ValidationError):
        result(**kw)


def test_row_cannot_be_served_from_itself():
    aid = uuid4()
    with pytest.raises(ValidationError):
        result(analysis_id=aid, served_from_analysis_id=aid)


def test_truncated_row():
    r = result(truncated=True, truncation_warning="input cut at 16384 tokens")
    assert r.truncated
