"""Request/response models for sessions, documents and analyses (ADR-004 §7.8)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from uuid import uuid4

import pytest
from pydantic import ValidationError

from aity_web.models import Document, Session, SustainabilityMetrics
from aity_web.models.api import (
    AnalysisCreate,
    AnalysisResultOut,
    DocumentOut,
    SessionCreate,
    SessionOut,
    SessionUpdate,
)
from aity_web.repositories.base import UNSET
from test_repository_contract import build_analysis

T = datetime(2026, 11, 26, 9, 0, tzinfo=timezone.utc)


def _doc() -> Document:
    uid, sid, did = uuid4(), uuid4(), uuid4()
    return Document(
        id=did, session_id=sid, user_id=uid, original_filename="paper.pdf", source_type="pdf",
        text_path=f"{uid}/{sid}/{did}_{uid}_20261126T090000Z.txt",
        text_sha256="a" * 64, char_count=1200, uploaded_at=T,
    )


# --- sessions ---


def test_session_create():
    body = SessionCreate.model_validate({"name": "  SMS pilot ", "description": "  "})
    assert body.name == "SMS pilot" and body.description is None
    with pytest.raises(ValidationError):
        SessionCreate.model_validate({"name": ""})
    with pytest.raises(ValidationError):
        SessionCreate.model_validate({"name": "x", "owner": "bob"})


@pytest.mark.parametrize(
    "body, changes",
    [
        ({}, {}),
        ({"name": "New"}, {"name": "New"}),
        ({"description": None}, {"description": None}),
        ({"description": " "}, {"description": None}),
        ({"name": "N", "description": "d"}, {"name": "N", "description": "d"}),
    ],
)
def test_session_update_changes_only_what_was_sent(body, changes):
    assert SessionUpdate.model_validate(body).changes() == changes


def test_session_update_changes_fit_the_service_signature():
    """Fields not in changes() fall back to UNSET in SessionService.update()."""
    kwargs = {"name": UNSET, "description": UNSET, **SessionUpdate.model_validate({"name": "N"}).changes()}
    assert kwargs == {"name": "N", "description": UNSET}


def test_session_update_refuses_null_name():
    with pytest.raises(ValidationError, match="name cannot be null"):
        SessionUpdate.model_validate({"name": None})


def test_session_out_hides_the_owner():
    s = Session(id=uuid4(), user_id=uuid4(), name="S", created_at=T, updated_at=T)
    out = SessionOut.from_domain(s, document_count=3)
    assert out.document_count == 3 and "user_id" not in out.model_dump()


# --- documents ---


def test_document_out_hides_paths_and_hashes():
    out = DocumentOut.from_domain(_doc()).model_dump()
    assert set(out) == {"id", "session_id", "original_filename", "source_type", "char_count", "uploaded_at"}


# --- analyses ---


def test_analysis_create():
    body = AnalysisCreate.model_validate({"provider": "gemini", "model": " gemini-3.5-flash-lite "})
    assert body.model == "gemini-3.5-flash-lite"
    for bad in ({"provider": "bard", "model": "x"}, {"provider": "gemini", "model": ""},
                {"provider": "gemini"}):
        with pytest.raises(ValidationError):
            AnalysisCreate.model_validate(bad)


def test_analysis_create_refuses_a_prompt_variant():
    with pytest.raises(ValidationError):
        AnalysisCreate.model_validate(
            {"provider": "gemini", "model": "m", "prompt_variant": "default_analysis"})


def test_analysis_result_out_hides_server_fields():
    metrics = SustainabilityMetrics(
        energy_kwh=1e-4, co2e_kg=5e-5, estimate_status="ok", estimation_source="ecologits",
        latency_seconds=2.0, warnings=["model-arch-not-released"], errors=[],
    )
    result = build_analysis(_doc(), sustainability=metrics)
    out = AnalysisResultOut.from_domain(result)
    data = json.loads(out.model_dump_json())

    for hidden in ("user_id", "content_hash", "cache_key_version", "served_from_analysis_id",
                   "schema_version"):
        assert hidden not in data
    assert "warnings" not in data["sustainability"]
    assert "model-arch-not-released" not in out.model_dump_json()
    assert data["cache_hit"] is False
    assert data["payload"]["summary"] == "A short summary."
    assert data["sustainability"]["energy_kwh"] == pytest.approx(1e-4)
    assert AnalysisResultOut.from_domain(result, cache_hit=True).cache_hit is True


def test_failed_result_out():
    result = build_analysis(_doc(), status="failed", payload=None, token_usage=None,
                            error="provider call failed (TimeoutError)")
    out = AnalysisResultOut.from_domain(result)
    assert out.payload is None and out.token_usage is None and out.status == "failed"
