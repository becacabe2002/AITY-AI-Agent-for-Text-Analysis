"""aity_web.db without a database: URL handling, table layout, the unit of
work's guards and the AnalysisResult <-> row mapping. The behaviour on a real
PostgreSQL is covered by the contract suite and tests/db (AITY_DB_TESTS=1)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from aity_web.db import (
    DatabaseConfigError,
    SqlAlchemyUnitOfWork,
    database_url,
    make_engine,
    make_session_factory,
    make_uow_factory,
    metadata,
)
from aity_web.db.tables import analysis_results
from aity_web.models import AnalysisResult, SustainabilityMetrics
from aity_web.repositories.sqlalchemy.analyses import from_row, to_row

T0 = datetime(2026, 11, 25, 10, 0, tzinfo=timezone.utc)


# --- database_url -------------------------------------------------------------------


@pytest.mark.parametrize("raw", [
    "postgresql://aity:pw@localhost:5432/aity",
    "postgres://aity:pw@localhost:5432/aity",
    "postgresql+asyncpg://aity:pw@localhost:5432/aity",
    "  postgresql://aity:pw@localhost:5432/aity\n",
])
def test_database_url_switches_to_asyncpg(raw):
    assert database_url(raw) == "postgresql+asyncpg://aity:pw@localhost:5432/aity"


def test_database_url_from_environment(monkeypatch):
    monkeypatch.setenv("AITY_DATABASE_URL", "postgresql://u@db/aity")
    assert database_url() == "postgresql+asyncpg://u@db/aity"
    assert database_url("postgresql://u@other/x") == "postgresql+asyncpg://u@other/x"  # argument wins


@pytest.mark.parametrize("raw", [
    "", "   ", "sqlite+aiosqlite:///x.db", "postgresql+psycopg2://u@h/db",
    "mysql://u@h/db", "postgresql://u@h", "not a url",
])
def test_database_url_refuses(raw, monkeypatch):
    monkeypatch.delenv("AITY_DATABASE_URL", raising=False)
    with pytest.raises(DatabaseConfigError):
        database_url(raw)


def test_database_url_missing_names_the_variable(monkeypatch):
    monkeypatch.delenv("AITY_DATABASE_URL", raising=False)
    with pytest.raises(DatabaseConfigError, match="AITY_DATABASE_URL"):
        database_url()


def test_error_message_never_contains_the_password():
    with pytest.raises(DatabaseConfigError) as info:
        database_url("mysql://u:secret-pw@h/db")
    assert "secret-pw" not in str(info.value)


def test_make_engine_does_not_connect():
    engine = make_engine("postgresql://u:pw@nowhere.invalid/aity")
    assert engine.url.drivername == "postgresql+asyncpg"
    assert make_session_factory(engine).kw["expire_on_commit"] is False


# --- tables ---------------------------------------------------------------------


def test_w3_tables_only():
    assert set(metadata.tables) == {"users", "sessions", "documents", "analysis_results"}


def test_planned_indexes():
    indexes = {i.name for t in metadata.tables.values() for i in t.indexes}
    assert indexes == {
        "ix_sessions_user_updated",
        "ix_documents_user_session_uploaded",
        "ix_analysis_results_user_session_created",
        "ix_analysis_results_user_document_created",
        "ix_analysis_results_cache_lookup",
        "ix_analysis_results_served_from",
    }
    cache = next(i for i in analysis_results.indexes if i.name == "ix_analysis_results_cache_lookup")
    assert [c.name for c in cache.columns] == [
        "user_id", "content_hash", "provider", "model", "prompt_variant",
    ]


def test_analysis_columns_match_the_domain_model():
    assert set(analysis_results.c.keys()) == set(AnalysisResult.model_fields)


# --- unit of work guards ------------------------------------------------------------


def _uow() -> SqlAlchemyUnitOfWork:
    engine = make_engine("postgresql://u@nowhere.invalid/aity")
    return make_uow_factory(make_session_factory(engine))()


@pytest.mark.asyncio
async def test_repositories_unusable_outside_the_block():
    with pytest.raises(RuntimeError):
        await _uow().users.get_by_username(username="x")


@pytest.mark.asyncio
async def test_unit_of_work_cannot_be_entered_twice():
    uow = _uow()
    async with uow:
        with pytest.raises(RuntimeError):
            async with uow:
                pass


@pytest.mark.asyncio
async def test_credentials_wait_for_w4():
    async with _uow() as uow:
        with pytest.raises(NotImplementedError, match="W4"):
            await uow.credentials.get(user_id=uuid4(), provider="openai")


# --- AnalysisResult <-> row ------------------------------------------------------------


class _Row:
    def __init__(self, mapping):
        self._mapping = mapping


def _result(**kw) -> AnalysisResult:
    data = dict(
        analysis_id=uuid4(), session_id=uuid4(), document_id=uuid4(), user_id=uuid4(),
        provider="ollama", model="qwen3.5:4b", prompt_variant="default_analysis",
        payload={"summary": "S.", "keywords": ["k"], "topics": [{"label": "L", "description": "D"}]},
        token_usage={"prompt_tokens": 10, "output_tokens": 5},
        sustainability=SustainabilityMetrics.unavailable(latency_seconds=1.5),
        content_hash="c" * 64, cache_key_version=1, status="ok",
        created_at=T0, finished_at=T0 + timedelta(seconds=2),
    )
    data.update(kw)
    return AnalysisResult(**data)


@pytest.mark.parametrize("kw", [
    {},
    {"status": "failed", "payload": None, "token_usage": None, "error": "TimeoutError"},
    {"truncated": True, "truncation_warning": "cut at 16k tokens"},
])
def test_analysis_row_round_trip(kw):
    result = _result(**kw)
    row = to_row(result)
    assert set(row) == set(analysis_results.c.keys())
    for name in ("payload", "token_usage", "sustainability"):
        assert row[name] is None or isinstance(row[name], dict)  # JSON-ready
    assert from_row(_Row(row)) == result


def test_row_with_unknown_schema_version_is_refused():
    row = to_row(_result())
    row["schema_version"] = 2
    with pytest.raises(ValueError):
        from_row(_Row(row))
