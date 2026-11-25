"""Table definitions (SQLAlchemy Core) for the W3 schema - ADR-004 §4.

This module is the source of truth for the schema, and where every schema
change starts. The repositories build their queries from it, and Alembic
autogenerate drafts the migrations by comparing ``metadata`` with the
database; each reviewed migration must produce exactly these tables.
tests/db/test_migrations.py compares the migrated database with
``metadata`` and fails on any difference.

Autogenerate does not read the Pydantic models in aity_web.models: a new,
renamed or removed top-level field there must be mirrored here by hand
(test_db_layer.py checks the analysis_results column names). Changes inside
the JSONB blobs (payload, token_usage, sustainability) need no table change.

Ownership is enforced twice: every repository query filters by user_id, and
composite foreign keys make it impossible to store a document in another
user's session, or an analysis whose (document, session, user) does not
match its document. The repositories raise ValueError / NotFoundError
before the database would refuse.

W4 adds user_api_keys and prompt_variants; W8 adds comparisons and syntheses.
"""

from __future__ import annotations

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    MetaData,
    SmallInteger,
    String,
    Table,
    Text,
    UniqueConstraint,
    Uuid,
)
from sqlalchemy.dialects.postgresql import JSONB

from aity_web.llm.types import PROVIDER_NAMES
from aity_web.models.document import FILENAME_MAX
from aity_web.models.session import SESSION_DESCRIPTION_MAX, SESSION_NAME_MAX
from aity_web.models.user import USERNAME_MAX

NAME_MAX = 200  # models._types.Name: model ids, prompt variant names

metadata = MetaData()


def _tz() -> DateTime:
    return DateTime(timezone=True)


def _in(column: str, values: tuple[str, ...]) -> str:
    return f"{column} IN ({', '.join(repr(v) for v in values)})"


users = Table(
    "users",
    metadata,
    Column("id", Uuid, primary_key=True),
    Column("username", String(USERNAME_MAX), nullable=False),
    Column("password_hash", Text, nullable=False),
    Column("created_at", _tz(), nullable=False),
    UniqueConstraint("username", name="uq_users_username"),  # case-sensitive
    CheckConstraint("username ~ '^[A-Za-z0-9]+$'", name="ck_users_username_chars"),
)

sessions = Table(
    "sessions",
    metadata,
    Column("id", Uuid, primary_key=True),
    Column(
        "user_id", Uuid,
        ForeignKey("users.id", name="fk_sessions_user", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("name", String(SESSION_NAME_MAX), nullable=False),
    Column("description", String(SESSION_DESCRIPTION_MAX), nullable=True),
    Column("created_at", _tz(), nullable=False),
    Column("updated_at", _tz(), nullable=False),
    # target of the composite ownership keys below
    UniqueConstraint("id", "user_id", name="uq_sessions_id_user"),
    CheckConstraint("updated_at >= created_at", name="ck_sessions_times"),
)
Index("ix_sessions_user_updated", sessions.c.user_id, sessions.c.updated_at.desc())

documents = Table(
    "documents",
    metadata,
    Column("id", Uuid, primary_key=True),
    Column("session_id", Uuid, nullable=False),
    Column(
        "user_id", Uuid,
        ForeignKey("users.id", name="fk_documents_user", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("original_filename", String(FILENAME_MAX), nullable=False),
    Column("source_type", String(8), nullable=False),
    Column("text_path", Text, nullable=False),
    Column("text_sha256", String(64), nullable=False),
    Column("char_count", Integer, nullable=False),
    Column("uploaded_at", _tz(), nullable=False),
    ForeignKeyConstraint(
        ["session_id", "user_id"], ["sessions.id", "sessions.user_id"],
        name="fk_documents_session", ondelete="CASCADE",
    ),
    UniqueConstraint("text_path", name="uq_documents_text_path"),
    UniqueConstraint("id", "session_id", "user_id", name="uq_documents_id_session_user"),
    CheckConstraint(_in("source_type", ("pdf", "docx", "txt")), name="ck_documents_source_type"),
    CheckConstraint("text_sha256 ~ '^[0-9a-f]{64}$'", name="ck_documents_text_sha256"),
    CheckConstraint("char_count >= 1", name="ck_documents_char_count"),
)
Index(
    "ix_documents_user_session_uploaded",
    documents.c.user_id, documents.c.session_id, documents.c.uploaded_at.desc(),
)

analysis_results = Table(
    "analysis_results",
    metadata,
    Column("analysis_id", Uuid, primary_key=True),
    Column("schema_version", SmallInteger, nullable=False),
    Column("session_id", Uuid, nullable=False),
    Column("document_id", Uuid, nullable=False),
    Column(
        "user_id", Uuid,
        ForeignKey("users.id", name="fk_analysis_results_user", ondelete="CASCADE"),
        nullable=False,
    ),
    # provenance
    Column("provider", String(32), nullable=False),
    Column("model", String(NAME_MAX), nullable=False),
    Column("prompt_variant", String(NAME_MAX), nullable=False),
    # blobs (read whole, never queried in v1). none_as_null: Python None is
    # SQL NULL, not the JSON value null (the CHECKs below test IS NULL).
    Column("payload", JSONB(none_as_null=True), nullable=True),
    Column("token_usage", JSONB(none_as_null=True), nullable=True),
    Column("sustainability", JSONB(none_as_null=True), nullable=False),
    # cache key (lookup arrives in W6, ADR-008)
    Column("content_hash", String(64), nullable=False),
    Column("cache_key_version", Integer, nullable=False),
    Column(
        "served_from_analysis_id", Uuid,
        ForeignKey(
            "analysis_results.analysis_id",
            name="fk_analysis_results_served_from", ondelete="SET NULL",
        ),
        nullable=True,
    ),
    # input handling (ADR-007)
    Column("truncated", Boolean, nullable=False),
    Column("truncation_warning", Text, nullable=True),
    # audit
    Column("status", String(8), nullable=False),
    Column("error", Text, nullable=True),
    Column("created_at", _tz(), nullable=False),
    Column("finished_at", _tz(), nullable=False),
    ForeignKeyConstraint(
        ["document_id", "session_id", "user_id"],
        ["documents.id", "documents.session_id", "documents.user_id"],
        name="fk_analysis_results_document", ondelete="CASCADE",
    ),
    ForeignKeyConstraint(
        ["session_id", "user_id"], ["sessions.id", "sessions.user_id"],
        name="fk_analysis_results_session", ondelete="CASCADE",
    ),
    CheckConstraint("schema_version >= 1", name="ck_analysis_results_schema_version"),
    CheckConstraint(_in("provider", PROVIDER_NAMES), name="ck_analysis_results_provider"),
    CheckConstraint(_in("status", ("ok", "partial", "failed")), name="ck_analysis_results_status"),
    CheckConstraint("content_hash ~ '^[0-9a-f]{64}$'", name="ck_analysis_results_content_hash"),
    CheckConstraint("cache_key_version >= 1", name="ck_analysis_results_cache_key_version"),
    CheckConstraint("(status = 'failed') = (payload IS NULL)", name="ck_analysis_results_payload"),
    CheckConstraint(
        "status = 'failed' OR token_usage IS NOT NULL", name="ck_analysis_results_token_usage"
    ),
    CheckConstraint(
        "served_from_analysis_id IS NULL OR status = 'ok'", name="ck_analysis_results_served_ok"
    ),
    CheckConstraint(
        "truncated = (truncation_warning IS NOT NULL)", name="ck_analysis_results_truncation"
    ),
    CheckConstraint("finished_at >= created_at", name="ck_analysis_results_times"),
)
# session feed
Index(
    "ix_analysis_results_user_session_created",
    analysis_results.c.user_id, analysis_results.c.session_id, analysis_results.c.created_at.desc(),
)
# one document's analyses
Index(
    "ix_analysis_results_user_document_created",
    analysis_results.c.user_id, analysis_results.c.document_id, analysis_results.c.created_at.desc(),
)
# W6 cache lookup; cache_key_version and status are checked inline
Index(
    "ix_analysis_results_cache_lookup",
    analysis_results.c.user_id, analysis_results.c.content_hash, analysis_results.c.provider,
    analysis_results.c.model, analysis_results.c.prompt_variant,
)
# deleting a result sets served_from_analysis_id = NULL on its copies
Index(
    "ix_analysis_results_served_from",
    analysis_results.c.served_from_analysis_id,
    postgresql_where=analysis_results.c.served_from_analysis_id.isnot(None),
)
