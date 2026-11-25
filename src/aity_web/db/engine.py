"""Async engine and session factory (SQLAlchemy 2.0 asyncio + asyncpg).

The URL comes from the argument, else from AITY_DATABASE_URL. A plain
``postgresql://`` or ``postgres://`` URL is switched to the asyncpg driver,
so the same value can be shared with psql-style tools; any other driver is
refused.
"""

from __future__ import annotations

import os
from typing import Any

from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

DATABASE_URL_ENV = "AITY_DATABASE_URL"
_DRIVER = "postgresql+asyncpg"


class DatabaseConfigError(RuntimeError):
    """No database URL, or one AITY cannot use."""


def database_url(url: str | None = None, *, env: str = DATABASE_URL_ENV) -> str:
    """Return the asyncpg URL: argument -> $env. Raises DatabaseConfigError."""
    raw = url if url is not None else os.environ.get(env, "")
    raw = raw.strip()
    if not raw:
        raise DatabaseConfigError(f"no database URL: set {env}")
    try:
        parsed = make_url(raw)
    except ArgumentError as exc:
        raise DatabaseConfigError(f"{env} is not a valid database URL") from exc
    if parsed.drivername in ("postgresql", "postgres", _DRIVER):
        parsed = parsed.set(drivername=_DRIVER)
    else:
        raise DatabaseConfigError(
            f"unsupported database driver {parsed.drivername!r}: AITY needs PostgreSQL ({_DRIVER})"
        )
    if not parsed.database:
        raise DatabaseConfigError(f"{env} names no database")
    return parsed.render_as_string(hide_password=False)


def make_engine(url: str | None = None, **kwargs: Any) -> AsyncEngine:
    """Create the async engine. Extra kwargs go to create_async_engine
    (tests pass poolclass=NullPool). Does not connect yet."""
    options: dict[str, Any] = {
        "pool_pre_ping": True,
        "connect_args": {"server_settings": {"timezone": "UTC", "application_name": "aity_web"}},
    }
    options.update(kwargs)
    return create_async_engine(database_url(url), **options)


def make_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """One AsyncSession per request or job (ADR-004 §3.5).
    expire_on_commit=False: the unit of work keeps working after commit()."""
    return async_sessionmaker(engine, expire_on_commit=False, autoflush=False)
