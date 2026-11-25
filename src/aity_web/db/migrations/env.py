"""Alembic environment for aity_web (async, asyncpg).

URL: the config's sqlalchemy.url when a caller set it (tests, db.migrate),
otherwise AITY_DATABASE_URL. alembic.ini never holds a URL.

Autogenerate compares aity_web.db.tables.metadata with the database, column
types and server defaults included; it does not read the Pydantic models.
"""

from __future__ import annotations

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection

from aity_web.db.engine import database_url, make_engine
from aity_web.db.tables import metadata

config = context.config

if config.config_file_name is not None and config.attributes.get("configure_logger", True):
    fileConfig(config.config_file_name)

target_metadata = metadata


def _url() -> str:
    return database_url(config.get_main_option("sqlalchemy.url") or None)


def run_migrations_offline() -> None:
    """`alembic upgrade head --sql`: print the SQL instead of running it."""
    context.configure(
        url=_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        transaction_per_migration=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def _run(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        transaction_per_migration=True,
        # autogenerate: also report column type and server default changes
        compare_type=True,
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()


async def _run_online() -> None:
    engine = make_engine(_url(), poolclass=pool.NullPool)
    try:
        async with engine.connect() as connection:
            await connection.run_sync(_run)
    finally:
        await engine.dispose()


def run_migrations_online() -> None:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(_run_online())
        return
    # Called from async code (pytest-asyncio, an app startup hook): asyncio.run
    # cannot nest, so run the migration on its own loop in a worker thread.
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=1) as pool_:
        pool_.submit(asyncio.run, _run_online()).result()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
