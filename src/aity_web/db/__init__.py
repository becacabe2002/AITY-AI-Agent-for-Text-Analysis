"""Database layer (ADR-004): async engine and session factory, table
definitions, the SQLAlchemy unit of work and the Alembic migrations.

    engine = make_engine()                    # AITY_DATABASE_URL
    uow_factory = make_uow_factory(make_session_factory(engine))
    async with uow_factory() as uow: ...

Services never import this package directly: they receive a unit-of-work
factory (repositories.base.UnitOfWork), so tests can pass the in-memory one.
"""

from aity_web.db.engine import (
    DATABASE_URL_ENV,
    DatabaseConfigError,
    database_url,
    make_engine,
    make_session_factory,
)
from aity_web.db.tables import metadata
from aity_web.db.uow import SqlAlchemyUnitOfWork, make_uow_factory

__all__ = [
    "DATABASE_URL_ENV",
    "DatabaseConfigError",
    "SqlAlchemyUnitOfWork",
    "database_url",
    "make_engine",
    "make_session_factory",
    "make_uow_factory",
    "metadata",
]
