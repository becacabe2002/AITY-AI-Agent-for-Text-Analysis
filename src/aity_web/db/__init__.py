"""Database layer: async engine, table definitions, SQLAlchemy unit of work and migrations."""

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
