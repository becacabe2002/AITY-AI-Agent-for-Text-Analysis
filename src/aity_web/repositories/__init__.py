"""
Services depend on the ABCs in base.py and receive a UnitOfWork factory.
memory.py holds in-memory fakes used only in tests; the SQLAlchemy
implementations (Postgres) live in repositories/sqlalchemy/ (W3 Group B).
"""

from aity_web.repositories.base import (
    DEFAULT_PAGE_SIZE,
    UNSET,
    AnalysisRepository,
    ConflictError,
    CredentialRepository,
    DocumentRepository,
    NotFoundError,
    RepositoryError,
    SessionRepository,
    UnitOfWork,
    Unset,
    UserRepository,
)

__all__ = [
    "DEFAULT_PAGE_SIZE",
    "UNSET",
    "AnalysisRepository",
    "ConflictError",
    "CredentialRepository",
    "DocumentRepository",
    "NotFoundError",
    "RepositoryError",
    "SessionRepository",
    "UnitOfWork",
    "Unset",
    "UserRepository",
]
