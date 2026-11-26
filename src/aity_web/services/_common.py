"""Small types shared by the W3 services."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone

from aity_web.repositories.base import UnitOfWork

UowFactory = Callable[[], UnitOfWork]
"""What services take: a fresh UnitOfWork per call (InMemoryStore.uow_factory
in tests, make_uow_factory(engine) in the backend)."""

Clock = Callable[[], datetime]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)
