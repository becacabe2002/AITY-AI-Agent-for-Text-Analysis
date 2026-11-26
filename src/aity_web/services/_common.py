"""Small types and helpers shared by the services."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone

from aity_web.repositories.base import UnitOfWork

UowFactory = Callable[[], UnitOfWork]
"""Factory that returns a fresh UnitOfWork per call."""

Clock = Callable[[], datetime]


def utc_now() -> datetime:
    """Return the current time in UTC."""
    return datetime.now(timezone.utc)
