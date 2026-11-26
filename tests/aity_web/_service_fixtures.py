"""Shared helpers for the W3 service tests (in-memory unit of work)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone


class StepClock:
    """Starts at a fixed instant and moves one second per call, so ordering
    by timestamps is deterministic."""

    def __init__(self, start: datetime | None = None) -> None:
        self.now = start or datetime(2026, 11, 25, 10, 0, 0, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        self.now += timedelta(seconds=1)
        return self.now


async def add_user(store, username: str = "alice"):
    async with store.uow_factory() as uow:
        user = await uow.users.add(username=username, password_hash="x" * 60)
        await uow.commit()
    return user
