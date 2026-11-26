"""Domain model of a session, a named workspace for one user's documents."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, BeforeValidator, StringConstraints, model_validator

from aity_web.models._types import DOMAIN_CONFIG, UtcDatetime

SESSION_NAME_MAX = 200
SESSION_DESCRIPTION_MAX = 2000


def _blank_to_none(value: object) -> object:
    """Turn a blank string into None."""
    if isinstance(value, str) and not value.strip():
        return None
    return value


SessionName = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=SESSION_NAME_MAX)
]
SessionDescription = Annotated[
    Annotated[str, StringConstraints(strip_whitespace=True, max_length=SESSION_DESCRIPTION_MAX)]
    | None,
    BeforeValidator(_blank_to_none),
]


class Session(BaseModel):
    """A named workspace for one user's documents; provider and model are chosen per analysis."""

    model_config = DOMAIN_CONFIG

    id: UUID
    user_id: UUID
    name: SessionName
    description: SessionDescription = None
    created_at: UtcDatetime
    updated_at: UtcDatetime

    @model_validator(mode="after")
    def _check_times(self) -> Session:
        """Reject an updated_at earlier than created_at."""
        if self.updated_at < self.created_at:
            raise ValueError("updated_at is earlier than created_at")
        return self
