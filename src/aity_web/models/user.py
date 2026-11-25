from __future__ import annotations

from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, Field, StringConstraints

from aity_web.models._types import DOMAIN_CONFIG, UtcDatetime

USERNAME_MAX = 64

# username rules: case-sensitive, only consist of alphabet and number characters, unique
Username = Annotated[
    str, StringConstraints(pattern=r"^[A-Za-z0-9]+$", min_length=1, max_length=USERNAME_MAX)
]


class User(BaseModel):
    """
    Model for user account
    * Never returned by the API as-is: password_hash stays in the service layer."""

    model_config = DOMAIN_CONFIG

    id: UUID
    username: Username
    password_hash: str = Field(min_length=1, repr=False)  # bcrypt hash (W4)
    created_at: UtcDatetime
