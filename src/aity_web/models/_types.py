"""Shared field types and Pydantic settings for the domain models."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated

from pydantic import AfterValidator, AwareDatetime, ConfigDict, StringConstraints


def _to_utc(value: datetime) -> datetime:
    """Convert an aware datetime to UTC."""
    return value.astimezone(timezone.utc)


UtcDatetime = Annotated[AwareDatetime, AfterValidator(_to_utc)]

Sha256Hex = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]

Name = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]

# Domain models are immutable values. from_attributes lets a SQLAlchemy
# repository build them straight from ORM rows (Model.model_validate(row)).
DOMAIN_CONFIG = ConfigDict(extra="forbid", frozen=True, from_attributes=True)
