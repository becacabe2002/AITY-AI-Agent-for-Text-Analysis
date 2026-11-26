"""Domain model of an uploaded document, stored as text only."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, Field, StringConstraints, model_validator

from aity_web.models._types import DOMAIN_CONFIG, Sha256Hex, UtcDatetime

SourceType = Literal["pdf", "docx", "txt"]

FILENAME_MAX = 255

OriginalFilename = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=FILENAME_MAX)
]


class Document(BaseModel):
    """An uploaded file kept as text: extracted text for PDF and DOCX, an exact copy for TXT."""

    model_config = DOMAIN_CONFIG

    id: UUID
    session_id: UUID
    user_id: UUID
    original_filename: OriginalFilename
    source_type: SourceType
    text_path: str
    text_sha256: Sha256Hex
    char_count: int = Field(ge=1)
    uploaded_at: UtcDatetime

    @model_validator(mode="after")
    def _check_text_path(self) -> Document:
        """Check that text_path is a clean relative path inside the owner's session folder."""
        raw = self.text_path
        path = PurePosixPath(raw)
        if (
            "\\" in raw
            or path.is_absolute()
            or ".." in path.parts
            or path.suffix != ".txt"
            or path.as_posix() != raw
        ):
            raise ValueError("text_path must be a clean relative POSIX path ending in .txt")
        if path.parts[:2] != (str(self.user_id), str(self.session_id)) or len(path.parts) != 3:
            raise ValueError("text_path must sit directly in {user_id}/{session_id}/")
        if not path.name.startswith(f"{self.id}_"):
            raise ValueError("text_path file name must start with the document id")
        return self
