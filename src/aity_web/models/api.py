"""Request and response bodies for the session, document and analysis endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from aity_web.llm.types import ProviderName, TokenUsage
from aity_web.models._types import Name
from aity_web.models.analysis_payload import LLMAnalysisPayload
from aity_web.models.analysis_result import AnalysisResult, AnalysisStatus
from aity_web.models.document import Document, SourceType
from aity_web.models.session import Session, SessionDescription, SessionName
from aity_web.models.sustainability import EstimateStatus, EstimationSource, SustainabilityMetrics

_REQUEST = ConfigDict(extra="forbid", frozen=True)
_RESPONSE = ConfigDict(extra="forbid", frozen=True)


# --- sessions ---


class SessionCreate(BaseModel):
    """Request body to create a session."""

    model_config = _REQUEST

    name: SessionName
    description: SessionDescription = None


class SessionUpdate(BaseModel):
    """PATCH body: omitted fields stay unchanged and a null description clears it."""

    model_config = _REQUEST

    name: SessionName | None = None
    description: SessionDescription = None

    @model_validator(mode="after")
    def _name_not_null(self) -> SessionUpdate:
        """Reject an explicit null name."""
        if "name" in self.model_fields_set and self.name is None:
            raise ValueError("name cannot be null")
        return self

    def changes(self) -> dict[str, Any]:
        """Return only the fields the client sent."""
        return {name: getattr(self, name) for name in self.model_fields_set}


class SessionOut(BaseModel):
    """A session as the API returns it, with its document count."""

    model_config = _RESPONSE

    id: UUID
    name: str
    description: str | None
    document_count: int = Field(ge=0)
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_domain(cls, session: Session, *, document_count: int) -> SessionOut:
        """Build the response from a session and its document count."""
        return cls(
            id=session.id, name=session.name, description=session.description,
            document_count=document_count,
            created_at=session.created_at, updated_at=session.updated_at,
        )


# --- documents ---


class DocumentOut(BaseModel):
    """A document as the API returns it (no paths or hashes)."""

    model_config = _RESPONSE

    id: UUID
    session_id: UUID
    original_filename: str
    source_type: SourceType
    char_count: int
    uploaded_at: datetime

    @classmethod
    def from_domain(cls, document: Document) -> DocumentOut:
        """Build the response from a document."""
        return cls(
            id=document.id, session_id=document.session_id,
            original_filename=document.original_filename, source_type=document.source_type,
            char_count=document.char_count, uploaded_at=document.uploaded_at,
        )


# --- analyses ---


class AnalysisCreate(BaseModel):
    """Request body to start an analysis: the user picks provider and model."""

    model_config = _REQUEST

    provider: ProviderName
    model: Name


class SustainabilityOut(BaseModel):
    """Sustainability metrics as the API returns them, without server-side warnings."""

    model_config = _RESPONSE

    energy_kwh: float | None
    co2e_kg: float | None
    estimate_status: EstimateStatus
    estimation_source: EstimationSource | None
    latency_seconds: float
    errors: list[str]

    @classmethod
    def from_domain(cls, metrics: SustainabilityMetrics) -> SustainabilityOut:
        """Build the response from the stored metrics."""
        return cls(**metrics.model_dump(exclude={"warnings"}))


class AnalysisResultOut(BaseModel):
    """An analysis as the API returns it; cache_hit is reported here, never stored."""

    model_config = _RESPONSE

    analysis_id: UUID
    session_id: UUID
    document_id: UUID
    provider: ProviderName
    model: str
    prompt_variant: str
    status: AnalysisStatus
    error: str | None
    payload: LLMAnalysisPayload | None
    token_usage: TokenUsage | None
    sustainability: SustainabilityOut
    truncated: bool
    truncation_warning: str | None
    cache_hit: bool = False
    created_at: datetime
    finished_at: datetime

    @classmethod
    def from_domain(cls, result: AnalysisResult, *, cache_hit: bool = False) -> AnalysisResultOut:
        """Build the response from a stored result."""
        return cls(
            analysis_id=result.analysis_id, session_id=result.session_id,
            document_id=result.document_id, provider=result.provider, model=result.model,
            prompt_variant=result.prompt_variant, status=result.status, error=result.error,
            payload=result.payload, token_usage=result.token_usage,
            sustainability=SustainabilityOut.from_domain(result.sustainability),
            truncated=result.truncated, truncation_warning=result.truncation_warning,
            cache_hit=cache_hit, created_at=result.created_at, finished_at=result.finished_at,
        )


__all__ = [
    "AnalysisCreate",
    "AnalysisResultOut",
    "DocumentOut",
    "SessionCreate",
    "SessionOut",
    "SessionUpdate",
    "SustainabilityOut",
]
