"""Request and response bodies for sessions, documents and analyses - ADR-004 §7.8.

Plain Pydantic: the W5 FastAPI routes (web_ui/backend) use them as request
bodies and response_model, and build the *Out models with from_domain().
No FastAPI import here.

    POST   /api/v1/sessions                    SessionCreate  -> 201 SessionOut
    PATCH  /api/v1/sessions/{sid}              SessionUpdate  -> 200 SessionOut
    DELETE /api/v1/sessions/{sid}                             -> 204
    POST   /api/v1/sessions/{sid}/documents    (multipart)    -> 201 DocumentOut
    POST   /api/v1/documents/{did}/analyses    AnalysisCreate -> 202 (job, ADR-006)
    GET    /api/v1/analyses/{aid}                             -> 200 AnalysisResultOut

What never leaves the server: user_id, Document.text_path and text_sha256,
AnalysisResult.content_hash / cache_key_version, sustainability.warnings.
No request accepts a prompt variant (the operator sets it); extra fields in a
request body are refused (422), so a client cannot sneak one in.
"""

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
    model_config = _REQUEST

    name: SessionName
    description: SessionDescription = None


class SessionUpdate(BaseModel):
    """PATCH body. An omitted field is left unchanged; description=null (or
    blank) clears the description; name cannot be null."""

    model_config = _REQUEST

    name: SessionName | None = None
    description: SessionDescription = None

    @model_validator(mode="after")
    def _name_not_null(self) -> SessionUpdate:
        if "name" in self.model_fields_set and self.name is None:
            raise ValueError("name cannot be null")
        return self

    def changes(self) -> dict[str, Any]:
        """Only the fields the client sent, ready for SessionService.update(**changes)
        (fields left out stay UNSET there)."""
        return {name: getattr(self, name) for name in self.model_fields_set}


class SessionOut(BaseModel):
    model_config = _RESPONSE

    id: UUID
    name: str
    description: str | None
    document_count: int = Field(ge=0)
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_domain(cls, session: Session, *, document_count: int) -> SessionOut:
        return cls(
            id=session.id, name=session.name, description=session.description,
            document_count=document_count,
            created_at=session.created_at, updated_at=session.updated_at,
        )


# --- documents ---


class DocumentOut(BaseModel):
    model_config = _RESPONSE

    id: UUID
    session_id: UUID
    original_filename: str
    source_type: SourceType
    char_count: int
    uploaded_at: datetime

    @classmethod
    def from_domain(cls, document: Document) -> DocumentOut:
        return cls(
            id=document.id, session_id=document.session_id,
            original_filename=document.original_filename, source_type=document.source_type,
            char_count=document.char_count, uploaded_at=document.uploaded_at,
        )


# --- analyses ---


class AnalysisCreate(BaseModel):
    """The user picks provider and model; the prompt variant is the operator's."""

    model_config = _REQUEST

    provider: ProviderName
    model: Name


class SustainabilityOut(BaseModel):
    """SustainabilityMetrics without the server-side warnings[]."""

    model_config = _RESPONSE

    energy_kwh: float | None
    co2e_kg: float | None
    estimate_status: EstimateStatus
    estimation_source: EstimationSource | None
    latency_seconds: float
    errors: list[str]

    @classmethod
    def from_domain(cls, metrics: SustainabilityMetrics) -> SustainabilityOut:
        return cls(**metrics.model_dump(exclude={"warnings"}))


class AnalysisResultOut(BaseModel):
    """What the API returns for one analysis. cache_hit lives here, never on
    the row (False until the W6 cache lookup)."""

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
