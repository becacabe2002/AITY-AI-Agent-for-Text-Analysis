from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, model_validator

from aity_web.llm.types import ProviderName, TokenUsage
from aity_web.models._types import DOMAIN_CONFIG, Name, Sha256Hex, UtcDatetime
from aity_web.models.analysis_payload import LLMAnalysisPayload
from aity_web.models.llm_config import LLMConfig
from aity_web.models.sustainability import SustainabilityMetrics

ANALYSIS_RESULT_SCHEMA_VERSION = 1

AnalysisStatus = Literal["ok", "partial", "failed"]


class AnalysisResult(BaseModel):
    """
    The persisted record of one analysis.
    Two layers: 
    * `payload` is what the LLM generated (LLMAnalysisPayload)
    * Other field is added by the service (provenance, usage, sustainability, cache key, audit). 
    
    One row = one analysis of one document with the provider/model
    the user picked and the prompt variant the operator set.
    The API never returns this model directly: AnalysisResultOut (W3 request / response models) drops user_id, content_hash 
    and cache_key_version, hides sustainability.warnings and adds cache_hit.
    """
    model_config = DOMAIN_CONFIG

    # Bumped on any breaking change to the persisted shape; a row with an
    # unknown version is refused, never coerced.
    schema_version: Literal[1] = ANALYSIS_RESULT_SCHEMA_VERSION

    # Identity
    analysis_id: UUID
    session_id: UUID
    document_id: UUID
    user_id: UUID

    # Provenance - what produced this result
    provider: ProviderName
    model: Name
    prompt_variant: Name

    # from analysis result
    payload: LLMAnalysisPayload | None = None

    # Cost and impact 
    token_usage: TokenUsage | None = None # None only when the call failed 
    sustainability: SustainabilityMetrics

    # Cache key - derived from the inputs; lookup arrives in W6
    content_hash: Sha256Hex
    cache_key_version: int = Field(ge=1)
    served_from_analysis_id: UUID | None = None

    # Input handling (ADR-007)
    truncated: bool = False # set only when content truncated (ADR-007)
    truncation_warning: str | None = None

    # Audit
    """
    Status rules (checked by the validator):

    status   payload   token_usage   error
    ok       set       set           None
    partial  set       set           payload issues, joined into one line
    failed   None      set or None   short sanitised reason
    """
    status: AnalysisStatus
    error: str | None = None
    created_at: UtcDatetime
    finished_at: UtcDatetime

    @model_validator(mode="after")
    def _check_consistency(self) -> AnalysisResult:
        has_error = bool(self.error and self.error.strip())

        if self.status == "failed":
            if self.payload is not None:
                raise ValueError("failed rows carry no payload")
            if not has_error:
                raise ValueError("failed rows need an error")
        else:
            if self.payload is None:
                raise ValueError(f"{self.status!r} rows need a payload")
            if self.token_usage is None:
                raise ValueError(f"{self.status!r} rows need token_usage")
            if self.status == "ok" and self.error is not None:
                raise ValueError("ok rows carry no error")
            if self.status == "partial" and not has_error:
                raise ValueError("partial rows need an error listing the payload issues")

        if self.served_from_analysis_id is not None:
            if self.served_from_analysis_id == self.analysis_id:
                raise ValueError("a row cannot be served from itself")
            if self.status != "ok":
                raise ValueError("only ok results are served from the cache")

        if self.truncated != (self.truncation_warning is not None):
            raise ValueError("truncation_warning is set iff truncated")

        if self.finished_at < self.created_at:
            raise ValueError("finished_at is earlier than created_at")
        return self

    @property
    def config(self) -> LLMConfig:
        """The LLMConfig that produced this row."""
        return LLMConfig(
            provider=self.provider, model=self.model, prompt_variant=self.prompt_variant
        )

    @property
    def is_cache_copy(self) -> bool:
        """True for a row copied from another session's result (W6)."""
        return self.served_from_analysis_id is not None
