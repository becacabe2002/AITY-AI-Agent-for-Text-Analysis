"""
Every row is an ESTIMATE; `estimation_source` says which method made it:
* "ecologits"      - cloud providers, from model class + token counts
* "aity_cpu_model" - AITY Built-in, from the ollama container's CPU time
* None             - no estimate (estimate_status="unavailable")

Only SustainabilityCollector (aity_web.services.sustainability) builds these.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

EstimateStatus = Literal["ok", "partial", "unavailable"]
EstimationSource = Literal["ecologits", "aity_cpu_model"]


class SustainabilityMetrics(BaseModel):
    """Persisted as the `sustainability` JSONB column of analysis_results.

    * energy_kwh / co2e_kg - None iff estimate_status="unavailable"
    * estimate_status      - "ok": primary method, no degradation;
      "partial": numbers present but the source flagged a degradation;
      "unavailable": no numbers
    * estimation_source    - None if estimate_status="unavailable"
    * latency_seconds      - wall clock of adapter.complete() only; always set
    * warnings             - server-side only, never returned by the API
    * errors               - why the estimate is partial / unavailable
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    energy_kwh: float | None = Field(default=None, ge=0)
    co2e_kg: float | None = Field(default=None, ge=0)
    estimate_status: EstimateStatus
    estimation_source: EstimationSource | None = None
    latency_seconds: float = Field(ge=0)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_consistency(self) -> SustainabilityMetrics:
        unavailable = self.estimate_status == "unavailable"
        has_numbers = self.energy_kwh is not None and self.co2e_kg is not None
        no_numbers = self.energy_kwh is None and self.co2e_kg is None
        if unavailable:
            if not no_numbers or self.estimation_source is not None:
                raise ValueError("unavailable rows carry no numbers and no estimation_source")
        else:
            if not has_numbers or self.estimation_source is None:
                raise ValueError(
                    f"{self.estimate_status!r} rows need energy_kwh, co2e_kg and estimation_source"
                )
        return self

    @classmethod
    def unavailable(
        cls, latency_seconds: float, errors: list[str] | None = None
    ) -> SustainabilityMetrics:
        """
        A valid row with no estimate - the degradation path.
        It will return an instance of SustainabilityMetrics 
        with estimation_source, energy_kwh and co2e_kg all are None.
        """
        return cls(
            estimate_status="unavailable",
            latency_seconds=latency_seconds,
            errors=list(errors or []),
        )
