"""Domain model of the sustainability estimate stored with every analysis."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

EstimateStatus = Literal["ok", "partial", "unavailable"]
EstimationSource = Literal["ecologits", "aity_cpu_model"]


class SustainabilityMetrics(BaseModel):
    """Estimated energy and CO2e of one analysis, with how and how well it was estimated."""

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
        """Require numbers and a source unless the estimate is unavailable."""
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
        """Build a valid row that carries no estimate."""
        return cls(
            estimate_status="unavailable",
            latency_seconds=latency_seconds,
            errors=list(errors or []),
        )
