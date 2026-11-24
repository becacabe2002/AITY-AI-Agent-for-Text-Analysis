import pytest
from pydantic import ValidationError

from aity_web.models import SustainabilityMetrics


def test_ok_row():
    m = SustainabilityMetrics(energy_kwh=1e-4, co2e_kg=2e-6, estimate_status="ok",
                              estimation_source="ecologits", latency_seconds=1.5)
    assert m.warnings == [] and m.errors == []


def test_unavailable_helper():
    m = SustainabilityMetrics.unavailable(2.0, ["boom"])
    assert (m.energy_kwh, m.co2e_kg, m.estimation_source) == (None, None, None)
    assert m.estimate_status == "unavailable" and m.errors == ["boom"]


@pytest.mark.parametrize("kwargs", [
    dict(estimate_status="ok", estimation_source="ecologits"),                      # numbers missing
    dict(estimate_status="partial", energy_kwh=1.0, co2e_kg=1.0),                    # source missing
    dict(estimate_status="unavailable", energy_kwh=1.0, co2e_kg=1.0),                # numbers present
    dict(estimate_status="unavailable", estimation_source="aity_cpu_model"),         # source present
    dict(estimate_status="ok", estimation_source="codecarbon", energy_kwh=1, co2e_kg=1),  # retired source
    dict(estimate_status="ok", estimation_source="ecologits", energy_kwh=-1, co2e_kg=1),  # negative
])
def test_inconsistent_rows_rejected(kwargs):
    with pytest.raises(ValidationError):
        SustainabilityMetrics(latency_seconds=1.0, **kwargs)


def test_latency_required_and_non_negative():
    with pytest.raises(ValidationError):
        SustainabilityMetrics(estimate_status="unavailable")
    with pytest.raises(ValidationError):
        SustainabilityMetrics(estimate_status="unavailable", latency_seconds=-0.1)


def test_frozen_and_json_round_trip():
    m = SustainabilityMetrics.unavailable(0.5)
    with pytest.raises(ValidationError):
        m.latency_seconds = 2.0
    assert SustainabilityMetrics.model_validate_json(m.model_dump_json()) == m
