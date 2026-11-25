import time

import pytest
from ecologits.impacts.modeling import GWP, Energy
from ecologits.status_messages import (
    ElectricityMixPEDefaultWarning,
    ModelArchMultimodalWarning,
    ModelArchNotReleasedWarning,
    ModelNotRegisteredError,
)
from ecologits.tracers.utils import ImpactsOutput
from ecologits.utils.range_value import RangeValue

from aity_web.llm.providers.base import ProviderResponse
from aity_web.services.sustainability import SustainabilityCollector, collect, wall_clock


def _resp(impacts=None):
    return ProviderResponse(text="{}", impacts=impacts)


def _impacts(energy=1e-4, gwp=5e-5, warnings=None, errors=None):
    return ImpactsOutput(
        energy=None if energy is None else Energy(value=energy),
        gwp=None if gwp is None else GWP(value=gwp),
        warnings=warnings, errors=errors,
    )


def _track(**over):
    """Raw shim readings (the /internal/track/stop payload)."""
    t = {"cpu_seconds": 3648.0, "duration_s": 456.0, "vcpus": 8.0,
         "resident_bytes": 4_200_000_000, "model": "qwen3.5:4b"}
    return t | over


# --- cloud / EcoLogits ---

@pytest.mark.parametrize("provider", ["gemini", "openai", "anthropic"])
def test_ecologits_plain_values_ok(provider):
    m = collect(provider, _resp(_impacts()), 1.25)
    assert m.estimation_source == "ecologits" and m.estimate_status == "ok"
    assert (m.energy_kwh, m.co2e_kg, m.latency_seconds) == (1e-4, 5e-5, 1.25)
    assert m.warnings == [] and m.errors == []


def test_ecologits_range_stored_as_midpoint_only():
    m = collect("openai", _resp(_impacts(energy=RangeValue(min=1e-4, max=3e-4),
                                         gwp=RangeValue(min=1e-5, max=3e-5))), 1.0)
    assert m.energy_kwh == pytest.approx(2e-4) and m.co2e_kg == pytest.approx(2e-5)
    assert m.estimate_status == "ok" and m.warnings == []    # no range kept anywhere


def test_model_level_warnings_are_informational():
    warns = [ModelArchNotReleasedWarning(), ModelArchMultimodalWarning()]
    m = collect("gemini", _resp(_impacts(warnings=warns)), 1.0)
    assert m.estimate_status == "ok"                          # every proprietary model has these
    assert m.warnings[0].startswith("model-arch-not-released: ")
    assert len(m.warnings) == 2 and "http" not in m.warnings[0]   # recorded, docs URL stripped


def test_other_ecologits_warnings_make_row_partial():
    warns = [ModelArchNotReleasedWarning(), ElectricityMixPEDefaultWarning()]
    m = collect("openai", _resp(_impacts(warnings=warns)), 1.0)
    assert m.estimate_status == "partial" and len(m.warnings) == 2


def test_ecologits_errors_make_row_unavailable():
    imp = ImpactsOutput(errors=[ModelNotRegisteredError()])   # what EcoLogits returns for unknown models
    m = collect("anthropic", _resp(imp), 2.0)
    assert m.estimate_status == "unavailable" and m.estimation_source is None
    assert m.errors[0].startswith("model-not-registered: ")
    assert m.latency_seconds == 2.0


def test_ecologits_missing_impacts_unavailable():
    m = collect("openai", _resp(None), 1.0)
    assert m.estimate_status == "unavailable" and "no impacts" in m.errors[0]


def test_ecologits_values_missing_without_errors():
    m = collect("openai", _resp(_impacts(energy=None)), 1.0)
    assert m.estimate_status == "unavailable" and m.errors


def test_cloud_ignores_track_result():
    m = collect("openai", _resp(_impacts()), 1.0, ollama_track_result=_track())
    assert m.estimation_source == "ecologits"


# --- AITY Built-in / shim ---

def test_shim_cpu_time_ok_with_inputs_entry():
    m = collect("ollama", _resp(), 456.0, ollama_track_result=_track())
    assert m.estimation_source == "aity_cpu_model" and m.estimate_status == "ok"
    assert m.energy_kwh == pytest.approx(3.635e-3, rel=1e-3)      # computed in the backend
    assert m.co2e_kg == pytest.approx(9.45e-5, rel=1e-2)
    assert m.warnings == [
        "aity_cpu_model v1: method=cpu_time, cpu_seconds=3648.0, duration_s=456.0, "
        "W_PER_VCPU=3.125, W_PER_GB_RAM=0.375, PUE=1.08, GRID_KG_PER_KWH=0.026, "
        "MODEL_RESIDENT_GB=4.2"
    ]
    assert m.latency_seconds == 456.0 and m.errors == []


def test_shim_wall_clock_is_partial():
    m = collect("ollama", _resp(), 1.0, ollama_track_result=_track(cpu_seconds=None))
    assert m.estimate_status == "partial"
    assert "method=wall_clock" in m.warnings[0]
    assert m.warnings[1].startswith("cgroup cpu.stat unreadable")


def test_shim_size_fallback_keeps_row_ok():
    m = collect("ollama", _resp(), 1.0, ollama_track_result=_track(resident_bytes=None))
    assert m.estimate_status == "ok"
    assert m.warnings[1].startswith("resident model size not available")


def test_shim_none_unavailable():
    m = collect("ollama", _resp(), 3.0, ollama_track_result=None)
    assert m.estimate_status == "unavailable" and "no shim result" in m.errors[0]


@pytest.mark.parametrize("bad", [
    {"duration_s": None}, {"cpu_seconds": "0.1"}, {"cpu_seconds": -1.0},
    {"cpu_seconds": True}, {"cpu_seconds": None, "vcpus": None},
])
def test_shim_malformed_unavailable(bad):
    m = collect("ollama", _resp(), 1.0, ollama_track_result=_track(**bad))
    assert m.estimate_status == "unavailable" and m.estimation_source is None
    assert m.errors[0].startswith("aity_cpu_model: malformed shim result")


def test_broken_constants_unavailable(monkeypatch, tmp_path):
    from aity_web.services import aity_cpu_model
    bad = tmp_path / "c.toml"
    bad.write_text("constants_version = 1\n")
    monkeypatch.setenv(aity_cpu_model.CONSTANTS_ENV, str(bad))
    aity_cpu_model.default_constants.cache_clear()
    try:
        m = collect("ollama", _resp(), 1.0, ollama_track_result=_track())
    finally:
        aity_cpu_model.default_constants.cache_clear()
    assert m.estimate_status == "unavailable"
    assert m.errors == ["aity_cpu_model: constants unavailable"]


def test_ollama_ignores_accidental_impacts():
    m = collect("ollama", _resp(_impacts()), 1.0, ollama_track_result=None)
    assert m.estimate_status == "unavailable"                 # never an ecologits row


# --- degradation ---

def test_unknown_provider_unavailable():
    m = collect("mistral", _resp(_impacts()), 1.0)
    assert m.estimate_status == "unavailable" and "mistral" in m.errors[0]


def test_collector_never_raises():
    class Weird:
        energy = property(lambda self: (_ for _ in ()).throw(RuntimeError("secret detail")))
    m = SustainabilityCollector.collect("gemini", _resp(Weird()), 1.0)
    assert m.estimate_status == "unavailable"
    assert m.errors == ["collector failed: RuntimeError"]     # sanitised: no message text


def test_negative_latency_clamped():
    assert collect("ollama", _resp(), -0.001).latency_seconds == 0.0


# --- timer ---

def test_wall_clock_measures_and_survives_exceptions():
    with wall_clock() as t:
        time.sleep(0.01)
    assert t.seconds >= 0.01
    with pytest.raises(ValueError):
        with wall_clock() as t2:
            raise ValueError
    assert t2.seconds is not None and t2.seconds >= 0


@pytest.mark.asyncio
async def test_wall_clock_around_await():
    import asyncio
    with wall_clock() as t:
        await asyncio.sleep(0.01)
    assert t.seconds >= 0.01


def test_failed_call_without_response_says_so():
    m = collect("gemini", None, 0.4)
    assert m.estimate_status == "unavailable"
    assert m.errors == ["ecologits: no provider response (the call failed)"]
