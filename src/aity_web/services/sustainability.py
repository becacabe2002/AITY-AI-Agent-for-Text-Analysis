"""Sustainability estimation for each provider call: EcoLogits setup, timer and collector."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from aity_web.llm.providers.base import ProviderResponse
from aity_web.models.sustainability import SustainabilityMetrics
from aity_web.services import aity_cpu_model

log = logging.getLogger(__name__)

# --- EcoLogits: one init per process ---

#: EcoLogits provider ids for the three BYOK cloud SDKs. Note "google_genai"
#: (underscore): "google-genai" raises EcoLogitsError.
ECOLOGITS_PROVIDERS: tuple[str, ...] = ("openai", "anthropic", "google_genai")

#: AITY provider names whose rows are estimated by EcoLogits.
ECOLOGITS_ROWS: frozenset[str] = frozenset({"openai", "anthropic", "gemini"})

_ecologits_lock = threading.Lock()
_ecologits_ready = False


def init_ecologits() -> bool:
    """Patch the cloud SDKs with EcoLogits once per process; return True if this call did it."""
    global _ecologits_ready
    with _ecologits_lock:
        if _ecologits_ready:
            return False
        from ecologits import EcoLogits  # imported here so importing this module patches nothing

        EcoLogits.init(providers=list(ECOLOGITS_PROVIDERS))
        _ecologits_ready = True
        log.info("EcoLogits initialised for %s", ", ".join(ECOLOGITS_PROVIDERS))
        return True


def ecologits_ready() -> bool:
    """True once init_ecologits() has run in this process."""
    return _ecologits_ready


# --- latency timer ---


@dataclass
class Stopwatch:
    """Holds the elapsed seconds set by wall_clock()."""

    seconds: float | None = None


@contextmanager
def wall_clock() -> Generator[Stopwatch]:
    """Time a block with a monotonic clock; seconds is set even if the block raises."""
    watch = Stopwatch()
    start = time.perf_counter()
    try:
        yield watch
    finally:
        watch.seconds = time.perf_counter() - start


# --- collector ---

_ERROR_MAX_LEN = 300


#: EcoLogits warnings that describe the model, not this request. Every
#: proprietary model carries them, so they are recorded but keep the row "ok".
INFORMATIONAL_ECOLOGITS_WARNINGS: frozenset[str] = frozenset(
    {"model-arch-not-released", "model-arch-multimodal"}
)


def _number(value: Any) -> float | None:
    """Convert an EcoLogits value (a number or a range) to one number."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    low, high = getattr(value, "min", None), getattr(value, "max", None)
    if isinstance(low, (int, float)) and isinstance(high, (int, float)):
        return (low + high) / 2
    raise TypeError(f"unexpected EcoLogits value type {type(value).__name__}")


def _message(item: Any) -> str:
    """Format an EcoLogits warning or error as 'code: message'."""
    code, text = getattr(item, "code", None), getattr(item, "message", None)
    if code and text:
        return f"{code}: {text}"
    return str(item)[:_ERROR_MAX_LEN]


class SustainabilityCollector:
    """Maps one provider call to SustainabilityMetrics."""

    @staticmethod
    def collect(
        provider: str,
        provider_response: ProviderResponse | None,
        latency_seconds: float,
        ollama_track_result: Mapping[str, Any] | None = None,
    ) -> SustainabilityMetrics:
        """Build the sustainability block for one provider call; never raises."""
        latency = max(0.0, float(latency_seconds))
        try:
            if provider in ECOLOGITS_ROWS:
                return _from_ecologits(provider_response, latency)
            if provider == "ollama":
                # provider_response.impacts is ignored on purpose (ADR-003 §7.3)
                return _from_shim(ollama_track_result, latency)
            return SustainabilityMetrics.unavailable(
                latency, [f"no estimation path for provider {provider!r}"]
            )
        except Exception as exc:  # degradation, never a failed analysis
            log.warning("sustainability collection failed for %s: %r", provider, exc)
            return SustainabilityMetrics.unavailable(
                latency, [f"collector failed: {type(exc).__name__}"]
            )


def _from_ecologits(resp: ProviderResponse | None, latency: float) -> SustainabilityMetrics:
    """Build metrics from the EcoLogits impacts of a cloud response."""
    if resp is None:
        return SustainabilityMetrics.unavailable(latency, ["ecologits: no provider response (the call failed)"])
    impacts = getattr(resp, "impacts", None)
    if impacts is None:
        return SustainabilityMetrics.unavailable(
            latency, ["ecologits: no impacts on the response (EcoLogits not initialised?)"]
        )

    errors = [_message(e) for e in (getattr(impacts, "errors", None) or [])]
    raw_warnings = getattr(impacts, "warnings", None) or []
    warnings = [_message(w) for w in raw_warnings]

    energy = _number(getattr(getattr(impacts, "energy", None), "value", None))
    co2e = _number(getattr(getattr(impacts, "gwp", None), "value", None))

    if errors or energy is None or co2e is None:
        return SustainabilityMetrics.unavailable(
            latency, errors or ["ecologits: impacts carried no energy / gwp value"]
        )

    # Model-level warnings are informational; any other warning makes the row partial.
    degrading = [w for w in raw_warnings
                 if getattr(w, "code", None) not in INFORMATIONAL_ECOLOGITS_WARNINGS]
    status = "partial" if degrading else "ok"
    return SustainabilityMetrics(
        energy_kwh=energy,
        co2e_kg=co2e,
        estimate_status=status,
        estimation_source="ecologits",
        latency_seconds=latency,
        warnings=warnings,
    )


#: estimate method -> estimate_status (ADR-003 §4 table)
_SHIM_STATUS = {"cpu_time": "ok", "wall_clock": "partial"}


def _from_shim(readings: Mapping[str, Any] | None, latency: float) -> SustainabilityMetrics:
    """Shim readings -> backend formula (aity_cpu_model) -> metrics row."""
    if readings is None:
        return SustainabilityMetrics.unavailable(
            latency, ["aity_cpu_model: no shim result (shim unreachable or failed)"]
        )
    try:
        est = aity_cpu_model.estimate(readings)
    except aity_cpu_model.ConstantsError as exc:
        log.error("sustainability constants unavailable: %s", exc)
        return SustainabilityMetrics.unavailable(latency, ["aity_cpu_model: constants unavailable"])
    except aity_cpu_model.ReadingsError as exc:
        return SustainabilityMetrics.unavailable(
            latency, [f"aity_cpu_model: malformed shim result ({str(exc)[:_ERROR_MAX_LEN]})"]
        )

    return SustainabilityMetrics(
        energy_kwh=est.energy_kwh,
        co2e_kg=est.co2e_kg,
        estimate_status=_SHIM_STATUS[est.method],
        estimation_source="aity_cpu_model",
        latency_seconds=latency,
        warnings=[est.inputs_entry(), *est.warnings],
    )


collect = SustainabilityCollector.collect
