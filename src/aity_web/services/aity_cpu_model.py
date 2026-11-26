"""Energy and CO2e estimate for AITY Built-in, based on the ollama container's CPU time."""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - Python 3.10
    import tomli as tomllib

CONSTANTS_ENV = "AITY_SUSTAINABILITY_CONSTANTS"
DEFAULT_CONSTANTS_FILE = Path(__file__).resolve().parents[1] / "config" / "sustainability_constants.toml"

FORMULA_KEYS: tuple[str, ...] = ("W_PER_VCPU", "W_PER_GB_RAM", "PUE", "GRID_KG_PER_KWH")
REQUIRED_KEYS: tuple[str, ...] = ("constants_version", *FORMULA_KEYS, "DEFAULT_MODEL_RESIDENT_GB")

Method = Literal["cpu_time", "wall_clock"]


class ConstantsError(ValueError):
    """The constants file is missing, unreadable or incomplete."""


class ReadingsError(ValueError):
    """The shim payload does not have the expected shape."""


# --- constants ---


def load_constants(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Read and check the constants file (argument -> $AITY_SUSTAINABILITY_CONSTANTS -> shipped file)."""
    source = Path(path or os.environ.get(CONSTANTS_ENV) or DEFAULT_CONSTANTS_FILE)
    try:
        constants = tomllib.loads(source.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConstantsError(f"cannot read sustainability constants {source}: {exc}") from exc

    missing = [k for k in REQUIRED_KEYS if k not in constants]
    if missing:
        raise ConstantsError(f"{source}: missing constants {missing}")
    if not isinstance(constants["constants_version"], int) or isinstance(constants["constants_version"], bool):
        raise ConstantsError(f"{source}: constants_version must be an int")
    for key in (*FORMULA_KEYS, "DEFAULT_MODEL_RESIDENT_GB"):
        if not _non_negative(constants[key]):
            raise ConstantsError(f"{source}: {key} must be a non-negative number")
    per_model = constants.get("model_resident_gb", {})
    if not isinstance(per_model, dict) or not all(_non_negative(v) for v in per_model.values()):
        raise ConstantsError(f"{source}: [model_resident_gb] values must be non-negative numbers")
    return constants


@lru_cache(maxsize=1)
def default_constants() -> dict[str, Any]:
    """The process-wide constants (read once; a failed read is retried next call)."""
    return load_constants()


# --- estimate ---


@dataclass(frozen=True, slots=True)
class CpuModelEstimate:
    """Result of the CPU-time model, with the inputs and constants behind it."""

    energy_kwh: float
    co2e_kg: float
    method: Method
    cpu_seconds: float
    duration_s: float
    resident_gb: float
    constants_version: int
    constants: Mapping[str, float]
    warnings: tuple[str, ...] = ()

    def inputs_entry(self) -> str:
        """One warnings[] line recording every input and constant of this row."""
        parts = [
            f"method={self.method}",
            f"cpu_seconds={round(self.cpu_seconds, 3)}",
            f"duration_s={round(self.duration_s, 3)}",
            *(f"{k}={v}" for k, v in self.constants.items()),
            f"MODEL_RESIDENT_GB={round(self.resident_gb, 2)}",
        ]
        return f"aity_cpu_model v{self.constants_version}: " + ", ".join(parts)


def estimate(readings: Mapping[str, Any], constants: Mapping[str, Any] | None = None) -> CpuModelEstimate:
    """Apply the formula to one shim payload. Raises ReadingsError / ConstantsError."""
    c = constants if constants is not None else default_constants()
    if not isinstance(readings, Mapping):
        raise ReadingsError(f"shim payload must be an object, got {type(readings).__name__}")

    duration = readings.get("duration_s")
    if not _non_negative(duration):
        raise ReadingsError(f"duration_s must be a non-negative number, got {duration!r}")
    duration = float(duration)

    warnings: list[str] = []
    cpu = readings.get("cpu_seconds")
    if cpu is None:
        vcpus = readings.get("vcpus")
        if not _non_negative(vcpus) or vcpus == 0:
            raise ReadingsError(f"vcpus must be a positive number when cpu_seconds is missing, got {vcpus!r}")
        method: Method = "wall_clock"
        cpu_seconds = duration * float(vcpus)
        warnings.append(
            "cgroup cpu.stat unreadable; cpu_seconds is an upper bound (duration_s x allotted vCPUs)"
        )
    elif _non_negative(cpu):
        method = "cpu_time"
        cpu_seconds = float(cpu)
    else:
        raise ReadingsError(f"cpu_seconds must be a non-negative number or null, got {cpu!r}")

    resident_gb, size_warning = _resident_gb(readings, c)
    if size_warning:
        warnings.append(size_warning)

    it_joules = cpu_seconds * c["W_PER_VCPU"] + duration * resident_gb * c["W_PER_GB_RAM"]
    energy_kwh = it_joules * c["PUE"] / 3.6e6
    return CpuModelEstimate(
        energy_kwh=energy_kwh,
        co2e_kg=energy_kwh * c["GRID_KG_PER_KWH"],
        method=method,
        cpu_seconds=cpu_seconds,
        duration_s=duration,
        resident_gb=resident_gb,
        constants_version=c["constants_version"],
        constants={k: c[k] for k in FORMULA_KEYS},
        warnings=tuple(warnings),
    )


def _resident_gb(readings: Mapping[str, Any], c: Mapping[str, Any]) -> tuple[float, str | None]:
    """Model size from Ollama's /api/ps (via the shim), else the per-model / default constant."""
    size = readings.get("resident_bytes")
    if size is not None and not _non_negative(size):
        raise ReadingsError(f"resident_bytes must be a non-negative number or null, got {size!r}")
    if size:
        return size / 1e9, None
    model = readings.get("model")
    per_model = c.get("model_resident_gb", {})
    gb = float(per_model.get(model, c["DEFAULT_MODEL_RESIDENT_GB"]) if isinstance(model, str)
               else c["DEFAULT_MODEL_RESIDENT_GB"])
    return gb, f"resident model size not available from /api/ps; used config value {gb} GB"


def _non_negative(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0
