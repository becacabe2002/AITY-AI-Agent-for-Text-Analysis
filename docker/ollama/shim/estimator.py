"""
Pure estimation logic of the shim (ADR-003 §4) - no web framework, so it is
unit-tested on its own (tests/ollama_shim/).

    cpu_seconds = (usage_usec_stop - usage_usec_start) / 1e6
    it_joules   = cpu_seconds * W_PER_VCPU + duration_s * MODEL_RESIDENT_GB * W_PER_GB_RAM
    energy_kwh  = it_joules * PUE / 3.6e6
    co2e_kg     = energy_kwh * GRID_KG_PER_KWH
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:  # the ollama base image may ship Python 3.10
    import tomli as tomllib

FORMULA_KEYS = ("W_PER_VCPU", "W_PER_GB_RAM", "PUE", "GRID_KG_PER_KWH")
REQUIRED_KEYS = ("constants_version", *FORMULA_KEYS, "DEFAULT_MODEL_RESIDENT_GB")


def load_constants(path: Path) -> dict[str, Any]:
    """Read sustainability_constants.toml; fail at boot if a value is missing."""
    constants = tomllib.loads(path.read_text(encoding="utf-8"))
    missing = [k for k in REQUIRED_KEYS if k not in constants]
    if missing:
        raise ValueError(f"{path}: missing constants {missing}")
    return constants


def read_cpu_usec(cgroup_dir: Path) -> int | None:
    """Total CPU time of every process in this container (cgroup v2), in microseconds."""
    try:
        for line in (cgroup_dir / "cpu.stat").read_text().splitlines():
            key, _, value = line.partition(" ")
            if key == "usage_usec":
                return int(value)
    except (OSError, ValueError):
        pass
    return None


def allotted_vcpus(cgroup_dir: Path) -> float:
    """vCPUs this container may use: the compose `cpus:` limit if set, else all VM vCPUs."""
    try:
        quota, period = (cgroup_dir / "cpu.max").read_text().split()
        if quota != "max":
            return int(quota) / int(period)
    except (OSError, ValueError):
        pass
    return float(os.cpu_count() or 1)


def estimate(
    *,
    start_usec: int | None,
    end_usec: int | None,
    duration_s: float,
    resident_gb: float,
    vcpus: float,
    constants: dict[str, Any],
    extra_warnings: list[str] | None = None,
) -> dict[str, Any]:
    """Apply the ADR-003 §4 formula and return the shim's stop payload."""
    warnings = list(extra_warnings or [])
    if start_usec is not None and end_usec is not None and end_usec >= start_usec:
        method = "cpu_time"
        cpu_seconds = (end_usec - start_usec) / 1e6
    else:
        method = "wall_clock"
        cpu_seconds = duration_s * vcpus
        warnings.insert(
            0, "cgroup cpu.stat unreadable; cpu_seconds is an upper bound (duration_s x allotted vCPUs)"
        )

    it_joules = (cpu_seconds * constants["W_PER_VCPU"]
                 + duration_s * resident_gb * constants["W_PER_GB_RAM"])
    energy_kwh = it_joules * constants["PUE"] / 3.6e6
    co2e_kg = energy_kwh * constants["GRID_KG_PER_KWH"]

    return {
        "energy_kwh": energy_kwh,
        "co2e_kg": co2e_kg,
        "cpu_seconds": round(cpu_seconds, 3),
        "duration_s": round(duration_s, 3),
        "method": method,
        "constants_version": constants["constants_version"],
        "constants": {**{k: constants[k] for k in FORMULA_KEYS},
                      "MODEL_RESIDENT_GB": round(resident_gb, 2)},
        "warnings": warnings,
        "errors": [],
    }
