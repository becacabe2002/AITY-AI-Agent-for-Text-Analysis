"""
Raw readings of the shim (ADR-003 §4, revised Nov 25th, 2026) - no web framework,
so it is unit-tested on its own (tests/ollama_shim/).

The shim only MEASURES. The formula and its constants live in the backend
(aity_web.services.aity_cpu_model), so changing them never needs a new
ollama image.
"""
from __future__ import annotations

import os
from pathlib import Path


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


def cpu_seconds_between(start_usec: int | None, end_usec: int | None) -> float | None:
    """CPU seconds between two readings; None if either is missing or the counter went back."""
    if start_usec is None or end_usec is None or end_usec < start_usec:
        return None
    return (end_usec - start_usec) / 1e6


def allotted_vcpus(cgroup_dir: Path) -> float:
    """vCPUs this container may use: the compose `cpus:` limit if set, else all VM vCPUs."""
    try:
        quota, period = (cgroup_dir / "cpu.max").read_text().split()
        if quota != "max":
            return int(quota) / int(period)
    except (OSError, ValueError):
        pass
    return float(os.cpu_count() or 1)
