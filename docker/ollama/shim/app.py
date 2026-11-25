"""AITY estimation shim - runs inside the ollama container (ADR-003 §7.4).

Estimates the energy of one Ollama call from the container's own CPU time
(cgroup v2 cpu.stat -> usage_usec) and the sourced constants in
sustainability_constants.toml. Bound to the internal Docker network only.
The formula lives in estimator.py.

Endpoints:
  POST /internal/track/start                     -> {"track_id"}
  POST /internal/track/stop {track_id, model?}   -> estimate payload (ADR-003 §4)
  GET  /internal/track/health                    -> readiness + whether cpu.stat is readable
"""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.request
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from estimator import allotted_vcpus, estimate, load_constants, read_cpu_usec

CGROUP_DIR = Path(os.environ.get("AITY_SHIM_CGROUP_DIR", "/sys/fs/cgroup"))
CONSTANTS_PATH = Path(
    os.environ.get(
        "AITY_SHIM_CONSTANTS",
        str(Path(__file__).with_name("sustainability_constants.toml")),
    )
)
OLLAMA_URL = os.environ.get("OLLAMA_INTERNAL_URL", "http://127.0.0.1:11434")
TRACK_TTL_S = 3600  # forget tracks that were started but never stopped

C = load_constants(CONSTANTS_PATH)

app = FastAPI(title="aity-estimation-shim", docs_url=None, redoc_url=None, openapi_url=None)
_tracks: dict[str, tuple[int | None, float]] = {}  # track_id -> (start usage_usec, start monotonic)
_lock = threading.Lock()


def resident_gb(model: str | None) -> tuple[float, str | None]:
    """Size of the loaded model(s) from Ollama's /api/ps; falls back to config."""
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/ps", timeout=2) as resp:
            models = json.load(resp).get("models", [])
        if model:
            models = [m for m in models if model in (m.get("name"), m.get("model"))]
        size = sum(m.get("size", 0) for m in models)
        if size > 0:
            return size / 1e9, None
    except (OSError, ValueError):
        pass
    per_model = C.get("model_resident_gb", {})
    gb = float(per_model.get(model or "", C["DEFAULT_MODEL_RESIDENT_GB"]))
    return gb, f"resident model size not available from /api/ps; used config value {gb} GB"


class StopRequest(BaseModel):
    track_id: str
    model: str | None = None


@app.post("/internal/track/start")
def start() -> dict:
    now = time.monotonic()
    track_id = str(uuid.uuid4())
    with _lock:
        for key, (_, t0) in list(_tracks.items()):
            if now - t0 > TRACK_TTL_S:
                del _tracks[key]
        _tracks[track_id] = (read_cpu_usec(CGROUP_DIR), now)
    return {"track_id": track_id}


@app.post("/internal/track/stop")
def stop(req: StopRequest) -> dict:
    end_usec, end_t = read_cpu_usec(CGROUP_DIR), time.monotonic()
    with _lock:
        entry = _tracks.pop(req.track_id, None)
    if entry is None:
        raise HTTPException(status_code=404, detail="unknown or expired track_id")
    start_usec, start_t = entry
    gb, size_warning = resident_gb(req.model)
    return estimate(
        start_usec=start_usec,
        end_usec=end_usec,
        duration_s=end_t - start_t,
        resident_gb=gb,
        vcpus=allotted_vcpus(CGROUP_DIR),
        constants=C,
        extra_warnings=[size_warning] if size_warning else None,
    )


@app.get("/internal/track/health")
def health() -> dict:
    return {
        "status": "ok",
        "cpu_stat_readable": read_cpu_usec(CGROUP_DIR) is not None,
        "constants_version": C["constants_version"],
    }
