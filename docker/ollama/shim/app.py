"""Estimation shim inside the ollama container: measures CPU time and model size per call."""
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

from readings import allotted_vcpus, cpu_seconds_between, read_cpu_usec

CGROUP_DIR = Path(os.environ.get("AITY_SHIM_CGROUP_DIR", "/sys/fs/cgroup"))
OLLAMA_URL = os.environ.get("OLLAMA_INTERNAL_URL", "http://127.0.0.1:11434")
TRACK_TTL_S = 3600  # forget tracks that were started but never stopped
PS_TIMEOUT_S = 1.0  # keep stop() well inside the backend client's timeout

app = FastAPI(title="aity-estimation-shim", docs_url=None, redoc_url=None, openapi_url=None)
_tracks: dict[str, tuple[int | None, float]] = {}  # track_id -> (start usage_usec, start monotonic)
_lock = threading.Lock()


def resident_bytes(model: str | None) -> int | None:
    """Size of the loaded model(s) from Ollama's /api/ps, or None if unavailable."""
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/ps", timeout=PS_TIMEOUT_S) as resp:
            models = json.load(resp).get("models", [])
        if model:
            models = [m for m in models if model in (m.get("name"), m.get("model"))]
        size = sum(int(m.get("size", 0)) for m in models)
        return size if size > 0 else None
    except (OSError, ValueError, TypeError):
        return None


class StopRequest(BaseModel):
    """Request body of the stop endpoint: the track to end and the model that ran."""

    track_id: str
    model: str | None = None


@app.post("/internal/track/start")
def start() -> dict:
    """Begin a CPU-time track and return its id."""
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
    """End a track and return the raw readings; 404 for an unknown or expired track."""
    end_usec, end_t = read_cpu_usec(CGROUP_DIR), time.monotonic()
    with _lock:
        entry = _tracks.pop(req.track_id, None)
    if entry is None:
        raise HTTPException(status_code=404, detail="unknown or expired track_id")
    start_usec, start_t = entry
    return {
        "cpu_seconds": cpu_seconds_between(start_usec, end_usec),
        "duration_s": end_t - start_t,
        "vcpus": allotted_vcpus(CGROUP_DIR),
        "resident_bytes": resident_bytes(req.model),
        "model": req.model,
    }


@app.get("/internal/track/health")
def health() -> dict:
    """Report readiness and whether the cgroup CPU counter is readable."""
    return {"status": "ok", "cpu_stat_readable": read_cpu_usec(CGROUP_DIR) is not None}
