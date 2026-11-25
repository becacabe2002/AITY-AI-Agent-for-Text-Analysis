"""
Backend-side client for the estimation shim in the ollama container

The analysis service wraps adapter.complete() for provider="ollama" with
start() / stop(), OUTSIDE the latency timer. OllamaProvider never calls this.

Failures never raise: they log and return None, and SustainabilityCollector
turns a None track result into estimate_status="unavailable".

    track_id = await shim.start()
    try:
        with wall_clock() as t:
            resp = await adapter.complete(...)
    finally:
        track = await shim.stop(track_id, model)   # always stop, even on failure
    metrics = collect("ollama", resp, t.seconds, ollama_track_result=track)

The CPU-time delta belongs to this one call only because AITY Built-in runs
one job at a time (per-provider concurrency = 1, ADR-006).
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

log = logging.getLogger(__name__)

DEFAULT_SHIM_URL = "http://ollama:8788"


class OllamaTrackShim:
    def __init__(
        self,
        client: httpx.AsyncClient,
        base_url: str | None = None,
        timeout_s: float = 2.0,
    ) -> None:
        self._client = client
        self._base = (base_url or os.environ.get("AITY_OLLAMA_SHIM_URL", DEFAULT_SHIM_URL)).rstrip("/")
        self._timeout = timeout_s

    async def start(self) -> str | None:
        """Begin a track; returns its id, or None if the shim is unavailable."""
        try:
            r = await self._client.post(f"{self._base}/internal/track/start", timeout=self._timeout)
            r.raise_for_status()
            track_id = r.json()["track_id"]
            return track_id if isinstance(track_id, str) else None
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
            log.warning("estimation shim start failed: %r", exc)
            return None

    async def stop(self, track_id: str | None, model: str | None = None) -> dict[str, Any] | None:
        """End a track; returns the shim's estimate payload, or None."""
        if track_id is None:
            return None
        try:
            r = await self._client.post(
                f"{self._base}/internal/track/stop",
                json={"track_id": track_id, "model": model},
                timeout=self._timeout,
            )
            r.raise_for_status()
            payload = r.json()
            return payload if isinstance(payload, dict) else None
        except (httpx.HTTPError, ValueError) as exc:
            log.warning("estimation shim stop failed: %r", exc)
            return None
