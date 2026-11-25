"""Shim endpoints: stop returns raw readings only (no energy numbers)."""
import pytest
from fastapi.testclient import TestClient

import app as shim_app


@pytest.fixture
def client(tmp_path, monkeypatch):
    stat = tmp_path / "cpu.stat"
    stat.write_text("usage_usec 1000000\n")
    (tmp_path / "cpu.max").write_text("800000 100000\n")
    monkeypatch.setattr(shim_app, "CGROUP_DIR", tmp_path)
    monkeypatch.setattr(shim_app, "resident_bytes", lambda model: 4_200_000_000)
    monkeypatch.setattr(shim_app, "_tracks", {})
    return TestClient(shim_app.app), stat


def test_start_stop_returns_readings(client):
    c, stat = client
    track_id = c.post("/internal/track/start").json()["track_id"]
    stat.write_text("usage_usec 41000000\n")                         # +40 CPU seconds
    body = c.post("/internal/track/stop", json={"track_id": track_id, "model": "qwen3.5:4b"}).json()

    assert set(body) == {"cpu_seconds", "duration_s", "vcpus", "resident_bytes", "model"}
    assert body["cpu_seconds"] == 40.0 and body["vcpus"] == 8.0
    assert body["resident_bytes"] == 4_200_000_000 and body["model"] == "qwen3.5:4b"
    assert body["duration_s"] >= 0


def test_unreadable_cpu_stat_gives_null(client):
    c, stat = client
    stat.unlink()
    track_id = c.post("/internal/track/start").json()["track_id"]
    body = c.post("/internal/track/stop", json={"track_id": track_id}).json()
    assert body["cpu_seconds"] is None and body["model"] is None


def test_unknown_track_is_404(client):
    c, _ = client
    assert c.post("/internal/track/stop", json={"track_id": "nope"}).status_code == 404


def test_health(client):
    c, stat = client
    assert c.get("/internal/track/health").json() == {"status": "ok", "cpu_stat_readable": True}
    stat.unlink()
    assert c.get("/internal/track/health").json()["cpu_stat_readable"] is False
