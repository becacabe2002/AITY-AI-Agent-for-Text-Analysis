import json

import httpx
import pytest

from aity_web.services.ollama_track_shim import OllamaTrackShim


def _shim(handler):
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return OllamaTrackShim(client, base_url="http://shim:8788/"), client


@pytest.mark.asyncio
async def test_start_and_stop_round_trip():
    seen = []

    def handler(req):
        seen.append((req.url.path, req.content))
        if req.url.path.endswith("/start"):
            return httpx.Response(200, json={"track_id": "t1"})
        return httpx.Response(200, json={"cpu_seconds": 30.0, "duration_s": 40.0})

    shim, client = _shim(handler)
    async with client:
        tid = await shim.start()
        result = await shim.stop(tid, "qwen3.5:4b")
    assert tid == "t1" and result["cpu_seconds"] == 30.0
    assert seen[0][0] == "/internal/track/start"
    assert json.loads(seen[1][1]) == {"track_id": "t1", "model": "qwen3.5:4b"}


@pytest.mark.asyncio
@pytest.mark.parametrize("response", [
    httpx.Response(500), httpx.Response(200, json={}), httpx.Response(200, text="nope"),
    httpx.Response(200, json={"track_id": 7}),
])
async def test_start_failures_return_none(response):
    shim, client = _shim(lambda req: response)
    async with client:
        assert await shim.start() is None


@pytest.mark.asyncio
async def test_start_connection_error_returns_none():
    def handler(req):
        raise httpx.ConnectError("down")
    shim, client = _shim(handler)
    async with client:
        assert await shim.start() is None


@pytest.mark.asyncio
async def test_stop_without_track_makes_no_call():
    calls = []
    shim, client = _shim(lambda req: calls.append(req) or httpx.Response(200, json={}))
    async with client:
        assert await shim.stop(None) is None
    assert calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("response", [httpx.Response(404), httpx.Response(200, json=[1])])
async def test_stop_failures_return_none(response):
    shim, client = _shim(lambda req: response)
    async with client:
        assert await shim.stop("t1") is None


def test_base_url_from_env(monkeypatch):
    monkeypatch.setenv("AITY_OLLAMA_SHIM_URL", "http://x:1/")
    assert OllamaTrackShim(httpx.AsyncClient())._base == "http://x:1"
