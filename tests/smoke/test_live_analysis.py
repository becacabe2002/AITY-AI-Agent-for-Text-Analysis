"""
Live end-to-end check of the analysis chain, one test per provider:

    LLMConfig -> catalogue/registry -> prompt library -> adapter (real API)
    -> payload validation/normalisation -> SustainabilityCollector

Replaces scripts/smoke_providers.py, which called the adapters directly.
See conftest.py for how to run it and the settings it reads.
"""

from __future__ import annotations

import os

import httpx
import pytest

from aity_web.llm.catalog import load_catalog
from aity_web.models.llm_config import LLMConfig
from aity_web.services.analysis_service import analyze

from .conftest import KEY_ENV, selected_providers

DOCUMENT = (
    "Quantum software libraries such as Qiskit, Cirq and PennyLane give developers "
    "high-level abstractions over quantum circuits. Recent studies compare their "
    "transpilation quality, simulator performance and documentation. Common problems "
    "include breaking API changes between versions, limited debugging tools and a "
    "steep learning curve for developers without a physics background. Researchers "
    "recommend better testing frameworks and clearer migration guides."
)


def _model_for(provider: str) -> str:
    override = os.environ.get(f"AITY_SMOKE_MODEL_{provider.upper()}", "").strip()
    return override or load_catalog().provider(provider).default_model.id


def _ollama_ready(model: str) -> None:
    """Skip unless the Ollama server answers and has `model` pulled."""
    base = os.environ["AITY_OLLAMA_BASE_URL"].rstrip("/")
    try:
        r = httpx.get(f"{base}/api/tags", timeout=3.0)
        r.raise_for_status()
        pulled = {m.get("name") for m in r.json().get("models", [])}
    except (httpx.HTTPError, ValueError) as exc:
        pytest.skip(f"Ollama not reachable at {base} ({type(exc).__name__})")
    if model not in pulled:
        pytest.skip(f"Ollama model {model!r} is not pulled (ollama pull {model})")


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", selected_providers())
async def test_analyze_end_to_end(provider: str, record_outcome) -> None:
    model = _model_for(provider)
    api_key = None
    if provider in KEY_ENV:
        api_key = os.environ.get(KEY_ENV[provider])
        if not api_key:
            pytest.skip(f"{KEY_ENV[provider]} is not set (add it to .env)")
    else:
        _ollama_ready(model)

    outcome = await analyze(DOCUMENT, LLMConfig(provider=provider, model=model), api_key=api_key)
    record_outcome(outcome)

    # 1. the provider accepted the schema and the output validated ("partial" = valid
    #    payload with count shortfalls; reported, not a failure)
    assert outcome.status != "failed", f"{provider}/{model}: {outcome.error} (details in the log)"
    assert outcome.payload is not None
    # 2. the model finished normally
    assert outcome.finish_reason == "stop"
    # 3. token usage was reported
    assert not outcome.usage.is_empty, "no token usage reported"
    # 4. sustainability: latency always; EcoLogits numbers on the cloud path
    assert outcome.sustainability.latency_seconds > 0
    if provider != "ollama":
        s = outcome.sustainability
        assert s.estimation_source == "ecologits" and s.estimate_status != "unavailable", (
            f"no EcoLogits estimate for {model}: {s.errors} "
            "(model missing from EcoLogits' repository?)"
        )
