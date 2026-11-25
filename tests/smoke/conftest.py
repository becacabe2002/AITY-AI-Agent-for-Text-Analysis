"""
Opt-in live smoke suite: real provider calls through aity_web.services.analysis_service.

Skipped unless AITY_LIVE_TESTS=1, and not in pytest.ini's testpaths, so a plain
`pytest` (and CI) never makes paid calls. Run from the repo root:

    # PowerShell
    $env:AITY_LIVE_TESTS = "1"; python -m pytest tests/smoke -v
    # bash
    AITY_LIVE_TESTS=1 python -m pytest tests/smoke -v

Settings (environment variables, or KEY=value lines in the gitignored repo-root
.env; real environment variables win and values are never printed):

    GEMINI_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY
        BYOK keys; a provider without a key is skipped.
    AITY_SMOKE_PROVIDERS=gemini,anthropic
        subset to run (default: all four).
    AITY_SMOKE_MODEL_<PROVIDER>=<model id>    e.g. AITY_SMOKE_MODEL_OPENAI
        model to test (default: the catalogue default). Must be in
        src/aity_web/config/models.json: add a new model there first, then
        smoke-test it and set its "verified" date.
        For ollama this also pins AITY_OLLAMA_MODEL (unless already set).
    AITY_OLLAMA_BASE_URL   (default here: http://localhost:11434)
    AITY_OLLAMA_SHIM_URL   (default here: http://localhost:8788)
        Ollama is skipped if the server is unreachable or the model is not pulled.
        Without the estimation shim, Ollama rows are "unavailable" - expected
        off Pouta.

A results table is printed at the end of the run.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
LIVE_ENV = "AITY_LIVE_TESTS"
ALL_PROVIDERS = ("gemini", "openai", "anthropic", "ollama")
KEY_ENV = {"gemini": "GEMINI_API_KEY", "openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}


def live_enabled() -> bool:
    return os.environ.get(LIVE_ENV, "").strip().lower() in {"1", "true", "yes"}


def load_dotenv(path: Path) -> list[str]:
    """Minimal .env reader: KEY=value lines, optional `export `, quotes and # comments.

    Variables already set in the environment are not overridden.
    Returns the names that were loaded (values are never printed).
    """
    if not path.is_file():
        return []
    loaded: list[str] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        elif " #" in value:  # inline comment on an unquoted value
            value = value.split(" #", 1)[0].rstrip()
        if key and key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded


if live_enabled():
    _LOADED = load_dotenv(REPO_ROOT / ".env")
    # Before analysis_service builds its registry (on the first call).
    os.environ.setdefault("AITY_OLLAMA_BASE_URL", "http://localhost:11434")
    os.environ.setdefault("AITY_OLLAMA_SHIM_URL", "http://localhost:8788")
    # AITY Built-in runs one pinned model (ADR-009): pin the one being smoke-tested.
    if os.environ.get("AITY_SMOKE_MODEL_OLLAMA", "").strip():
        os.environ.setdefault("AITY_OLLAMA_MODEL", os.environ["AITY_SMOKE_MODEL_OLLAMA"].strip())
else:
    _LOADED = []


def selected_providers() -> list[str]:
    raw = os.environ.get("AITY_SMOKE_PROVIDERS", "")
    names = [p.strip().lower() for p in raw.split(",") if p.strip()] or list(ALL_PROVIDERS)
    unknown = sorted(set(names) - set(ALL_PROVIDERS))
    if unknown:
        raise pytest.UsageError(f"AITY_SMOKE_PROVIDERS: unknown providers {unknown}")
    return names


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "live: real provider calls (opt-in, AITY_LIVE_TESTS=1)")
    config.stash[_RESULTS] = []


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    here = Path(__file__).parent
    skip = pytest.mark.skip(reason=f"live smoke tests: set {LIVE_ENV}=1 to run (real, paid API calls)")
    for item in items:
        if here in Path(item.fspath).parents:
            item.add_marker("live")
            if not live_enabled():
                item.add_marker(skip)


_RESULTS = pytest.StashKey[list]()


@pytest.fixture
def record_outcome(request: pytest.FixtureRequest):
    """Collect each outcome for the end-of-run table."""
    results = request.config.stash[_RESULTS]
    return results.append


def pytest_terminal_summary(terminalreporter, exitstatus, config: pytest.Config) -> None:
    results = config.stash.get(_RESULTS, [])
    if not live_enabled():
        return
    tr = terminalreporter
    tr.section("AITY live smoke results")
    tr.write_line(f".env: loaded {', '.join(_LOADED) if _LOADED else 'nothing'} from {REPO_ROOT / '.env'}")
    for o in results:
        s = o.sustainability
        energy = f"{s.energy_kwh:.3g} kWh / {s.co2e_kg:.3g} kg" if s.energy_kwh is not None else "-"
        tr.write_line(
            f"{o.provider:9s} {o.model:26s} {o.status:8s} {s.latency_seconds:6.1f} s  "
            f"tokens={o.usage.total_tokens}  finish={o.finish_reason}  "
            f"sustainability={s.estimate_status}/{s.estimation_source} {energy}"
        )
        if o.payload is not None:
            tr.write_line(f"{'':9s} keywords : {o.payload.keywords}")
            tr.write_line(f"{'':9s} topics   : {[t.label for t in o.payload.topics]}")
            tr.write_line(f"{'':9s} summary  : {o.payload.summary[:160]}")
        for line in (*o.issues, *([o.error] if o.error else []), *s.errors):
            tr.write_line(f"{'':9s} ! {line}")
