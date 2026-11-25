import re
from pathlib import Path

import pytest

import aity_web.services.sustainability as sus

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def fake_init(monkeypatch):
    calls = []
    import ecologits
    monkeypatch.setattr(ecologits.EcoLogits, "init", staticmethod(lambda **kw: calls.append(kw)))
    monkeypatch.setattr(sus, "_ecologits_ready", False)
    return calls


def test_init_runs_once_with_cloud_providers(fake_init):
    assert sus.init_ecologits() is True
    assert sus.init_ecologits() is False
    assert sus.ecologits_ready()
    assert fake_init == [{"providers": ["openai", "anthropic", "google_genai"]}]


def test_provider_ids_are_valid_ecologits_names():
    from ecologits._ecologits import _INSTRUMENTS
    assert set(sus.ECOLOGITS_PROVIDERS) <= set(_INSTRUMENTS)   # catches "google-genai"


def test_importing_the_module_does_not_patch():
    src = (REPO / "src/aity_web/services/sustainability.py").read_text(encoding="utf-8")
    assert not re.search(r"^(from|import) ecologits", src, re.M)   # lazy import only


def test_only_one_ecologits_init_call_site():
    """EcoLogits.init() appears only in services/sustainability.py (ADR-003 §7.3)."""
    allowed = REPO / "src/aity_web/services/sustainability.py"
    roots = [REPO / "src/aity_web", REPO / "web_ui", REPO / "scripts"]
    offenders = [
        p.relative_to(REPO).as_posix()
        for root in roots if root.exists()
        for p in root.rglob("*.py")
        if p != allowed and "EcoLogits.init(" in p.read_text(encoding="utf-8")
    ]
    assert offenders == []
