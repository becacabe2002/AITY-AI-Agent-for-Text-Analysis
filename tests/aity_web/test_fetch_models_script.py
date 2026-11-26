"""scripts/fetch_models.py (no network)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "fetch_models.py"
_spec = importlib.util.spec_from_file_location("fetch_models_script", SCRIPT)
fm = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = fm
_spec.loader.exec_module(fm)


def _clear_keys(monkeypatch):
    for names in fm.KEY_ENV.values():
        for n in names:
            monkeypatch.delenv(n, raising=False)
    monkeypatch.setattr(fm, "load_dotenv", lambda path: None)


def test_dotenv_does_not_override_env(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text('OPENAI_API_KEY="from-file"\nGEMINI_API_KEY=g # comment\n', encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "from-env")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    fm.load_dotenv(env)
    assert fm.api_key("openai") == "from-env"
    assert fm.api_key("gemini") == "g"


def test_google_api_key_fallback(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "gk")
    assert fm.api_key("gemini") == "gk"


def test_no_keys_skips_everything(monkeypatch, capsys):
    _clear_keys(monkeypatch)
    assert fm.main([]) == 1
    assert capsys.readouterr().out.count("skipped, no key") == 3


def test_writes_listing(tmp_path, monkeypatch, capsys):
    _clear_keys(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setattr(fm, "OUT_DIR", tmp_path)
    monkeypatch.setattr(fm, "REPO_ROOT", tmp_path)
    monkeypatch.setitem(fm.FETCHERS, "openai", lambda key: [{"id": "gpt-5.5"}])
    assert fm.main(["--provider", "openai"]) == 0
    assert json.loads((tmp_path / "models_openai.json").read_text(encoding="utf-8")) == [{"id": "gpt-5.5"}]
    assert "openai: 1 models" in capsys.readouterr().out


def test_failure_prints_type_only(monkeypatch, capsys):
    _clear_keys(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "k")

    def boom(key):
        raise RuntimeError("Incorrect API key sk-SECRET")

    monkeypatch.setitem(fm.FETCHERS, "openai", boom)
    assert fm.main(["--provider", "openai"]) == 1
    out = capsys.readouterr().out
    assert "openai: failed (RuntimeError)" in out and "SECRET" not in out
