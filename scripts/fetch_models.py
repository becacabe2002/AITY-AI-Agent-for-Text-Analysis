#!/usr/bin/env python3
"""Pull the model lists of the cloud providers and save them as JSON."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "var" / "models"
TIMEOUT_S = 30.0

KEY_ENV: dict[str, tuple[str, ...]] = {
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "openai": ("OPENAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
}


def load_dotenv(path: Path) -> None:
    """KEY=value lines, optional quotes and # comments; the environment wins."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.removeprefix("export ").strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        os.environ.setdefault(key, value)


def api_key(provider: str) -> str | None:
    """Return the first configured API key for the provider, or None."""
    return next((os.environ[n] for n in KEY_ENV[provider] if os.environ.get(n, "").strip()), None)


def fetch_gemini(key: str) -> list[dict[str, Any]]:
    """List the Gemini models as JSON-ready dicts."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=int(TIMEOUT_S * 1000)))
    return [m.model_dump(mode="json", exclude_none=True) for m in client.models.list()]


def fetch_openai(key: str) -> list[dict[str, Any]]:
    """List the OpenAI models as JSON-ready dicts."""
    from openai import OpenAI

    client = OpenAI(api_key=key, timeout=TIMEOUT_S, max_retries=1)
    return [m.model_dump(mode="json") for m in client.models.list()]


def fetch_anthropic(key: str) -> list[dict[str, Any]]:
    """List the Anthropic models as JSON-ready dicts."""
    from anthropic import Anthropic

    client = Anthropic(api_key=key, timeout=TIMEOUT_S, max_retries=1)
    return [m.model_dump(mode="json", exclude_none=True) for m in client.models.list(limit=100)]


FETCHERS: dict[str, Callable[[str], list[dict[str, Any]]]] = {
    "gemini": fetch_gemini,
    "openai": fetch_openai,
    "anthropic": fetch_anthropic,
}


def main(argv: list[str] | None = None) -> int:
    """Fetch each requested provider's model list into var/models; return the exit code."""
    parser = argparse.ArgumentParser(
        description=(
            "Pull the cloud providers' model lists and write them to var/models/models_<provider>.json.\n"
            "API keys come from GEMINI_API_KEY (or GOOGLE_API_KEY), OPENAI_API_KEY and\n"
            "ANTHROPIC_API_KEY, in the environment or the repo-root .env (the environment wins;\n"
            "values are never printed). Read-only: models.json is edited by hand, and Ollama is\n"
            "not covered."
        ),
        epilog=(
            "examples (repo root):\n"
            "  python scripts/fetch_models.py                  # all three providers\n"
            "  python scripts/fetch_models.py --provider anthropic\n\n"
            "exit codes: 0 if at least one list was written, 1 otherwise."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--provider", choices=[*FETCHERS, "all"], default="all", help="provider to fetch (default: all)"
    )
    args = parser.parse_args(argv)

    load_dotenv(REPO_ROOT / ".env")
    providers = list(FETCHERS) if args.provider == "all" else [args.provider]
    fetched = 0
    for provider in providers:
        key = api_key(provider)
        if key is None:
            print(f"{provider}: skipped, no key ({' or '.join(KEY_ENV[provider])})")
            continue
        try:
            models = FETCHERS[provider](key)
        except Exception as exc:  # noqa: BLE001 - SDK text can echo request details
            print(f"{provider}: failed ({type(exc).__name__})")
            continue
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUT_DIR / f"models_{provider}.json"
        path.write_text(json.dumps(models, indent=2) + "\n", encoding="utf-8")
        print(f"{provider}: {len(models)} models -> {path.relative_to(REPO_ROOT)}")
        fetched += 1
    return 0 if fetched else 1


if __name__ == "__main__":
    raise SystemExit(main())
