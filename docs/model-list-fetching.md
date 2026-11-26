# Model list fetching

`scripts/fetch_models.py` downloads the current model lists from Gemini, OpenAI and Anthropic, so the operator can see what exists before editing the catalogue by hand.

- It is read-only: it never changes `models.json`. See [model catalogue](model-catalog.md).
- Results are written under `var/models/`, which is not committed.
- API keys come from the environment or a local `.env` file and are never printed.
- Ollama is not covered, because the operator chooses and pulls that model.

Run `python scripts/fetch_models.py --help` for usage and exit codes.
