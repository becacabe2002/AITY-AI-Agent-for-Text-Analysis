# aity_desktop — frozen desktop app

This package is the original AITY desktop application: a Tkinter UI with two analysis modes,
**Gemini** (online, needs a Gemini API key) and **BERTs** (KeyBERT + BERTopic, runs locally).

## Status: frozen

Frozen since 2026-11-23, until a further decision (see ADR-001, revised 2026-11-23):

- Bug fixes only. No new features and no new providers.
- The code was moved here unchanged from `src/`. Only the import paths changed
  (`import state` → `from aity_desktop import state`), and `main.py` became `__main__.py`.
- The new web service lives in `src/aity_web/` and is developed separately.

## How to run

From the `src/` folder, with a virtual environment active:

```
pip install -r aity_desktop/requirements.txt
python -m aity_desktop
```

The first BERTs run downloads its models (about 400–500 MB). Your API key and settings are
stored locally in `aity_desktop/user_data.json` (gitignored), and results are saved under
`aity_desktop/output/`.

Run the unit tests from the repository root:

```
python -m unittest tests/unit_tests.py
```

## Import rule (one-way)

- `aity_desktop` **may** import from `aity_web` (for example, if the desktop app is later
  unfrozen and given the web service's providers).
- `aity_web` **must never** import from `aity_desktop`. This keeps BERTs, Tkinter and the desktop
  dependencies out of the web service and its Docker image.

## Sustainability tracking

The desktop app keeps its original tracker: EcoLogits for Gemini runs and CodeCarbon for BERTs runs.
The web service uses a different approach (ADR-003) and does not use CodeCarbon.

## Bringing BERTs into the web service

BERTs is not part of the web service. If it is ever needed there (for example for a corpus-analysis
view), port `keybert_analyzer.py` and `bertopic_analyzer.py` into `aity_web` rather than importing
them from here, and add their dependencies to the web service's own requirements.
