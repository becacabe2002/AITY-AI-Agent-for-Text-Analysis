"""
Model catalogue: which models AITY offers per provider (config/models.json).

* Hand-written; the model-list scraper is W3 work.
* Cloud entries are maintained in the repo. AITY Built-in entries are
  operator-controlled: they list the models the operator may run on the
  ollama container. On Pouta the operator can point AITY_MODELS_JSON at
  their own copy of the file instead of rebuilding the image.
* AITY Built-in runs ONE model at a time (ADR-009): the registry pins it
  from $AITY_OLLAMA_MODEL (default: the catalogue's `default` entry). The
  other Built-in entries stay resolvable so past results can be displayed,
  but cannot be run (ModelNotActiveError). Changing the model = restart.
* A provider missing from the file is not offered at all.
* Per-model `options` are default adapter kwargs (e.g. num_ctx,
  structured_output_mode); kwargs passed at call time win.
* `prompt_profile` picks the prompt override ("small" = shorter, more
  directive prompts for small local models; see llm/prompts).
* `selectable: false` hides a model from new sessions but keeps it
  resolvable, so existing sessions that use it still work. (For AITY
  Built-in, the pinned model decides what is offered, not `selectable`.)
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from aity_web.llm.types import ProviderName

MODELS_JSON_ENV = "AITY_MODELS_JSON"
DEFAULT_MODELS_JSON = Path(__file__).resolve().parent.parent / "config" / "models.json"

PromptProfile = Literal["standard", "small"]
Tier = Literal["fast", "balanced", "capable"]

_FROZEN = ConfigDict(extra="forbid", frozen=True)


# --- errors ---


class CatalogError(ValueError):
    """models.json is missing or invalid."""


class UnknownProviderError(LookupError):
    """The provider is not registered or not offered in the catalogue."""


class UnknownModelError(LookupError):
    """The model is not in the provider's catalogue."""


class ModelNotActiveError(LookupError):
    """The model is in the catalogue but the operator has not made it the
    active model for its provider (AITY Built-in runs one model at a time).
    Results produced earlier with it can still be viewed, not re-run."""


# --- schema ---


class ModelSpec(BaseModel):
    model_config = _FROZEN

    id: str = Field(min_length=1, max_length=200)        # exact string sent to the provider
    display_name: str = Field(min_length=1)
    tier: Tier
    default: bool = False
    selectable: bool = True
    context_window: int | None = Field(default=None, gt=0)
    prompt_profile: PromptProfile = "standard"
    options: dict[str, Any] = Field(default_factory=dict)
    latency_hint: str = ""
    verified: date | None = None                          # last successful live smoke test
    notes: str = ""


class ProviderCatalog(BaseModel):
    """Models AITY offers per provider"""
    model_config = _FROZEN

    display_name: str = Field(min_length=1)
    models: tuple[ModelSpec, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_models(self) -> ProviderCatalog:
        ids = [m.id for m in self.models]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise ValueError(f"duplicate model ids: {duplicates}")
        defaults = [m for m in self.models if m.default]
        if len(defaults) != 1:
            raise ValueError(f"exactly one default model required, found {len(defaults)}")
        if not defaults[0].selectable:
            raise ValueError(f"default model {defaults[0].id!r} must be selectable")
        return self

    def get(self, model_id: str) -> ModelSpec | None:
        return next((m for m in self.models if m.id == model_id), None)

    @property
    def default_model(self) -> ModelSpec:
        return next(m for m in self.models if m.default)

    @property
    def selectable_models(self) -> tuple[ModelSpec, ...]:
        return tuple(m for m in self.models if m.selectable)


class ModelCatalog(BaseModel):
    model_config = _FROZEN

    catalog_version: Literal[1]
    providers: dict[ProviderName, ProviderCatalog] = Field(min_length=1)

    def provider(self, name: str) -> ProviderCatalog:
        try:
            return self.providers[name]  # type: ignore[index]
        except KeyError:
            raise UnknownProviderError(
                f"provider {name!r} is not offered (catalogue has {sorted(self.providers)})"
            ) from None

    def model(self, provider: str, model_id: str) -> ModelSpec:
        spec = self.provider(provider).get(model_id)
        if spec is None:
            raise UnknownModelError(f"model {model_id!r} is not in the {provider!r} catalogue")
        return spec


# --- loading ---


def load_catalog(path: str | os.PathLike[str] | None = None) -> ModelCatalog:
    """Load and validate the catalogue.

    Path order: the `path` argument, then $AITY_MODELS_JSON, then the copy
    shipped in aity_web/config/models.json. Raises CatalogError so a broken
    file fails at startup, not at the first analysis.
    """
    source = Path(path or os.environ.get(MODELS_JSON_ENV) or DEFAULT_MODELS_JSON)
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise CatalogError(f"cannot read model catalogue {source}: {exc}") from exc
    try:
        return ModelCatalog.model_validate_json(text)
    except ValidationError as exc:
        raise CatalogError(f"invalid model catalogue {source}:\n{exc}") from exc
