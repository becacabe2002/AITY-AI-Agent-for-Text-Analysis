"""Model catalogue: which models AITY offers per provider, loaded from models.json."""

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
    """The model is in the catalogue but is not the operator's active model."""


# --- schema ---


class ModelSpec(BaseModel):
    """One model offered for a provider, with its tier, options and prompt profile."""

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
        """Require unique ids and exactly one selectable default model."""
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
        """Return the model with this id, or None."""
        return next((m for m in self.models if m.id == model_id), None)

    @property
    def default_model(self) -> ModelSpec:
        """The provider's default model."""
        return next(m for m in self.models if m.default)

    @property
    def selectable_models(self) -> tuple[ModelSpec, ...]:
        """Models a user may pick for a new session."""
        return tuple(m for m in self.models if m.selectable)


class ModelCatalog(BaseModel):
    """All providers and models that AITY offers."""

    model_config = _FROZEN

    catalog_version: Literal[1]
    providers: dict[ProviderName, ProviderCatalog] = Field(min_length=1)

    def provider(self, name: str) -> ProviderCatalog:
        """Return a provider's catalogue or raise UnknownProviderError."""
        try:
            return self.providers[name]  # type: ignore[index]
        except KeyError:
            raise UnknownProviderError(
                f"provider {name!r} is not offered (catalogue has {sorted(self.providers)})"
            ) from None

    def model(self, provider: str, model_id: str) -> ModelSpec:
        """Return a model spec or raise UnknownModelError."""
        spec = self.provider(provider).get(model_id)
        if spec is None:
            raise UnknownModelError(f"model {model_id!r} is not in the {provider!r} catalogue")
        return spec


# --- loading ---


def load_catalog(path: str | os.PathLike[str] | None = None) -> ModelCatalog:
    """Load and validate the catalogue from a path, the environment or the shipped file."""
    source = Path(path or os.environ.get(MODELS_JSON_ENV) or DEFAULT_MODELS_JSON)
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise CatalogError(f"cannot read model catalogue {source}: {exc}") from exc
    try:
        return ModelCatalog.model_validate_json(text)
    except ValidationError as exc:
        raise CatalogError(f"invalid model catalogue {source}:\n{exc}") from exc
