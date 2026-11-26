"""Registry that maps provider names to adapters and checks models against the catalogue."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from aity_web.llm.catalog import (
    CatalogError,
    ModelCatalog,
    ModelNotActiveError,
    ModelSpec,
    PromptProfile,
    UnknownModelError,
    UnknownProviderError,
    load_catalog,
)
from aity_web.llm.errors import ProviderError, to_provider_error
from aity_web.llm.providers.base import BaseProviderAdapter, ProviderResponse
from aity_web.llm.types import PROVIDER_NAMES, Message

OLLAMA_BASE_URL_ENV = "AITY_OLLAMA_BASE_URL"
OLLAMA_MODEL_ENV = "AITY_OLLAMA_MODEL"

AdapterFactory = Callable[[], BaseProviderAdapter]


@dataclass(frozen=True, slots=True)
class ModelHandle:
    """An adapter bound to one catalogue model; what create() returns."""

    provider: str
    spec: ModelSpec
    adapter: BaseProviderAdapter = field(repr=False, compare=False)

    @property
    def model(self) -> str:
        """The model id sent to the provider."""
        return self.spec.id

    @property
    def requires_api_key(self) -> bool:
        """True if the provider needs a user API key."""
        return self.adapter.requires_api_key

    @property
    def prompt_profile(self) -> PromptProfile:
        """The prompt profile of the model."""
        return self.spec.prompt_profile

    async def complete(
        self,
        *,
        api_key: str | None,
        messages: Sequence[Message],
        system_prompt: str,
        response_schema: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Call the adapter with the catalogue defaults; provider errors become ProviderError."""
        try:
            return await self.adapter.complete(
                api_key=api_key,
                model=self.spec.id,
                messages=messages,
                system_prompt=system_prompt,
                response_schema=response_schema,
                **{**self.spec.options, **kwargs}, # values passed at calltime can override catalogue defaults.
            )
        except ProviderError:
            raise
        except Exception as exc:
            mapped = to_provider_error(self.provider, exc)
            if mapped is None:
                raise
            raise mapped from exc


class LLMProviderRegistry:
    """Maps provider names to adapters, checked against the catalogue and any pinned models."""

    def __init__(self, catalog: ModelCatalog, *, pinned: Mapping[str, str] | None = None) -> None:
        """Create the registry; pinned maps a provider to the only model it may run."""
        self._catalog = catalog
        self._classes: dict[str, type[BaseProviderAdapter]] = {}
        self._factories: dict[str, AdapterFactory] = {} # functions to build the adapter
        self._instances: dict[str, BaseProviderAdapter] = {} # cached adapter instances, one per provider
        self._pinned: dict[str, ModelSpec] = {}
        for name, model in (pinned or {}).items():
            try:
                self._pinned[name] = catalog.model(name, model)
            except (UnknownProviderError, UnknownModelError) as exc:
                raise CatalogError(f"pinned model {model!r} for {name!r} is not usable: {exc}") from exc

    @property
    def catalog(self) -> ModelCatalog:
        """The model catalogue."""
        return self._catalog

    # --- registration ---

    def register(
        self, adapter_cls: type[BaseProviderAdapter], *, factory: AdapterFactory | None = None
    ) -> None:
        """Register an adapter class under its provider name."""
        name = getattr(adapter_cls, "name", None)
        if name not in PROVIDER_NAMES:
            raise ValueError(f"{adapter_cls.__name__}.name must be one of {PROVIDER_NAMES}, got {name!r}")
        if name in self._classes:
            raise ValueError(f"provider {name!r} is already registered")
        self._classes[name] = adapter_cls
        self._factories[name] = factory or adapter_cls

    def check(self) -> None:
        """Fail if the catalogue offers a provider that has no adapter."""
        missing = sorted(set(self._catalog.providers) - set(self._classes))
        if missing:
            raise ValueError(f"catalogue offers providers with no registered adapter: {missing}")

    # --- lookup ---

    def names(self) -> tuple[str, ...]:
        """Providers that are registered AND offered in the catalogue."""
        return tuple(n for n in PROVIDER_NAMES if n in self._classes and n in self._catalog.providers)

    def adapter_class(self, name: str) -> type[BaseProviderAdapter]:
        """Return the adapter class of a registered provider."""
        try:
            return self._classes[name]
        except KeyError:
            raise UnknownProviderError(f"provider {name!r} is not registered") from None

    def requires_api_key(self, name: str) -> bool:
        """True if the provider needs a user API key."""
        return self.adapter_class(name).requires_api_key

    def pinned_model(self, name: str) -> ModelSpec | None:
        """The operator-chosen model for `name`, or None if it is not pinned."""
        return self._pinned.get(name)

    def offered_models(self, name: str) -> tuple[ModelSpec, ...]:
        """Models a user may pick for a new analysis."""
        pinned = self._pinned.get(name)
        if pinned is not None:
            return (pinned,)
        return self._catalog.provider(name).selectable_models

    def adapter(self, name: str) -> BaseProviderAdapter:
        """Return the shared adapter instance for a provider, built on first use."""
        self.adapter_class(name)
        if name not in self._instances:
            self._instances[name] = self._factories[name]()
        return self._instances[name]

    def create(self, name: str, model: str) -> ModelHandle:
        """Bind a provider to a catalogue model; refuse unknown models and inactive pinned ones."""
        self.adapter_class(name)
        spec = self._catalog.model(name, model)
        pinned = self._pinned.get(name)
        if pinned is not None and spec.id != pinned.id:
            raise ModelNotActiveError(
                f"{name}: model {model!r} is not active (the operator runs {pinned.id!r}); "
                "results made with it can be viewed but not re-run"
            )
        return ModelHandle(provider=name, spec=spec, adapter=self.adapter(name))


def build_default_registry(
    catalog: ModelCatalog | None = None,
    *,
    ollama_base_url: str | None = None,
    ollama_model: str | None = None,
) -> LLMProviderRegistry:
    """Build the registry with the four production adapters and the active Built-in model."""
    from aity_web.llm.providers.anthropic import AnthropicProvider
    from aity_web.llm.providers.gemini import GeminiProvider
    from aity_web.llm.providers.ollama import DEFAULT_BASE_URL, OllamaProvider
    from aity_web.llm.providers.openai import OpenAIProvider

    base_url = ollama_base_url or os.environ.get(OLLAMA_BASE_URL_ENV) or DEFAULT_BASE_URL
    catalog = catalog or load_catalog()

    pinned: dict[str, str] = {}
    requested = ollama_model or os.environ.get(OLLAMA_MODEL_ENV)
    if "ollama" in catalog.providers:
        pinned["ollama"] = requested or catalog.provider("ollama").default_model.id
    elif requested:
        raise CatalogError(f"{OLLAMA_MODEL_ENV}={requested!r} is set but the catalogue offers no ollama models")

    registry = LLMProviderRegistry(catalog, pinned=pinned)
    registry.register(GeminiProvider)
    registry.register(OpenAIProvider)
    registry.register(AnthropicProvider)
    registry.register(OllamaProvider, factory=lambda: OllamaProvider(base_url=base_url))
    registry.check()
    return registry
