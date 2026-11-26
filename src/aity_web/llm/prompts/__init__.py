"""Prompt library: the registered prompt variants and helpers to render them."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from aity_web.llm.catalog import PromptProfile
from aity_web.llm.prompts.base import (
    ALLOWED_PARAMS,
    DEFAULT_PARAMS,
    PromptTemplate,
    PromptVariant,
    PromptVariantNotAvailableError,
    RenderedPrompt,
    UnknownPromptVariantError,
)
from aity_web.llm.prompts.default_analysis import DEFAULT_ANALYSIS
from aity_web.llm.prompts.placeholders import EXTRACTION_V1, QA_V1

PROMPT_VARIANTS: Mapping[str, PromptVariant] = MappingProxyType(
    {v.name: v for v in (DEFAULT_ANALYSIS, EXTRACTION_V1, QA_V1)}
)


def get_prompt_variant(name: str) -> PromptVariant:
    """Return a prompt variant by name or raise UnknownPromptVariantError."""
    try:
        return PROMPT_VARIANTS[name]
    except KeyError:
        raise UnknownPromptVariantError(
            f"unknown prompt variant {name!r} (known: {sorted(PROMPT_VARIANTS)})"
        ) from None


def available_variants() -> tuple[PromptVariant, ...]:
    """Variants a session can use (placeholders excluded)."""
    return tuple(v for v in PROMPT_VARIANTS.values() if v.available)


def render_prompt(
    name: str, document: str, *, profile: PromptProfile = "standard", **params: int
) -> RenderedPrompt:
    """Render a prompt variant for a document."""
    return get_prompt_variant(name).render(document, profile=profile, **params)


__all__ = [
    "ALLOWED_PARAMS",
    "DEFAULT_PARAMS",
    "PROMPT_VARIANTS",
    "PromptTemplate",
    "PromptVariant",
    "PromptVariantNotAvailableError",
    "RenderedPrompt",
    "UnknownPromptVariantError",
    "available_variants",
    "get_prompt_variant",
    "render_prompt",
]
