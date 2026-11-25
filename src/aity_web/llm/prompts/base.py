"""
Prompt variant types and rendering.

Rules:
* Variant names are immutable (decided 2026-11-19, option B): once a name has
  produced results, its text never changes. A text change ships under a new
  name (default_analysis -> default_analysis_v2). tests/aity_web/test_prompts.py
  pins a fingerprint per variant to catch accidental edits.
* A variant has a "standard" template and optional per-profile overrides
  ("small" = shorter, more directive text for small local models). The
  profile comes from the model's catalogue entry, so (variant, model) always
  resolves to the same text and the cache key needs only the variant name.
* Templates use string.Template ($name) so JSON braces need no escaping.
  A literal dollar sign is written $$.
* The document goes in the user turn, never in the system prompt.
* Placeholder variants reserve a name; they have no text and cannot render.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from string import Template
from types import MappingProxyType
from typing import Literal

from aity_web.llm.catalog import PromptProfile
from aity_web.llm.types import Message, user_message
from aity_web.models.analysis_payload import KEYWORDS_MAX, KEYWORDS_MIN, TOPICS_MAX, TOPICS_MIN

PromptStatus = Literal["active", "placeholder"]

#: Every parameter a template may use, with its default. `document` has no
#: default and must always be passed.
DEFAULT_PARAMS: Mapping[str, int] = MappingProxyType(
    {
        "max_summary_words": 200,
        "keywords_min": KEYWORDS_MIN,
        "keywords_max": KEYWORDS_MAX,
        "topics_min": TOPICS_MIN,
        "topics_max": TOPICS_MAX,
    }
)
ALLOWED_PARAMS: frozenset[str] = frozenset(DEFAULT_PARAMS) | {"document"}

_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


class UnknownPromptVariantError(LookupError):
    """No prompt variant has this name."""


class PromptVariantNotAvailableError(LookupError):
    """The variant is a reserved placeholder with no prompt text yet."""


def _identifiers(text: str) -> set[str]:
    """$name / ${name} identifiers in a template (Template.get_identifiers is 3.11+)."""
    found: set[str] = set()
    for match in Template.pattern.finditer(text):
        if match.group("invalid") is not None:
            raise ValueError(f"invalid '$' at position {match.start()} (write $$ for a literal $)")
        name = match.group("named") or match.group("braced")
        if name:
            found.add(name)
    return found


@dataclass(frozen=True, slots=True)
class PromptTemplate:
    system: str
    user: str

    def __post_init__(self) -> None:
        names = _identifiers(self.system) | _identifiers(self.user)
        unknown = names - ALLOWED_PARAMS
        if unknown:
            raise ValueError(f"unknown template parameters: {sorted(unknown)}")
        if "document" in _identifiers(self.system):
            raise ValueError("$document belongs in the user template, not the system prompt")
        if "document" not in _identifiers(self.user):
            raise ValueError("the user template must contain $document")


@dataclass(frozen=True, slots=True)
class RenderedPrompt:
    """Arguments for BaseProviderAdapter.complete() / ModelHandle.complete()."""

    variant: str
    profile: PromptProfile
    system_prompt: str
    messages: tuple[Message, ...]


@dataclass(frozen=True, slots=True)
class PromptVariant:
    name: str
    description: str
    status: PromptStatus = "active"
    template: PromptTemplate | None = None
    overrides: Mapping[str, PromptTemplate] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not _NAME_RE.match(self.name):
            raise ValueError(f"invalid prompt variant name {self.name!r}")
        if self.status == "active" and self.template is None:
            raise ValueError(f"{self.name}: an active variant needs a template")
        if self.status == "placeholder" and (self.template is not None or self.overrides):
            raise ValueError(f"{self.name}: a placeholder variant has no text")
        bad = set(self.overrides) - {"small"}
        if bad or "standard" in self.overrides:
            raise ValueError(f"{self.name}: unknown override profiles {sorted(bad)}")
        object.__setattr__(self, "overrides", MappingProxyType(dict(self.overrides)))

    @property
    def available(self) -> bool:
        return self.status == "active"

    def template_for(self, profile: PromptProfile = "standard") -> PromptTemplate:
        """Override for `profile` if there is one, else the standard template."""
        if self.template is None:
            raise PromptVariantNotAvailableError(
                f"prompt variant {self.name!r} is a placeholder and cannot be used yet"
            )
        return self.overrides.get(profile, self.template)

    def render(
        self, document: str, *, profile: PromptProfile = "standard", **params: int
    ) -> RenderedPrompt:
        """Fill the template for `profile`.

        `params` override DEFAULT_PARAMS (e.g. the W3 LLMConfig limits). The
        document is substituted as-is; a '$' inside it is not re-parsed.
        """
        unknown = set(params) - set(DEFAULT_PARAMS)
        if unknown:
            raise ValueError(f"unknown prompt parameters: {sorted(unknown)}")
        values = {**DEFAULT_PARAMS, **params}
        for key, value in values.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{key} must be a positive int, got {value!r}")
        for lo, hi in (("keywords_min", "keywords_max"), ("topics_min", "topics_max")):
            if values[lo] > values[hi]:
                raise ValueError(f"{lo} ({values[lo]}) is greater than {hi} ({values[hi]})")
        if not document.strip():
            raise ValueError("document is empty")

        tpl = self.template_for(profile)
        system = Template(tpl.system).substitute(values)
        user = Template(tpl.user).substitute({**values, "document": document})
        return RenderedPrompt(
            variant=self.name,
            profile=profile if profile in self.overrides else "standard",
            system_prompt=system,
            messages=(user_message(user),),
        )

    def fingerprint(self) -> str:
        """SHA-256 over every template of the variant (standard + overrides).

        Changes whenever any text changes; the W4 prompt_variants table can
        store it to reject edits to an existing name.
        """
        h = hashlib.sha256(self.name.encode())
        templates: list[tuple[str, PromptTemplate]] = []
        if self.template is not None:
            templates.append(("standard", self.template))
        templates.extend(sorted(self.overrides.items()))
        for profile, tpl in templates:
            for part in (profile, tpl.system, tpl.user):
                h.update(b"\x00" + part.encode())
        return h.hexdigest()
