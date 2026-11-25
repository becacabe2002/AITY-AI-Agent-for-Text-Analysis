from __future__ import annotations
from typing import Annotated
from pydantic import BaseModel, ConfigDict, StringConstraints
from aity_web.llm.types import ProviderName

DEFAULT_PROMPT_VARIANT = "default_analysis"

_Name = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]

class LLMConfig(BaseModel):
    """
    Configuration of one LLM call: an analysis, a comparison or a synthesis. It is built per call by the service: the user picks provider and model, the operator sets prompt_variant for the use case.

    * No sampling configurations (temperature, seed, max output tokens, think) -> on adapters
    * No API keys -> resolve by ApiKeyService at call time
    """
    model_config = ConfigDict(extra="forbid", frozen=True)
    provider: ProviderName
    model: _Name
    prompt_variant: _Name = DEFAULT_PROMPT_VARIANT

    @property
    def needs_api_key(self) -> bool:
        """True for the three BYOK cloud providers; AITY Built-in never needs a key."""
        return self.provider != "ollama"
