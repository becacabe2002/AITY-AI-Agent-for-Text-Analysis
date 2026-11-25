from __future__ import annotations

from typing import Literal, get_args
from uuid import UUID

from pydantic import BaseModel, Field, model_validator

from aity_web.models._types import DOMAIN_CONFIG, UtcDatetime

CloudProviderName = Literal["gemini", "openai", "anthropic"]
CLOUD_PROVIDER_NAMES: tuple[str, ...] = get_args(CloudProviderName)


class UserApiKey(BaseModel):
    """
    Store one BYOK credential. This will never leaves the service layer.
    Only the three cloud providers need a key. Ollama doesn't.
    One row per (user_id, provider)
    """
    model_config = DOMAIN_CONFIG

    user_id: UUID
    provider: CloudProviderName
    ciphertext: bytes = Field(min_length=1, repr=False) # ApiKeyService will encrypts and decrypts, the repositories nerver see a plaintext key.
    created_at: UtcDatetime
    updated_at: UtcDatetime

    @model_validator(mode="after")
    def _check_times(self) -> UserApiKey:
        if self.updated_at < self.created_at:
            raise ValueError("updated_at is earlier than created_at")
        return self
