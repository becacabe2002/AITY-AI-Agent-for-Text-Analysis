import pytest
from typing import get_args
from pydantic import ValidationError

from aity_web.llm import (
    Message,
    Role,
    TokenUsage,
    assistant_message,
    user_message,
)

FIELDS = ("prompt_tokens", "output_tokens", "thought_tokens", "total_tokens")


# --- TokenUsage ---

def test_usage_defaults_to_unreported():
    u = TokenUsage()
    assert all(getattr(u, f) is None for f in FIELDS)
    assert u.is_empty


def test_usage_keeps_reported_counts():
    u = TokenUsage(prompt_tokens=100, output_tokens=40, thought_tokens=15, total_tokens=140)
    assert (u.prompt_tokens, u.output_tokens, u.thought_tokens, u.total_tokens) == (100, 40, 15, 140)
    assert not u.is_empty


def test_usage_fills_missing_total():
    # Anthropic reports no total: prompt + output (thinking already inside output).
    assert TokenUsage(prompt_tokens=10, output_tokens=5).total_tokens == 15


def test_usage_fills_total_from_dict_input():
    # Same path as reading the JSONB blob back from Postgres.
    u = TokenUsage.model_validate({"prompt_tokens": 7, "output_tokens": 3})
    assert u.total_tokens == 10


def test_usage_does_not_overwrite_reported_total():
    # A provider total can differ from prompt + output (e.g. cached or tool tokens).
    assert TokenUsage(prompt_tokens=10, output_tokens=5, total_tokens=20).total_tokens == 20


@pytest.mark.parametrize(
    "kwargs",
    [
        {"prompt_tokens": 10},       # output unknown
        {"output_tokens": 5},        # prompt unknown
        {"thought_tokens": 3},       # thought alone never makes a total
    ],
)
def test_usage_never_guesses_total_from_partial_counts(kwargs):
    assert TokenUsage(**kwargs).total_tokens is None


def test_usage_zero_counts_are_reported_not_empty():
    u = TokenUsage(prompt_tokens=0, output_tokens=0)
    assert u.total_tokens == 0
    assert not u.is_empty


@pytest.mark.parametrize("field", FIELDS)
def test_usage_single_count_is_not_empty(field):
    assert not TokenUsage(**{field: 1}).is_empty


@pytest.mark.parametrize(
    "kwargs",
    [
        {"prompt_tokens": "12"},                    # strict: no str -> int coercion
        {"output_tokens": 3.0},                     # strict: no float -> int coercion
        {"prompt_tokens": True},                    # bool is not a token count
        {"output_tokens": -1},                      # counts are non-negative
        {"total_tokens": -5},
        {"prompt_tokens": -1, "output_tokens": 5},  # fill-in must not hide a bad count
        {"cached_tokens": 3},                       # extra="forbid"
    ],
)
def test_usage_rejects_invalid(kwargs):
    with pytest.raises(ValidationError):
        TokenUsage(**kwargs)


def test_usage_is_frozen_and_hashable():
    u = TokenUsage(prompt_tokens=1, output_tokens=2)
    with pytest.raises(ValidationError):
        u.prompt_tokens = 5
    assert hash(u) == hash(TokenUsage(prompt_tokens=1, output_tokens=2, total_tokens=3))


@pytest.mark.parametrize(
    "u",
    [
        TokenUsage(),
        TokenUsage(prompt_tokens=10, output_tokens=5),
        TokenUsage(prompt_tokens=100, output_tokens=40, thought_tokens=15, total_tokens=140),
    ],
)
def test_usage_json_round_trip(u):
    # The token_usage JSONB column stores exactly this dump.
    assert TokenUsage.model_validate_json(u.model_dump_json()) == u


def test_usage_dump_has_the_four_slots():
    # Same shape as the desktop build_token_usage() dict.
    assert set(TokenUsage().model_dump()) == set(FIELDS)


# --- Message ---

def test_user_message():
    assert user_message("hi") == {"role": "user", "content": "hi"}


def test_assistant_message():
    assert assistant_message("ok") == {"role": "assistant", "content": "ok"}


def test_message_is_a_plain_dict():
    # ADR-002 contract: complete(messages: list[dict]); adapters may pass it straight on.
    m: Message = user_message("hi")
    assert type(m) is dict


def test_message_has_no_system_role():
    # The system prompt is its own argument of complete(), never a message.
    assert set(Message.__annotations__) == {"role", "content"}
    assert set(get_args(Role)) == {"user", "assistant"}