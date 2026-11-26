"""run_analysis(): upload -> analyse -> stored AnalysisResult (ADR-004 §7.4).

In-memory unit of work, a real UploadStorage in tmp_path, ApiKeyService with
FernetCipher, and the scripted adapters from test_analysis_service. No network,
no database.
"""

from __future__ import annotations

import json
from uuid import uuid4

import pytest
import pytest_asyncio
from cryptography.fernet import Fernet

import aity_web.services.analysis_service as svc
from _service_fixtures import StepClock, add_user
from aity_web.llm.catalog import ModelNotActiveError, UnknownModelError, UnknownProviderError, load_catalog
from aity_web.llm.registry import LLMProviderRegistry
from aity_web.llm.types import TokenUsage
from aity_web.models import SustainabilityMetrics
from aity_web.repositories import NotFoundError
from aity_web.repositories.memory import InMemoryStore
from aity_web.services.analysis_service import (
    AnalysisOutcome,
    AnalysisService,
    DocumentGoneError,
    DocumentTextError,
    MissingApiKeyError,
    outcome_to_result,
)
from aity_web.services.apikey_service import ApiKeyService, StoredKeyUnreadableError
from aity_web.services.cache_key import CACHE_KEY_VERSION, clean_text, content_hash, text_sha256
from aity_web.services.crypto import FernetCipher
from aity_web.services.document_service import DocumentService
from aity_web.services.upload_storage import UploadStorage
from test_analysis_service import GOOD, TRACK, FakeGemini, FakeOllama, FakeShim, _resp

pytestmark = pytest.mark.asyncio

RAW = b"Quantum software libraries  \r\n\r\n\r\ngive developers abstractions over circuits.\r\n"
CLEANED = clean_text(RAW.decode())


@pytest.fixture(autouse=True)
def no_ecologits(monkeypatch):
    monkeypatch.setattr(svc, "init_ecologits", lambda: None)


class World:
    """Alice with one session and one uploaded TXT; Bob with nothing."""

    def __init__(self, tmp_path, *, gemini=None, ollama=None, pinned=None):
        self.store = InMemoryStore(clock=StepClock())
        self.storage = UploadStorage(tmp_path / "uploads")
        self.apikeys = ApiKeyService(self.store.uow_factory, FernetCipher(Fernet.generate_key()))
        self.registry = LLMProviderRegistry(load_catalog(), pinned=pinned)
        self.gem = FakeGemini(gemini if gemini is not None else _resp())
        self.oll = FakeOllama(ollama if ollama is not None else _resp())
        self.registry.register(FakeGemini, factory=lambda: self.gem)
        self.registry.register(FakeOllama, factory=lambda: self.oll)
        self.events: list[str] = []
        self.shim = FakeShim(self.events, TRACK)
        self.clock = StepClock()
        self.open_uows = 0
        self.service = AnalysisService(
            self._counting_factory, self.storage, self.apikeys,
            registry=self.registry, shim=self.shim, clock=self.clock,
        )
        self.documents = DocumentService(self.store.uow_factory, self.storage, clock=StepClock())

    def _counting_factory(self):
        """uow_factory that tracks how many units of work are open."""
        return _Counting(self.store.uow_factory(), self)

    async def setup(self, *, with_key=True):
        self.alice = await add_user(self.store, "alice")
        self.bob = await add_user(self.store, "bob")
        async with self.store.uow_factory() as uow:
            self.session = await uow.sessions.add(user_id=self.alice.id, name="SMS pilot")
            await uow.commit()
        self.doc = await self.documents.upload(
            user_id=self.alice.id, session_id=self.session.id, filename="paper.txt", data=RAW
        )
        if with_key:
            await self.apikeys.set_key(user_id=self.alice.id, provider="gemini", api_key="sk-test-123")
        return self

    async def run(self, provider="gemini", model="gemini-3.5-flash-lite", user=None, doc=None):
        return await self.service.run_analysis(
            user_id=(user or self.alice).id, document_id=(doc or self.doc).id,
            provider=provider, model=model,
        )

    async def stored(self):
        async with self.store.uow_factory() as uow:
            return await uow.analyses.list_for_session(user_id=self.alice.id, session_id=self.session.id)


class _Counting:
    """Wraps a unit of work; `async with` goes through the world's counter."""

    def __init__(self, uow, world):
        self._uow, self._world = uow, world

    async def __aenter__(self):
        self._world.open_uows += 1
        return await self._uow.__aenter__()

    async def __aexit__(self, *exc):
        self._world.open_uows -= 1
        return await self._uow.__aexit__(*exc)


@pytest_asyncio.fixture
async def world(tmp_path):
    return await World(tmp_path).setup()


# --- the end-to-end path ---


async def test_upload_analyse_store(world):
    result = await world.run()

    assert result.status == "ok" and result.error is None
    assert (result.user_id, result.session_id, result.document_id) == (
        world.alice.id, world.session.id, world.doc.id)
    assert (result.provider, result.model, result.prompt_variant) == (
        "gemini", "gemini-3.5-flash-lite", "default_analysis")
    assert result.payload.summary == "A short summary."
    assert result.token_usage == TokenUsage(prompt_tokens=100, output_tokens=50)
    assert result.cache_key_version == CACHE_KEY_VERSION
    assert result.content_hash == content_hash(
        text_sha256=text_sha256(CLEANED), provider="gemini",
        model="gemini-3.5-flash-lite", prompt_variant="default_analysis")
    assert result.truncated is False and result.served_from_analysis_id is None
    assert result.finished_at > result.created_at

    assert await world.stored() == [result]
    assert await world.service.get(user_id=world.alice.id, analysis_id=result.analysis_id) == result


async def test_adapter_gets_the_cleaned_text_and_the_saved_key(world):
    await world.run()
    (call,) = world.gem.calls
    assert call["api_key"] == "sk-test-123"
    user_turn = call["messages"][-1]["content"]
    assert CLEANED in user_turn and "\r" not in user_turn


async def test_session_updated_at_is_bumped(world):
    before = world.session.updated_at
    await world.run()
    async with world.store.uow_factory() as uow:
        after = (await uow.sessions.get(user_id=world.alice.id, session_id=world.session.id)).updated_at
    assert after > before


async def test_no_transaction_is_open_during_the_call(world):
    seen = []
    original = world.gem.complete

    async def complete(**kw):
        seen.append(world.open_uows)
        return await original(**kw)

    world.gem.complete = complete
    await world.run()
    assert seen == [0]
    assert world.open_uows == 0


async def test_two_runs_are_two_rows_with_the_same_content_hash(world):
    a = await world.run()
    b = await world.run()
    assert a.analysis_id != b.analysis_id and a.content_hash == b.content_hash
    assert [r.analysis_id for r in await world.stored()] == [b.analysis_id, a.analysis_id]


async def test_content_hash_depends_on_the_model(world):
    a = await world.run(model="gemini-3.5-flash-lite")
    b = await world.run(model="gemini-3.6-flash")
    assert a.content_hash != b.content_hash


# --- partial and failed rows are stored too ---


async def test_partial_is_stored_with_the_issues(tmp_path):
    w = await World(tmp_path, gemini=_resp({**GOOD, "keywords": ["a", "b"]})).setup()
    result = await w.run()
    assert result.status == "partial" and result.payload is not None
    assert result.error == "keywords: 2 (expected 5–10)"
    assert await w.stored() == [result]


async def test_provider_exception_is_a_stored_failed_row(tmp_path):
    w = await World(tmp_path, gemini=RuntimeError("401 key sk-test-123 rejected")).setup()
    result = await w.run()
    assert result.status == "failed" and result.payload is None
    assert result.error == "provider call failed (RuntimeError)"
    assert result.token_usage is None
    assert result.sustainability.estimate_status == "unavailable"
    assert "sk-test" not in result.model_dump_json()
    assert await w.stored() == [result]


async def test_truncated_output_is_failed_but_keeps_its_usage(tmp_path):
    w = await World(tmp_path, gemini=_resp(finish="length")).setup()
    result = await w.run()
    assert result.status == "failed" and result.error == "output hit the token limit"
    assert result.token_usage == TokenUsage(prompt_tokens=100, output_tokens=50)


# --- AITY Built-in ---


async def test_builtin_needs_no_key_and_uses_the_shim(tmp_path):
    w = await World(tmp_path).setup(with_key=False)

    async def no_keys(**kw):
        raise AssertionError("ApiKeyService must not be consulted for ollama")

    w.apikeys.get_key = no_keys
    result = await w.run(provider="ollama", model="qwen3.5:4b")
    assert result.status == "ok" and result.provider == "ollama"
    assert w.events == ["start", "stop"]
    assert w.oll.calls[0]["api_key"] is None
    assert result.sustainability.estimation_source == "aity_cpu_model"


async def test_inactive_builtin_model_is_refused_before_anything(tmp_path):
    w = await World(tmp_path, pinned={"ollama": "qwen3.5:4b"}).setup()
    with pytest.raises(ModelNotActiveError):
        await w.run(provider="ollama", model="gemma3:4b")
    assert w.oll.calls == [] and w.events == [] and await w.stored() == []


# --- refused before the call: nothing stored, adapter untouched ---


@pytest.mark.parametrize(
    "provider, model, exc",
    [
        ("gemini", "gemini-0", UnknownModelError),
        ("openai", "gpt-5.4-mini-2026-03-17", UnknownProviderError),  # not registered here
        ("bard", "x", UnknownProviderError),
    ],
)
async def test_bad_provider_or_model(world, provider, model, exc):
    with pytest.raises(exc):
        await world.run(provider=provider, model=model)
    assert world.gem.calls == [] and await world.stored() == []


async def test_foreign_or_missing_document_is_not_found(world):
    with pytest.raises(NotFoundError):
        await world.run(user=world.bob)
    with pytest.raises(NotFoundError):
        await world.service.run_analysis(
            user_id=world.alice.id, document_id=uuid4(), provider="gemini",
            model="gemini-3.5-flash-lite")
    assert world.gem.calls == []


async def test_missing_key(tmp_path):
    w = await World(tmp_path).setup(with_key=False)
    with pytest.raises(MissingApiKeyError, match="no gemini API key"):
        await w.run()
    assert w.gem.calls == [] and await w.stored() == []


async def test_unreadable_stored_key(world):
    async with world.store.uow_factory() as uow:
        await uow.credentials.upsert(user_id=world.alice.id, provider="gemini", ciphertext=b"junk")
        await uow.commit()
    with pytest.raises(StoredKeyUnreadableError):
        await world.run()
    assert world.gem.calls == []


async def test_missing_text_file(world):
    world.storage.resolve(world.doc.text_path).unlink()
    with pytest.raises(DocumentTextError, match="could not be read"):
        await world.run()
    assert world.gem.calls == [] and await world.stored() == []


async def test_text_file_changed_on_disk(world):
    world.storage.resolve(world.doc.text_path).write_bytes(b"something else entirely")
    with pytest.raises(DocumentTextError, match="changed since upload"):
        await world.run()
    assert world.gem.calls == []


async def test_text_file_reformatted_only_still_matches(world):
    """The hash is over clean_text(), so CRLF -> LF on disk is not a change."""
    path = world.storage.resolve(world.doc.text_path)
    path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n"))
    assert (await world.run()).status == "ok"


# --- the document disappears while the model works ---


async def test_document_deleted_during_the_call_drops_the_result(world, caplog):
    original = world.gem.complete

    async def complete(**kw):
        await world.documents.delete(user_id=world.alice.id, document_id=world.doc.id)
        return await original(**kw)

    world.gem.complete = complete
    with pytest.raises(DocumentGoneError) as info:
        await world.run()
    assert isinstance(info.value, NotFoundError)
    assert await world.stored() == []
    assert "dropped" in caplog.text


# --- list / delete ---


async def test_list_and_delete_are_owner_only(world):
    r = await world.run()
    assert await world.service.list_for_document(user_id=world.alice.id, document_id=world.doc.id) == [r]
    with pytest.raises(NotFoundError):
        await world.service.list_for_document(user_id=world.bob.id, document_id=world.doc.id)
    with pytest.raises(NotFoundError):
        await world.service.get(user_id=world.bob.id, analysis_id=r.analysis_id)
    with pytest.raises(NotFoundError):
        await world.service.delete(user_id=world.bob.id, analysis_id=r.analysis_id)
    await world.service.delete(user_id=world.alice.id, analysis_id=r.analysis_id)
    assert await world.stored() == []


# --- outcome_to_result on its own ---


def _outcome(**kw):
    base = dict(
        status="ok", provider="gemini", model="m", prompt_variant="default_analysis",
        prompt_profile="standard",
        sustainability=SustainabilityMetrics.unavailable(latency_seconds=1.0),
        payload=None, usage=TokenUsage(prompt_tokens=1, output_tokens=2), finish_reason="stop",
    )
    base.update(kw)
    return AnalysisOutcome(**base)


def _map(outcome, doc):
    from datetime import datetime, timedelta, timezone

    t0 = datetime(2026, 11, 26, tzinfo=timezone.utc)
    return outcome_to_result(
        outcome, analysis_id=uuid4(), document=doc, content_hash="e" * 64,
        created_at=t0, finished_at=t0 + timedelta(seconds=3),
    )


async def test_mapping_table(world):
    from aity_web.models import LLMAnalysisPayload

    payload = LLMAnalysisPayload.model_validate(json.loads(json.dumps(GOOD)))
    ok = _map(_outcome(payload=payload), world.doc)
    assert ok.status == "ok" and ok.error is None and ok.token_usage.total_tokens == 3

    partial = _map(_outcome(status="partial", payload=payload, issues=("a: 1", "b: 2")), world.doc)
    assert partial.error == "a: 1; b: 2"

    raised = _map(_outcome(status="failed", error="provider call failed (X)", finish_reason=None,
                           usage=TokenUsage()), world.doc)
    assert raised.token_usage is None and raised.payload is None

    answered = _map(_outcome(status="failed", error="output was not valid JSON"), world.doc)
    assert answered.token_usage.output_tokens == 2

    assert {ok.session_id, ok.user_id, ok.document_id} == {
        world.doc.session_id, world.doc.user_id, world.doc.id}
