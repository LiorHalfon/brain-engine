from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from brain_engine.graph import GraphitiBackend
from brain_engine.service import MemoryService
from brain_engine.store import MemoryStore, content_sha256

errors = pytest.importorskip("graphiti_core.errors")


class FakeGraphiti:
    """Stands in for the Graphiti client. It removes and adds episodes the way Graphiti 0.31 does,
    including NodeNotFoundError for an episode that is already gone, and yields at every call so
    concurrent ingests interleave, and notes any writes that overlap. Each episode's text is its one
    fact, so search finds a fact only while its episode is in the graph."""

    def __init__(self) -> None:
        self.episodes: dict[str, str] = {}
        self.sources: list[str] = []
        self.added = 0
        self.remove_error: Exception | None = None
        self.writers = 0
        self.writes_overlapped = False

    async def _write(self) -> None:
        self.writers += 1
        self.writes_overlapped = self.writes_overlapped or self.writers > 1
        await asyncio.sleep(0)
        self.writers -= 1

    async def remove_episode(self, episode_uuid: str) -> None:
        await self._write()
        if self.remove_error is not None:
            raise self.remove_error
        if episode_uuid not in self.episodes:
            raise errors.NodeNotFoundError(episode_uuid)
        del self.episodes[episode_uuid]

    async def add_episode(self, *, name: str, episode_body: str, **kwargs) -> SimpleNamespace:
        await self._write()
        self.added += 1
        self.sources.append(kwargs["source_description"])
        episode_uuid = f"ep-{self.added}"
        self.episodes[episode_uuid] = episode_body
        return SimpleNamespace(episode=SimpleNamespace(uuid=episode_uuid), edges=[])

    async def search(self, query: str) -> list:
        await asyncio.sleep(0)
        terms = query.casefold().split()
        return [
            SimpleNamespace(uuid=f"fact-{uuid}", fact=body.strip(), valid_at=None, invalid_at=None, episodes=[uuid])
            for uuid, body in self.episodes.items()
            if any(term in body.casefold() for term in terms)
        ]


def run(coro):
    return asyncio.run(coro)


def make_backend(tmp_path: Path) -> tuple[MemoryStore, GraphitiBackend, FakeGraphiti]:
    store = MemoryStore(tmp_path / "memory", actor="test/1", human_id="owner")
    backend = GraphitiBackend(store.state_dir, name="test-vault")
    client = FakeGraphiti()
    backend._client = client
    return store, backend, client


def ingested_then_hand_edited(store: MemoryStore, backend: GraphitiBackend) -> str:
    created = store.remember(title="Fix plan", content="The first draft names someone.", kind="note")
    run(backend.ingest(store.record(created["id"])))
    store.edit(created["id"], "The scrubbed draft names no one.")
    return created["id"]


def test_episode_source_is_the_vault_name_and_path(tmp_path: Path) -> None:
    store, backend, client = make_backend(tmp_path)
    created = store.remember(title="Carrier", content="Acme ships orders with Parcelco.", kind="note")

    run(backend.ingest(store.record(created["id"])))

    assert client.sources == [f"test-vault:{created['path']}"]


def test_backend_hands_its_models_to_the_llm_client(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "unused")  # building the client makes no OpenAI calls

    chosen = GraphitiBackend(tmp_path / "a", model="m-big", small_model="m-small")._llm_client()
    default = GraphitiBackend(tmp_path / "b")._llm_client()

    assert (chosen.model, chosen.small_model) == ("m-big", "m-small")
    assert (default.model, default.small_model) == ("gpt-6-luna", "gpt-6-luna")


def test_ingest_adds_new_episode_when_old_one_is_already_gone(tmp_path: Path) -> None:
    store, backend, client = make_backend(tmp_path)
    memory_id = ingested_then_hand_edited(store, backend)
    client.episodes.clear()  # e.g. removed by another caller, or the graph was rebuilt

    result = run(backend.ingest(store.record(memory_id)))

    assert result["indexed"] is True
    assert client.episodes == {result["episode_uuid"]: "The scrubbed draft names no one.\n"}
    state = backend._state(memory_id)
    assert state["status"] == "ok" and state["episode_uuid"] == result["episode_uuid"]
    # The next sync skips it instead of failing again.
    assert run(backend.ingest(store.record(memory_id)))["skipped"] is True


def test_ingest_records_remove_failure_and_retries_the_removal(tmp_path: Path) -> None:
    store, backend, client = make_backend(tmp_path)
    memory_id = ingested_then_hand_edited(store, backend)
    old_episode = backend._state(memory_id)["episode_uuid"]
    client.remove_error = RuntimeError("falkordb is down")

    with pytest.raises(RuntimeError, match="falkordb is down"):
        run(backend.ingest(store.record(memory_id)))

    state = backend._state(memory_id)
    assert state["status"] == "error" and "falkordb is down" in state["last_error"]
    # The old episode is still in the graph, so the next attempt must still remove it.
    assert state["episode_uuid"] == old_episode
    assert list(client.episodes) == [old_episode]

    client.remove_error = None
    result = run(backend.ingest(store.record(memory_id)))
    assert list(client.episodes) == [result["episode_uuid"]]
    assert backend._state(memory_id)["status"] == "ok"


def test_concurrent_ingests_of_one_memory_leave_one_episode(tmp_path: Path) -> None:
    store, backend, client = make_backend(tmp_path)
    memory_id = ingested_then_hand_edited(store, backend)
    record = store.record(memory_id)

    async def both():
        # edit_memory and a `just pull` sync landing on the same memory at once.
        return await asyncio.gather(backend.ingest(record), backend.ingest(record))

    first, second = run(both())

    assert second.get("skipped") is True
    assert client.episodes == {first["episode_uuid"]: "The scrubbed draft names no one.\n"}
    assert backend._state(memory_id)["episode_uuid"] == first["episode_uuid"]


def test_sync_removes_the_episode_of_a_deleted_memory(tmp_path: Path) -> None:
    # A memory file deleted by hand, or by a `git pull`, must take its facts out of the graph too.
    store, backend, client = make_backend(tmp_path)
    service = MemoryService(store, backend)
    kept = store.remember(title="Carrier", content="Acme ships orders with Parcelco.", kind="note")
    deleted = store.remember(title="Office", content="The Springfield office opens at 9.", kind="note")
    assert run(service.sync_graph())["indexed"] == 2
    assert [f["fact"] for f in run(service.search("Springfield"))["facts"]] == ["The Springfield office opens at 9."]
    episode = backend._state(deleted["id"])["episode_uuid"]

    (store.root / deleted["path"]).unlink()
    result = run(service.sync_graph())

    assert result == {"indexed": 0, "removed": 1, "skipped": 1, "errors": []}
    assert episode not in client.episodes and backend._state(deleted["id"]) is None
    assert run(service.search("Springfield"))["facts"] == []
    assert [f["source_memory_ids"] for f in run(service.search("Parcelco"))["facts"]] == [[kept["id"]]]
    assert run(service.sync_graph()) == {"indexed": 0, "removed": 0, "skipped": 1, "errors": []}


def test_sync_forgets_deleted_memories_that_have_no_episode_left(tmp_path: Path) -> None:
    store, backend, client = make_backend(tmp_path)
    service = MemoryService(store, backend)
    store.remember(title="Mascot", content="Acme's mascot is a heron named Pip.", kind="note")
    gone = store.remember(title="Carrier", content="Acme ships orders with Parcelco.", kind="note")
    failed = store.remember(title="Office", content="The Springfield office opens at 9.", kind="note")
    run(service.sync_graph())
    client.episodes.pop(backend._state(gone["id"])["episode_uuid"])  # e.g. dropped by a rebuilt graph
    backend._write_state(failed["id"], "hash", episode_uuid=None, status="error", error="rate limited")

    for created in (gone, failed):
        (store.root / created["path"]).unlink()

    assert run(service.sync_graph()) == {"indexed": 0, "removed": 2, "skipped": 1, "errors": []}
    assert backend._state(gone["id"]) is None and backend._state(failed["id"]) is None


def test_a_failed_removal_is_tried_again_by_the_next_sync(tmp_path: Path) -> None:
    store, backend, client = make_backend(tmp_path)
    service = MemoryService(store, backend)
    store.remember(title="Carrier", content="Acme ships orders with Parcelco.", kind="note")
    deleted = store.remember(title="Office", content="The Springfield office opens at 9.", kind="note")
    run(service.sync_graph())
    (store.root / deleted["path"]).unlink()
    client.remove_error = RuntimeError("falkordb is down")

    result = run(service.sync_graph())

    assert result["removed"] == 0
    assert result["errors"] == [{"memory_id": deleted["id"], "error": "falkordb is down"}]
    assert backend._state(deleted["id"]) is not None
    client.remove_error = None
    assert run(service.sync_graph())["removed"] == 1
    assert run(service.search("Springfield"))["facts"] == []


def test_a_removal_waits_for_an_ingest_of_the_same_memory(tmp_path: Path) -> None:
    store, backend, client = make_backend(tmp_path)
    created = store.remember(title="Carrier", content="Acme ships orders with Parcelco.", kind="note")
    record = store.record(created["id"])

    async def both() -> None:
        # A queued ingest still running when a sync finds the memory's file gone.
        await asyncio.gather(backend.ingest(record), backend.remove(created["id"]))

    run(both())

    assert client.episodes == {} and backend._state(created["id"]) is None


def test_graph_writes_run_one_at_a_time(tmp_path: Path) -> None:
    # Graphiti reads the nodes and edges an episode touches, then saves them back by uuid. A second
    # writer in between can bring back what the first deleted, or lose what it added.
    store, backend, client = make_backend(tmp_path)
    deleted = store.remember(title="Carrier", content="Acme ships orders with Parcelco.", kind="note")
    first = store.remember(title="Office", content="The Springfield office opens at 9.", kind="note")
    second = store.remember(title="Mascot", content="Acme's mascot is a heron named Pip.", kind="note")
    run(backend.ingest(store.record(deleted["id"])))

    async def all_at_once() -> None:
        await asyncio.gather(
            backend.remove(deleted["id"]),
            backend.ingest(store.record(first["id"])),
            backend.ingest(store.record(second["id"])),
        )

    run(all_at_once())

    assert client.writes_overlapped is False
    assert len(client.episodes) == 2


def test_overlapping_syncs_count_a_removal_once(tmp_path: Path) -> None:
    store, backend, client = make_backend(tmp_path)
    service = MemoryService(store, backend)
    store.remember(title="Carrier", content="Acme ships orders with Parcelco.", kind="note")
    deleted = store.remember(title="Office", content="The Springfield office opens at 9.", kind="note")
    run(service.sync_graph())
    (store.root / deleted["path"]).unlink()

    async def both() -> list[dict]:
        return await asyncio.gather(service.sync_graph(), service.sync_graph())

    first, second = run(both())

    assert first["removed"] + second["removed"] == 1


async def test_close_saves_the_graph_and_stops_its_redis_server(tmp_path: Path, monkeypatch) -> None:
    # FalkorDB Lite runs a redis-server process that saves graphiti.db only on a timer. Nothing
    # stopped it, so it either outlived the service or was SIGKILLed at exit, losing the writes
    # since its last save.
    psutil = pytest.importorskip("psutil")
    monkeypatch.setenv("OPENAI_API_KEY", "unused")  # opening the graph makes no OpenAI calls
    backend = GraphitiBackend(tmp_path)
    await backend.start()
    pid = backend._client.driver.client.connection.pid
    try:
        await backend._client.driver.execute_query("CREATE (:Probe {name: 'kept'})")
        await backend.close()
        # The server drops its connection as it exits. On Linux its pid then stays in the process
        # table, as a zombie, until init reaps it.
        deadline = time.monotonic() + 5
        while psutil.pid_exists(pid) and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        assert not psutil.pid_exists(pid), "redis-server still running after close"
    finally:
        if psutil.pid_exists(pid):
            psutil.Process(pid).kill()

    reopened = GraphitiBackend(tmp_path)
    await reopened.start()
    try:
        records, _, _ = await reopened._client.driver.execute_query("MATCH (p:Probe) RETURN p.name AS name")
    finally:
        await reopened.close()
    assert [record["name"] for record in records] == ["kept"]


async def test_memories_whose_episode_is_gone_are_ingested_again(tmp_path: Path, monkeypatch, caplog) -> None:
    # A leftover redis-server that wrote an older graph over graphiti.db leaves memories marked ok
    # in graph-state whose episodes are no longer in the graph.
    pytest.importorskip("redislite")
    monkeypatch.setenv("OPENAI_API_KEY", "unused")  # opening the graph makes no OpenAI calls
    store = MemoryStore(tmp_path / "memory", actor="test/1", human_id="owner")
    kept = store.record(store.remember(title="Kept", content="Acme ships orders with Parcelco.")["id"])
    lost = store.record(store.remember(title="Lost", content="The Springfield office opens at 9.")["id"])
    backend = GraphitiBackend(store.state_dir)
    await backend.start()
    await backend._client.driver.execute_query("CREATE (:Episodic {uuid: 'ep-kept'})")
    backend._write_state(kept.id, content_sha256("note", kept.content), episode_uuid="ep-kept", status="ok")
    backend._write_state(lost.id, content_sha256("note", lost.content), episode_uuid="ep-lost", status="ok")
    await backend.close()

    reopened = GraphitiBackend(store.state_dir)
    with caplog.at_level(logging.WARNING, logger="brain_engine.graph"):
        await reopened.start()
    await reopened.close()

    assert reopened._state(kept.id)["status"] == "ok"
    assert reopened._state(lost.id)["status"] == "missing" and reopened._state(lost.id)["episode_uuid"] is None
    assert "ingested again: 1" in caplog.text
    reopened._client = FakeGraphiti()
    assert (await reopened.ingest(kept))["skipped"] is True
    assert (await reopened.ingest(lost)).get("skipped") is None
    assert reopened._state(lost.id)["status"] == "ok"


class SearchGraphiti:
    """Graphiti's two searches: the default hybrid one, which waits on OpenAI to embed the query,
    and `search_` with a config, here only ever asked for the keyword index."""

    def __init__(self, embed_seconds: float) -> None:
        self.embed_seconds = embed_seconds
        self.configs: list = []

    @staticmethod
    def _edge(fact: str) -> SimpleNamespace:
        return SimpleNamespace(uuid=fact, fact=fact, valid_at=None, invalid_at=None, episodes=[])

    async def search(self, query: str) -> list:
        await asyncio.sleep(self.embed_seconds)
        return [self._edge("hybrid match")]

    async def search_(self, query: str, config) -> SimpleNamespace:
        self.configs.append(config)
        return SimpleNamespace(edges=[self._edge("keyword match")])


def test_search_uses_the_hybrid_search_when_openai_answers_in_time(tmp_path: Path) -> None:
    backend = GraphitiBackend(tmp_path / "state")
    backend._client = SearchGraphiti(embed_seconds=0)

    facts = run(backend.search("office hours"))

    assert [fact.fact for fact in facts] == ["hybrid match"]


def test_search_falls_back_to_keywords_when_openai_is_slow(tmp_path: Path) -> None:
    from graphiti_core.search.search_config import EdgeSearchMethod

    backend = GraphitiBackend(tmp_path / "state")
    client = SearchGraphiti(embed_seconds=30)
    backend._client = client

    async def timed() -> tuple[list, float]:
        start = asyncio.get_running_loop().time()
        facts = await backend.search("office hours")
        return facts, asyncio.get_running_loop().time() - start

    facts, seconds = run(timed())

    assert [fact.fact for fact in facts] == ["keyword match"]
    assert seconds < 1
    assert client.configs[0].edge_config.search_methods == [EdgeSearchMethod.bm25]
