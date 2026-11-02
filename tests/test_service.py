from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

from brain_engine import service as service_module
from brain_engine import store as store_module
from brain_engine.graph import GraphFact
from brain_engine.service import MemoryService
from brain_engine.store import MemoryRecord, MemoryStore


class FakeGraph:
    enabled = True

    def __init__(self) -> None:
        self.ingested: list[str] = []
        self.records: dict[str, MemoryRecord] = {}
        self.fail_search = False
        self.started = 0

    async def start(self) -> None:
        self.started += 1

    async def ingest(self, record: MemoryRecord) -> dict:
        self.ingested.append(record.id)
        self.records[record.id] = record
        return {"indexed": True, "memory_id": record.id, "episode_uuid": f"ep-{record.id}"}

    async def search(self, query: str, limit: int = 5, include_invalidated: bool = False) -> list[GraphFact]:
        if self.fail_search:
            raise RuntimeError("graph unavailable")
        return [
            GraphFact(
                uuid="fact-1",
                fact="Acme ships orders with Parcelco.",
                valid_at="2026-09-01T00:00:00+00:00",
                invalid_at=None,
                source_memory_ids=list(self.records)[:1],
            )
        ][:limit]

    async def remove(self, memory_id: str) -> bool:
        return self.records.pop(memory_id, None) is not None

    def memory_ids(self) -> set[str]:
        return set(self.records)

    async def snapshot(self) -> dict:
        if self.fail_search:
            raise RuntimeError("graph unavailable")
        ids = list(self.records)
        nodes = [{"uuid": "n-acme", "name": "Acme", "summary": ""}, {"uuid": "n-parcelco", "name": "Parcelco", "summary": ""}]
        edges = [
            {
                "uuid": f"e-{memory_id}",
                "source": "n-acme",
                "target": "n-parcelco",
                "name": "SHIPS_WITH",
                "fact": "Acme ships orders with Parcelco.",
                "valid_at": None,
                "invalid_at": None,
                "source_memory_ids": [memory_id],
            }
            for memory_id in ids
        ]
        return {"enabled": True, "nodes": nodes, "edges": edges}

    async def close(self) -> None:
        return None


def run(coro):
    return asyncio.run(coro)


def _rows(service: MemoryService) -> list[sqlite3.Row]:
    with sqlite3.connect(service.metrics.db_path) as conn:
        conn.row_factory = sqlite3.Row
        return conn.execute("SELECT * FROM operations ORDER BY rowid").fetchall()


def test_exact_duplicate_does_not_reingest(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory", actor="test/1", human_id="owner")
    graph = FakeGraph()
    service = MemoryService(store, graph)

    first = run(
        service.remember(
            title="Opening hours",
            content="The Springfield office opens at 9.",
            kind="note",
        )
    )
    second = run(
        service.remember(
            title="Same fact with another title",
            content="The Springfield office opens at 9.",
            kind="note",
        )
    )

    assert first["duplicate"] is False
    assert second["duplicate"] is True
    assert second["id"] == first["id"]
    assert graph.ingested == [first["id"]]


def test_correction_preserves_history_and_hides_superseded_from_default_search(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory", actor="test/1", human_id="owner")
    graph = FakeGraph()
    service = MemoryService(store, graph)

    old = run(
        service.remember(
            title="Opening hours",
            content="The Springfield office opens at 9.",
            kind="note",
        )
    )
    correction = run(
        service.correct(
            old["id"],
            correction="The Springfield office opens at 10.",
            reason="Hours changed",
        )
    )

    old_record = store.record(old["id"])
    new_record = store.record(correction["id"])
    assert old_record.metadata["status"] == "superseded"
    assert old_record.metadata["superseded_by"] == correction["id"]
    assert new_record.metadata["corrects"] == old["id"]

    default_results = store.search("Springfield office opens")
    assert [r.id for r in default_results] == [correction["id"]]

    history_results = store.search("Springfield office opens", include_superseded=True)
    assert {r.id for r in history_results} == {old["id"], correction["id"]}

    # Correction is a new graph episode; old graph episode is deliberately preserved so Graphiti
    # can model temporal invalidation instead of erasing history.
    assert graph.ingested == [old["id"], correction["id"]]


def test_search_returns_fts_when_graph_fails(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory", actor="test/1", human_id="owner")
    graph = FakeGraph()
    service = MemoryService(store, graph)

    created = run(
        service.remember(
            title="Memory architecture",
            content="Markdown remains the source of truth.",
            kind="note",
        )
    )
    graph.fail_search = True

    result = run(service.search("Markdown source truth"))
    assert result["facts"] == []
    assert result["memories"][0]["id"] == created["id"]
    assert "graph unavailable" in result["graph_error"]


def test_edit_reingests_same_memory_id(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory", actor="test/1", human_id="owner")
    graph = FakeGraph()
    service = MemoryService(store, graph)

    created = run(
        service.remember(
            title="Product detail",
            content="The stored text has a typo.",
            kind="note",
        )
    )
    edited = run(service.edit(created["id"], "The stored text is corrected.", reason="typo"))

    assert edited["id"] == created["id"]
    assert graph.ingested == [created["id"], created["id"]]
    assert store.record(created["id"]).content == "The stored text is corrected.\n"


def test_sync_keeps_the_episodes_of_superseded_memories(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory", actor="test/1", human_id="owner")
    graph = FakeGraph()
    service = MemoryService(store, graph)
    old = run(service.remember(title="Opening hours", content="The Springfield office opens at 9.", kind="note"))
    new = run(service.correct(old["id"], correction="The Springfield office opens at 10.", reason="Hours changed"))

    result = run(service.sync_graph())

    # Only a deleted file removes an episode. A superseded memory's episode is the graph's history.
    assert result["removed"] == 0 and graph.memory_ids() == {old["id"], new["id"]}


def _remembered(service: MemoryService, count: int) -> list[dict]:
    return [run(service.remember(title=f"Fact {i}", content=f"Fact number {i}.", kind="note")) for i in range(count)]


def test_sync_removes_nothing_when_the_markdown_folder_is_empty(tmp_path: Path) -> None:
    # A wrong MEMORY_ROOT reads as an empty vault. Removing every episode would cost a full rebuild.
    store = MemoryStore(tmp_path / "memory", actor="test/1", human_id="owner")
    graph = FakeGraph()
    service = MemoryService(store, graph)
    created = _remembered(service, 2)
    for memory in created:
        (store.root / memory["path"]).unlink()

    result = run(service.sync_graph())

    message = f"not removed: {store.root} holds no memories"
    assert result == {"indexed": 0, "removed": 0, "skipped": 0, "errors": [{"error": message}]}
    assert graph.memory_ids() == {memory["id"] for memory in created}
    assert _rows(service)[-1]["error"] == message

    assert run(service.sync_graph(allow_mass_removal=True))["removed"] == 2
    assert graph.memory_ids() == set()


def test_sync_removes_up_to_the_limit_without_being_told_to(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(service_module, "MAX_UNCONFIRMED_REMOVALS", 2)
    store = MemoryStore(tmp_path / "memory", actor="test/1", human_id="owner")
    graph = FakeGraph()
    service = MemoryService(store, graph)
    created = _remembered(service, 3)
    for memory in created[:2]:
        (store.root / memory["path"]).unlink()

    result = run(service.sync_graph())

    assert result["removed"] == 2 and result["errors"] == []
    assert graph.memory_ids() == {created[2]["id"]}


def test_sync_keeps_the_graph_past_the_limit_until_told_to_go_ahead(tmp_path: Path, monkeypatch) -> None:
    # A checkout on an older commit, or a MEMORY_ROOT at a subfolder, still holds some memories. Each
    # one removed by mistake costs a new ingest once the folder is right.
    monkeypatch.setattr(service_module, "MAX_UNCONFIRMED_REMOVALS", 2)
    store = MemoryStore(tmp_path / "memory", actor="test/1", human_id="owner")
    graph = FakeGraph()
    service = MemoryService(store, graph)
    created = _remembered(service, 4)
    for memory in created[:3]:
        (store.root / memory["path"]).unlink()

    result = run(service.sync_graph())

    assert result["removed"] == 0 and graph.memory_ids() == {memory["id"] for memory in created}
    assert len(result["errors"]) == 1 and "--allow-mass-removal" in result["errors"][0]["error"]

    assert run(service.sync_graph(allow_mass_removal=True)) == {"indexed": 1, "removed": 3, "skipped": 0, "errors": []}
    assert graph.memory_ids() == {created[3]["id"]}


def test_a_file_that_no_longer_parses_is_not_taken_for_a_deleted_memory(tmp_path: Path) -> None:
    # Conflict markers or a YAML typo in the frontmatter. Fixing the file must not cost a new ingest.
    store = MemoryStore(tmp_path / "memory", actor="test/1", human_id="owner")
    graph = FakeGraph()
    service = MemoryService(store, graph)
    broken, deleted, other = _remembered(service, 3)
    path = store.root / broken["path"]
    good_text = path.read_text(encoding="utf-8")
    path.write_text("---\nid: [unclosed\n---\n\nSome text.\n", encoding="utf-8")
    (store.root / deleted["path"]).unlink()

    for go_ahead in (False, True):  # a broken file holds removals back either way
        result = run(service.sync_graph(allow_mass_removal=go_ahead))
        assert result["removed"] == 0
        assert len(result["errors"]) == 1 and broken["path"] in result["errors"][0]["error"]
    assert graph.memory_ids() == {broken["id"], deleted["id"], other["id"]}

    path.write_text(good_text, encoding="utf-8")
    result = run(service.sync_graph())

    assert result["removed"] == 1 and result["errors"] == []
    assert graph.memory_ids() == {broken["id"], other["id"]}


def test_a_stray_file_that_does_not_parse_is_no_error_while_nothing_looks_deleted(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory", actor="test/1", human_id="owner")
    service = MemoryService(store, FakeGraph())
    _remembered(service, 1)
    (store.root / "scratch.md").write_text("Notes without frontmatter.\n", encoding="utf-8")

    assert run(service.sync_graph())["errors"] == []


def test_search_hides_the_facts_of_a_memory_deleted_since_the_last_sync(tmp_path: Path) -> None:
    # The episode stays in the graph until a sync, but FTS has already dropped the memory.
    store = MemoryStore(tmp_path / "memory", actor="test/1", human_id="owner")
    graph = FakeGraph()
    service = MemoryService(store, graph)
    created = _remembered(service, 1)[0]
    assert run(service.search("Parcelco"))["facts"]

    (store.root / created["path"]).unlink()

    assert run(service.search("Parcelco"))["facts"] == []
    assert run(service.search("Parcelco", include_history=True))["facts"]
    assert [edge["current"] for edge in run(service.graph_snapshot())["edges"]] == [False]
    assert created["id"] in graph.memory_ids()


def test_search_keeps_a_fact_while_a_remaining_memory_supports_it(tmp_path: Path) -> None:
    class SharedFact(FakeGraph):
        async def search(self, query: str, limit: int = 5, include_invalidated: bool = False) -> list[GraphFact]:
            facts = await super().search(query, limit, include_invalidated)
            return [GraphFact(**{**fact.__dict__, "source_memory_ids": list(self.records)}) for fact in facts]

    store = MemoryStore(tmp_path / "memory", actor="test/1", human_id="owner")
    service = MemoryService(store, SharedFact())
    deleted, _ = _remembered(service, 2)
    (store.root / deleted["path"]).unlink()

    assert run(service.search("Parcelco"))["facts"]


def test_search_keeps_a_fact_with_no_known_source(tmp_path: Path) -> None:
    # Graph-state has no row for its episodes, so nothing says its source is gone.
    store = MemoryStore(tmp_path / "memory", actor="test/1", human_id="owner")

    assert run(MemoryService(store, FakeGraph()).search("Parcelco"))["facts"]


def test_graph_snapshot_reads_each_memory_once(tmp_path: Path, monkeypatch) -> None:
    store = MemoryStore(tmp_path / "memory", actor="test/1", human_id="owner")
    service = MemoryService(store, FakeGraph())
    for i in range(10):
        run(service.remember(title=f"Fact {i}", content=f"Fact number {i}.", kind="note"))
    parsed: list[Path] = []
    real_parse = store_module.parse_markdown
    monkeypatch.setattr(store_module, "parse_markdown", lambda path: parsed.append(path) or real_parse(path))

    snapshot = run(service.graph_snapshot())

    assert len(snapshot["memories"]) == 10 and all(e["current"] for e in snapshot["edges"])
    # One pass over the vault, not a scan per fact: with ~500 facts that took 10 seconds.
    assert len(parsed) == 10


def test_search_checks_fact_sources_without_rereading_the_vault(tmp_path: Path, monkeypatch) -> None:
    store = MemoryStore(tmp_path / "memory", actor="test/1", human_id="owner")
    service = MemoryService(store, FakeGraph())
    for i in range(10):
        run(service.remember(title=f"Fact {i}", content=f"Fact number {i}.", kind="note"))
    parsed: list[Path] = []
    real_parse = store_module.parse_markdown
    monkeypatch.setattr(store_module, "parse_markdown", lambda path: parsed.append(path) or real_parse(path))

    result = run(service.search("fact number"))

    assert result["facts"] and result["memories"]
    # The superseded check used to read the vault once per fact source.
    assert parsed == []


class BlockingGraph(FakeGraph):
    """Holds every ingest until `release` is set, so a test sees what a write returns first."""

    def __init__(self) -> None:
        super().__init__()
        self.release = asyncio.Event()

    async def ingest(self, record: MemoryRecord) -> dict:
        await self.release.wait()
        return await super().ingest(record)


def test_writes_return_before_the_graph_ingests_them(tmp_path: Path) -> None:
    async def scenario() -> None:
        store = MemoryStore(tmp_path / "memory", actor="test/1", human_id="owner")
        graph = BlockingGraph()
        service = MemoryService(store, graph)
        await service.start(ingest_in_background=True)

        created = await asyncio.wait_for(
            service.remember(title="Opening hours", content="The Springfield office opens at 9.", kind="note"), 1
        )
        assert created["graph"] == {"queued": True, "memory_id": created["id"]}
        assert graph.ingested == []

        graph.release.set()
        await asyncio.wait_for(service.graph_idle(), 1)
        assert graph.ingested == [created["id"]]
        await service.close()

    run(scenario())


def test_an_edit_while_queued_is_ingested_once_with_the_new_text(tmp_path: Path) -> None:
    async def scenario() -> None:
        store = MemoryStore(tmp_path / "memory", actor="test/1", human_id="owner")
        graph = BlockingGraph()
        service = MemoryService(store, graph)
        await service.start(ingest_in_background=True)

        first = await service.remember(title="First", content="The first memory.", kind="note")
        await asyncio.sleep(0)  # the worker takes the first memory and waits on the graph
        second = await service.remember(title="Second", content="A draft text.", kind="note")
        await service.edit(second["id"], "The final text.")
        graph.release.set()
        await asyncio.wait_for(service.graph_idle(), 1)

        assert graph.ingested == [first["id"], second["id"]]
        assert graph.records[second["id"]].content == "The final text.\n"
        await service.close()

    run(scenario())


def test_start_queues_the_memories_sync_would_ingest(tmp_path: Path) -> None:
    # Memories written while the service was down, pulled from git, or still queued when it stopped.
    store = MemoryStore(tmp_path / "memory", actor="test/1", human_id="owner")
    old = store.remember(title="Opening hours", content="The Springfield office opens at 9.", kind="note")
    new = store.correct(old["id"], "The Springfield office opens at 10.", reason="Hours changed")
    graph = FakeGraph()
    service = MemoryService(store, graph)

    async def scenario() -> None:
        await service.start(ingest_in_background=True)
        await asyncio.wait_for(service.graph_idle(), 1)
        await service.close()

    run(scenario())
    assert graph.ingested == [new["id"]]


def test_close_does_not_wait_for_a_running_ingest(tmp_path: Path) -> None:
    async def scenario() -> None:
        store = MemoryStore(tmp_path / "memory", actor="test/1", human_id="owner")
        graph = BlockingGraph()
        service = MemoryService(store, graph)
        await service.start(ingest_in_background=True)
        await service.remember(title="Opening hours", content="The Springfield office opens at 9.", kind="note")
        await asyncio.sleep(0)

        await asyncio.wait_for(service.close(), 1)
        assert graph.ingested == []
        assert [row["op"] for row in _rows(service)] == ["remember"]  # the cut-off ingest is not timed

    run(scenario())


def test_writes_ingest_inline_when_the_graph_did_not_start(tmp_path: Path) -> None:
    class BrokenGraph(FakeGraph):
        async def start(self) -> None:
            raise RuntimeError("OPENAI_API_KEY is required")

        async def ingest(self, record: MemoryRecord) -> dict:
            raise RuntimeError("OPENAI_API_KEY is required")

    async def scenario() -> dict:
        store = MemoryStore(tmp_path / "memory", actor="test/1", human_id="owner")
        service = MemoryService(store, BrokenGraph())
        await service.start(ingest_in_background=True)
        try:
            return await service.remember(title="Opening hours", content="The Springfield office opens at 9.", kind="note")
        finally:
            await service.close()

    # Queueing would only fail later, once per memory, where nobody sees it.
    assert run(scenario())["graph"] == {"indexed": False, "error": "OPENAI_API_KEY is required"}


def test_leftover_servers_stop_start_instead_of_falling_back(tmp_path: Path) -> None:
    import pytest

    from brain_engine.leftovers import LeftoverServersError

    class Contested(FakeGraph):
        async def start(self) -> None:
            raise LeftoverServersError("2 leftover redis-servers hold unsaved writes")

    service = MemoryService(MemoryStore(tmp_path / "memory", actor="test/1", human_id="owner"), Contested())

    with pytest.raises(LeftoverServersError):
        run(service.start(ingest_in_background=True))
