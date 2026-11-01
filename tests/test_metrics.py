from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

import pytest

from brain_engine.metrics import Metrics
from brain_engine.service import MemoryService
from brain_engine.store import MemoryRecord, MemoryStore
from test_service import FakeGraph, run


class SlowGraph(FakeGraph):
    async def ingest(self, record: MemoryRecord) -> dict:
        await asyncio.sleep(0.05)
        return await super().ingest(record)


def _rows(service: MemoryService) -> list[sqlite3.Row]:
    with sqlite3.connect(service.metrics.db_path) as conn:
        conn.row_factory = sqlite3.Row
        return conn.execute("SELECT * FROM operations ORDER BY rowid").fetchall()


def test_each_operation_records_its_time_split_into_store_and_graph(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory", actor="test/1", human_id="owner")
    service = MemoryService(store, SlowGraph())

    created = run(service.remember(title="Opening hours", content="The Springfield office opens at 9.", kind="note"))
    service.get(created["id"])
    run(service.search("office opens"))

    remember, get, search = _rows(service)
    assert remember["op"] == "remember"
    assert remember["graph_ms"] >= 50
    assert remember["store_ms"] == pytest.approx(remember["total_ms"] - remember["graph_ms"])
    assert remember["memories"] == 1
    assert remember["chars"] == len("The Springfield office opens at 9.")
    assert remember["error"] is None
    assert [get["op"], search["op"]] == ["get_memory", "search_memory"]


def test_failed_operation_records_the_error_and_still_raises(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory", actor="test/1", human_id="owner")
    service = MemoryService(store, FakeGraph())

    with pytest.raises(KeyError):
        service.get("mem_missing")

    (row,) = _rows(service)
    assert row["op"] == "get_memory" and "mem_missing" in row["error"]


def test_graph_failure_is_recorded_on_an_operation_that_falls_back(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory", actor="test/1", human_id="owner")
    graph = FakeGraph()
    graph.fail_search = True
    service = MemoryService(store, graph)

    run(service.search("anything"))

    (row,) = _rows(service)
    assert row["error"] == "graph unavailable"


def test_a_broken_metrics_database_does_not_break_the_operation(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory", actor="test/1", human_id="owner")
    service = MemoryService(store, FakeGraph())
    service.metrics.db_path.unlink()
    service.metrics.db_path.mkdir()  # sqlite cannot open a directory

    created = run(service.remember(title="Fact", content="Still saved.", kind="note"))

    assert store.record(created["id"]).content == "Still saved.\n"


def test_report_summarises_operations_and_their_trend_by_vault_size(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory", actor="test/1", human_id="owner")
    service = MemoryService(store, FakeGraph())
    for i in range(3):
        run(service.remember(title=f"Fact {i}", content=f"Fact number {i}.", kind="note"))
    run(service.search("fact"))

    report = service.metrics.report(days=30)

    assert "remember" in report and "search_memory" in report
    assert "0-49" in report  # the vault-size bucket these calls ran at


def test_report_without_timings_says_so(tmp_path: Path) -> None:
    assert "No timings" in Metrics(tmp_path).report(days=30)


def test_background_ingest_is_timed_and_a_failure_does_not_stop_the_queue(tmp_path: Path) -> None:
    class FlakyGraph(FakeGraph):
        async def ingest(self, record: MemoryRecord) -> dict:
            if "fails" in record.content:
                raise RuntimeError("OpenAI timed out")
            return await super().ingest(record)

    store = MemoryStore(tmp_path / "memory", actor="test/1", human_id="owner")
    graph = FlakyGraph()
    service = MemoryService(store, graph)

    async def scenario() -> tuple[dict, dict]:
        await service.start(ingest_in_background=True)
        failing = await service.remember(title="One", content="This one fails.", kind="note")
        working = await service.remember(title="Two", content="This one works.", kind="note")
        await asyncio.wait_for(service.graph_idle(), 1)
        await service.close()
        return failing, working

    failing, working = run(scenario())

    assert graph.ingested == [working["id"]]
    ingests = [row for row in _rows(service) if row["op"] == "graph_ingest"]
    assert [row["error"] for row in ingests] == ["RuntimeError: OpenAI timed out", None]
    assert ingests[1]["chars"] == len("This one works.\n")
    # The writes themselves no longer wait on the graph.
    assert all(row["graph_ms"] == 0 for row in _rows(service) if row["op"] == "remember")


def test_catching_up_on_unchanged_memories_records_no_timings(tmp_path: Path) -> None:
    class CurrentGraph(FakeGraph):
        async def ingest(self, record: MemoryRecord) -> dict:
            return {"indexed": True, "skipped": True, "memory_id": record.id}

    store = MemoryStore(tmp_path / "memory", actor="test/1", human_id="owner")
    store.remember(title="Opening hours", content="The Springfield office opens at 9.", kind="note")
    service = MemoryService(store, CurrentGraph())

    async def scenario() -> None:
        await service.start(ingest_in_background=True)
        await asyncio.wait_for(service.graph_idle(), 1)
        await service.close()

    run(scenario())
    assert _rows(service) == []
