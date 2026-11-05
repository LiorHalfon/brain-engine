"""Every operation closes its SQLite connections itself.

`with sqlite3.connect(path)` only commits. The connection's statement cache refers back to it, so it
keeps its file descriptor until the cyclic garbage collector runs. A sync opens one per memory, which
adds up to hundreds of descriptors between collections, enough to reach launchd's default limit of 256.
These tests turn the collector off, so a connection nobody closed still shows as an open descriptor.
"""

from __future__ import annotations

import gc
import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from brain_engine.service import MemoryService
from test_graph import make_backend
from test_service import run


def open_descriptors() -> int:
    return len(os.listdir("/dev/fd"))


@pytest.fixture
def no_collector() -> Iterator[None]:
    gc.collect()
    gc.disable()
    yield
    gc.enable()


def test_service_operations_leave_no_connection_open(tmp_path: Path, no_collector: None) -> None:
    before = open_descriptors()

    store, backend, _ = make_backend(tmp_path)
    service = MemoryService(store, backend)
    created = run(service.remember(title="Carrier", content="Acme ships orders with Parcelco.", kind="note"))
    memory_id = created["id"]
    run(service.sync_graph())
    run(service.search("Parcelco"))
    service.get(memory_id)
    run(service.edit(memory_id, "Acme ships orders with Boxco."))
    service.verify(memory_id)
    corrected = run(service.correct(memory_id, "Acme ships orders with Crateco.", "Carrier changed"))
    run(service.remember_session(title="Carrier session", summary="Looked at the carrier."))
    run(service.sync_graph())
    run(service.graph.remove(corrected["id"]))
    service.metrics.report()

    assert open_descriptors() == before


def test_a_sync_leaves_no_connection_open(tmp_path: Path, no_collector: None) -> None:
    store, backend, _ = make_backend(tmp_path)
    service = MemoryService(store, backend)
    for number in range(20):
        run(service.remember(title=f"Note {number}", content=f"Acme ships order {number}.", kind="note"))
    run(service.sync_graph())
    before = open_descriptors()

    run(service.sync_graph())

    assert open_descriptors() == before


def test_a_failed_write_rolls_back_and_closes(tmp_path: Path, no_collector: None) -> None:
    store, _, _ = make_backend(tmp_path)
    before = open_descriptors()

    with pytest.raises(RuntimeError), store._connect() as conn:
        conn.execute("INSERT INTO meta(key, value) VALUES ('half-done', 'x')")
        raise RuntimeError("the write failed")

    assert open_descriptors() == before
    with store._connect() as conn:
        assert conn.execute("SELECT value FROM meta WHERE key='half-done'").fetchone() is None
