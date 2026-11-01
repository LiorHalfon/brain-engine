from __future__ import annotations

import shutil
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

psutil = pytest.importorskip("psutil")
pytest.importorskip("redislite")

from brain_engine.graph import GraphitiBackend  # noqa: E402
from brain_engine.leftovers import LeftoverServersError, find_servers, stop_leftover_servers  # noqa: E402

# Starts FalkorDB Lite's redis-server for a database file the way a service does, optionally writes
# to it, then exits without stopping it: what a service killed by a signal used to leave behind.
LEAVE_A_SERVER = """
import os, sys, redislite
db, mode = sys.argv[1], sys.argv[2]
registry = db + ".settings"
if os.path.exists(registry):
    os.remove(registry)  # start a new server instead of reattaching to the last one
client = redislite.Redis(db)
if mode in ("write", "save"):
    client.set("probe", "kept")
if mode == "save":
    client.save()
print(client.pid, flush=True)
os._exit(0)
"""


@pytest.fixture
def leave_server() -> Iterator:
    pids: list[int] = []

    def leave(db: Path, mode: str) -> int:
        out = subprocess.run([sys.executable, "-c", LEAVE_A_SERVER, str(db), mode], capture_output=True, text=True, check=True)
        pid = int(out.stdout.strip())
        pids.append(pid)
        return pid

    yield leave
    for pid in pids:
        if psutil.pid_exists(pid):
            psutil.Process(pid).kill()


def _gone(pid: int) -> bool:
    try:
        psutil.Process(pid).wait(5)
    except psutil.NoSuchProcess:
        pass
    return not psutil.pid_exists(pid)


def _probe(db: Path) -> str | None:
    import redislite

    (db.parent / f"{db.name}.settings").unlink(missing_ok=True)
    client = redislite.Redis(str(db))
    try:
        value = client.get("probe")
        return value.decode() if value else None
    finally:
        client.shutdown(nosave=True)


def test_find_servers_sees_only_its_own_database(tmp_path: Path, leave_server) -> None:
    mine, other = tmp_path / "a" / "graphiti.db", tmp_path / "b" / "graphiti.db"
    mine.parent.mkdir()
    other.parent.mkdir()
    pid = leave_server(mine, "write")
    leave_server(other, "write")

    (server,) = find_servers(mine)

    assert server.pid == pid and server.unsaved_changes == 1


def test_a_saved_leftover_is_stopped_without_saving(tmp_path: Path, leave_server) -> None:
    db = tmp_path / "graphiti.db"
    pid = leave_server(db, "save")

    stopped = stop_leftover_servers(db)

    assert [server.pid for server in stopped] == [pid]
    assert _gone(pid) and find_servers(db) == []
    assert _probe(db) == "kept"


def test_the_one_unsaved_leftover_is_saved_then_stopped(tmp_path: Path, leave_server) -> None:
    db = tmp_path / "graphiti.db"
    pid = leave_server(db, "write")

    stop_leftover_servers(db)

    assert _gone(pid) and find_servers(db) == []
    assert _probe(db) == "kept"


def test_two_unsaved_leftovers_refuse_to_start(tmp_path: Path, leave_server) -> None:
    db = tmp_path / "graphiti.db"
    first, second = leave_server(db, "write"), leave_server(db, "write")

    with pytest.raises(LeftoverServersError) as exc:
        stop_leftover_servers(db)

    assert str(first) in str(exc.value) and str(second) in str(exc.value)
    assert "INFO persistence" in str(exc.value)
    assert psutil.pid_exists(first) and psutil.pid_exists(second)


async def test_opening_the_graph_stops_leftovers_first(tmp_path: Path, leave_server, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "unused")  # opening the graph makes no OpenAI calls
    backend = GraphitiBackend(tmp_path)
    pid = leave_server(backend.graph_db_path, "write")

    await backend.start()
    try:
        assert _gone(pid)
        assert [server.pid for server in find_servers(backend.graph_db_path)] == [backend._falkor.connection.pid]
    finally:
        await backend.close()


async def test_a_copied_registry_never_attaches_to_another_vaults_server(
    tmp_path: Path, leave_server, monkeypatch
) -> None:
    # A copied state dir carries redislite's graphiti.db.settings, which names the other vault's
    # running server. Attaching to it would write into that vault's graph, and closing would stop it.
    monkeypatch.setenv("OPENAI_API_KEY", "unused")
    original, copy = tmp_path / "original", tmp_path / "copy"
    original.mkdir()
    other_pid = leave_server(original / "graphiti.db", "write")
    shutil.copytree(original, copy)

    backend = GraphitiBackend(copy)
    await backend.start()
    try:
        own_pid = backend._falkor.connection.pid
        assert own_pid != other_pid
        assert [server.pid for server in find_servers(copy / "graphiti.db")] == [own_pid]
    finally:
        await backend.close()
    assert psutil.pid_exists(other_pid)
    assert find_servers(original / "graphiti.db")[0].unsaved_changes == 1


def test_a_reused_pid_is_not_the_owner() -> None:
    from brain_engine.watchdog import _owner_alive

    me = psutil.Process()
    assert _owner_alive(me.pid, me.create_time())
    assert not _owner_alive(me.pid, me.create_time() - 60)  # same pid, started earlier: someone else's


def test_a_path_in_another_case_still_finds_the_server(tmp_path: Path, leave_server) -> None:
    folder = tmp_path / "CaseVault"
    folder.mkdir()
    if not (tmp_path / "casevault").exists():
        pytest.skip("case-sensitive file system")
    pid = leave_server(folder / "graphiti.db", "write")

    assert [server.pid for server in find_servers(tmp_path / "casevault" / "graphiti.db")] == [pid]


def test_a_server_that_does_not_answer_refuses_start(tmp_path: Path, leave_server, monkeypatch) -> None:
    # It may be busy saving; starting a second server beside it could let it overwrite the graph later.
    import signal

    from brain_engine import leftovers

    monkeypatch.setattr(leftovers, "QUERY_TIMEOUT_SECONDS", 0.5)
    db = tmp_path / "graphiti.db"
    pid = leave_server(db, "write")
    psutil.Process(pid).send_signal(signal.SIGSTOP)
    try:
        with pytest.raises(LeftoverServersError, match=f"pid {pid}.*doesn't answer"):
            stop_leftover_servers(db)
    finally:
        psutil.Process(pid).send_signal(signal.SIGCONT)
