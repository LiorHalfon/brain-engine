from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

psutil = pytest.importorskip("psutil")
pytest.importorskip("redislite")

from brain_engine.leftovers import find_servers  # noqa: E402
from conftest import make_vault  # noqa: E402

# Ctrl-C, `kill <pid>` and closing the terminal that runs the service.
SIGNALS = [signal.SIGINT, signal.SIGTERM, signal.SIGHUP]


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_until(condition, timeout: float, what: str) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise TimeoutError(what)
        time.sleep(0.02)


def _port_open(port: int) -> bool:
    try:
        socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
        return True
    except OSError:
        return False


@pytest.fixture
def service(tmp_path: Path) -> Iterator:
    """Start `brain-engine serve` on a fresh vault in a process of its own."""
    started: list[subprocess.Popen] = []

    def start() -> tuple[subprocess.Popen, Path, int, Path]:
        vault = make_vault(tmp_path / "vault")
        port = _free_port()
        log = tmp_path / "serve.log"
        env = {**os.environ, "OPENAI_API_KEY": "unused", "MEMORY_PORT": str(port)}  # opening the graph calls no API
        process = subprocess.Popen(
            [sys.executable, "-m", "brain_engine.cli", "serve", "--vault", str(vault)],
            env=env,
            stdout=log.open("w"),
            stderr=subprocess.STDOUT,
        )
        started.append(process)
        return process, vault / ".brain" / "graphiti.db", port, log

    yield start
    for process in started:
        if process.poll() is None:
            process.kill()
    for server in find_servers(tmp_path / "vault" / ".brain" / "graphiti.db"):
        psutil.Process(server.pid).kill()


def _watchers(owner: int) -> list[int]:
    return [
        p.pid
        for p in psutil.process_iter(["cmdline"])
        if "brain_engine.watchdog" in (p.info["cmdline"] or []) and str(owner) in (p.info["cmdline"] or [])
    ]


@pytest.mark.parametrize("sig", SIGNALS, ids=lambda sig: sig.name)
def test_signal_closes_the_service_and_its_graph(service, sig: signal.Signals) -> None:
    process, graph_db, port, log = service()
    _wait_until(lambda: _port_open(port), 30, "service did not start")
    (redis,) = find_servers(graph_db)
    (watcher,) = _watchers(process.pid)
    before = time.time()
    time.sleep(0.05)  # so the save on close gets a later mtime than `before`

    process.send_signal(sig)

    assert process.wait(10) == 0, log.read_text()
    assert "test-vault stopped" in log.read_text()
    _wait_until(lambda: not psutil.pid_exists(redis.pid), 5, "redis-server outlived the service")
    assert find_servers(graph_db) == []
    assert graph_db.stat().st_mtime > before, "closing did not save graphiti.db"
    # The service stopped its server itself, so the watcher has nothing left to do.
    _wait_until(lambda: not psutil.pid_exists(watcher), 5, "watcher outlived the server")


def test_sigterm_during_startup_still_closes_the_graph(service) -> None:
    process, graph_db, port, log = service()
    _wait_until(lambda: graph_db.with_name("graphiti.db.settings").exists() and find_servers(graph_db), 30, "graph never opened")

    process.send_signal(signal.SIGTERM)

    assert process.wait(10) == 0, log.read_text()
    assert find_servers(graph_db) == []


def test_a_killed_service_tree_still_saves_and_stops_its_graph(service) -> None:
    # Closing a tab in some terminals (the Claude desktop app's panel, for one) SIGKILLs the
    # service and every process it started, so no handler in the service can run.
    process, graph_db, port, log = service()
    _wait_until(lambda: _port_open(port), 30, "service did not start")
    (redis,) = find_servers(graph_db)
    before = time.time()
    time.sleep(0.05)

    tree = [process.pid] + [child.pid for child in psutil.Process(process.pid).children(recursive=True)]
    assert _watchers(process.pid) and not set(_watchers(process.pid)) & set(tree), "the watcher must not die with the tree"
    for pid in tree:
        os.kill(pid, signal.SIGKILL)
    process.wait(5)

    _wait_until(lambda: not psutil.pid_exists(redis.pid), 10, "redis-server outlived the killed service")
    assert find_servers(graph_db) == []
    assert graph_db.stat().st_mtime > before, "graphiti.db was not saved"


def _start_stdio(vault: Path, log: Path) -> subprocess.Popen:
    env = {**os.environ, "OPENAI_API_KEY": "unused"}
    return subprocess.Popen(
        [sys.executable, "-m", "brain_engine.cli", "stdio", "--vault", str(vault)],
        env=env,
        stdin=subprocess.PIPE,  # held open, like a connected client
        stdout=subprocess.PIPE,
        stderr=log.open("w"),
    )


def _initialize(process: subprocess.Popen) -> None:
    """Complete MCP's initialize handshake, so the server is up and reading stdin."""
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "test", "version": "0"}},
    }
    process.stdin.write((json.dumps(request) + "\n").encode())
    process.stdin.flush()
    response = json.loads(process.stdout.readline())
    assert response["id"] == 1 and "result" in response, response


@pytest.mark.parametrize("close_stdin_first", [False, True], ids=["stdin-open", "stdin-closed"])
def test_sigterm_stops_a_stdio_server_and_its_graph(tmp_path: Path, close_stdin_first: bool) -> None:
    # MCP's stdin reader blocks in a thread that cancellation can't reach. Clients usually close
    # stdin and then send SIGTERM, which used to land in the middle of closing the graph.
    vault = make_vault(tmp_path / "vault")
    graph_db = vault / ".brain" / "graphiti.db"
    log = tmp_path / "stdio.log"
    process = _start_stdio(vault, log)
    try:
        _initialize(process)
        (redis,) = find_servers(graph_db)
        before = time.time()
        time.sleep(0.05)

        if close_stdin_first:
            process.stdin.close()
        process.send_signal(signal.SIGTERM)

        assert process.wait(10) == 0, log.read_text()
        _wait_until(lambda: not psutil.pid_exists(redis.pid), 5, "redis-server outlived the service")
        assert find_servers(graph_db) == []
        assert graph_db.stat().st_mtime > before, "closing did not save graphiti.db"
    finally:
        if process.poll() is None:
            process.kill()
        for server in find_servers(graph_db):
            psutil.Process(server.pid).kill()


def test_sighup_is_left_alone_under_nohup(service, monkeypatch) -> None:
    # `nohup brain-engine serve &` must keep running when its terminal closes.
    real_popen = subprocess.Popen

    def ignoring_sighup(*args, **kwargs):
        return real_popen(*args, preexec_fn=lambda: signal.signal(signal.SIGHUP, signal.SIG_IGN), **kwargs)

    monkeypatch.setattr(subprocess, "Popen", ignoring_sighup)
    process, graph_db, port, log = service()
    monkeypatch.setattr(subprocess, "Popen", real_popen)
    _wait_until(lambda: _port_open(port), 30, "service did not start")

    process.send_signal(signal.SIGHUP)
    time.sleep(1)
    assert process.poll() is None, "SIGHUP stopped a service started with SIGHUP ignored"

    process.send_signal(signal.SIGTERM)
    assert process.wait(10) == 0
    assert find_servers(graph_db) == []
