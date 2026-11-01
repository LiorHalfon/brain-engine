from __future__ import annotations

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


@pytest.mark.parametrize("sig", SIGNALS, ids=lambda sig: sig.name)
def test_signal_closes_the_service_and_its_graph(service, sig: signal.Signals) -> None:
    process, graph_db, port, log = service()
    _wait_until(lambda: _port_open(port), 30, "service did not start")
    (redis,) = find_servers(graph_db)
    before = time.time()
    time.sleep(0.05)  # so the save on close gets a later mtime than `before`

    process.send_signal(sig)

    assert process.wait(10) == 0, log.read_text()
    assert "test-vault stopped" in log.read_text()
    assert not psutil.pid_exists(redis.pid), "redis-server outlived the service"
    assert find_servers(graph_db) == []
    assert graph_db.stat().st_mtime > before, "closing did not save graphiti.db"


def test_sigterm_during_startup_still_closes_the_graph(service) -> None:
    process, graph_db, port, log = service()
    _wait_until(lambda: graph_db.with_name("graphiti.db.settings").exists() and find_servers(graph_db), 30, "graph never opened")

    process.send_signal(signal.SIGTERM)

    assert process.wait(10) == 0, log.read_text()
    assert find_servers(graph_db) == []
