from __future__ import annotations

import os
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

psutil = pytest.importorskip("psutil")

from brain_engine import leftovers, watchdog  # noqa: E402
from brain_engine.watchdog import RELEASE, watch  # noqa: E402


class FakeRedis:
    def __init__(self, calls: list[str], *, answers: bool) -> None:
        self.calls = calls
        self.answers = answers

    def ping(self) -> bool:
        if not self.answers:
            raise ConnectionError("Error 61 connecting to the socket. Connection refused.")
        self.calls.append("ping")
        return True

    def shutdown(self, *, save: bool) -> None:
        self.calls.append("shutdown save" if save else "shutdown")

    def close(self) -> None:
        pass


@pytest.fixture
def watcher(tmp_path: Path, monkeypatch) -> Iterator:
    """Run `watch` in a thread against a stand-in server, with this test as its owner."""
    monkeypatch.setattr(watchdog, "POLL_SECONDS", 0.01)
    calls: list[str] = []
    socket = tmp_path / "redis.socket"
    socket.touch()
    log = tmp_path / "watchdog.log"
    started: list[tuple[threading.Thread, int]] = []

    def start(*, answers: bool = True) -> tuple[threading.Thread, int, list[str], Path, Path]:
        monkeypatch.setattr(leftovers, "connect", lambda path, timeout=None: FakeRedis(calls, answers=answers))
        read_end, write_end = os.pipe()
        thread = threading.Thread(target=watch, args=(read_end, 4242, str(socket), str(log)), daemon=True)
        thread.start()
        started.append((thread, read_end))
        return thread, write_end, calls, socket, log

    yield start
    for thread, read_end in started:
        thread.join(2)
        os.close(read_end)


def test_a_live_owner_keeps_its_server_whatever_its_start_time_says(watcher, monkeypatch) -> None:
    # psutil 7.2 on macOS moves create_time() by a second whenever the boot time crosses one, which
    # macOS's clock corrections do a few times a day. The watcher used to take that for a new
    # process in its owner's pid, and stopped the running service's redis-server.
    monkeypatch.setattr(psutil.Process, "create_time", lambda self: 0.0)
    thread, owner_end, calls, _, log = watcher()

    time.sleep(0.2)
    assert thread.is_alive() and calls == []

    os.close(owner_end)  # the owner exits, however it exits
    thread.join(2)
    assert not thread.is_alive()
    assert calls == ["ping", "shutdown save"]
    assert "(pid 4242) is gone, so it saved and stopped the graph's redis-server" in log.read_text()


def test_a_released_watcher_leaves_the_server_alone(watcher) -> None:
    thread, owner_end, calls, _, log = watcher()

    os.write(owner_end, RELEASE)  # the owner stopped or dropped the server itself
    os.close(owner_end)
    thread.join(2)

    assert not thread.is_alive() and calls == []
    assert not log.exists()


def test_the_watcher_exits_once_the_socket_is_gone(watcher) -> None:
    thread, owner_end, calls, socket, _ = watcher()

    socket.unlink()
    thread.join(2)
    os.close(owner_end)

    assert not thread.is_alive() and calls == []


def test_a_watcher_that_cannot_reach_the_server_says_so(watcher) -> None:
    # SHUTDOWN reports nothing when it can't connect, so the log must not claim a save.
    thread, owner_end, calls, _, log = watcher(answers=False)

    os.close(owner_end)
    thread.join(2)

    assert calls == []
    assert "could not be saved and stopped: Error 61" in log.read_text()
