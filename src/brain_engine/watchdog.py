"""Save and stop the graph's redis-server once the process that opened it is gone.

The service closes the graph itself on Ctrl-C, SIGTERM and SIGHUP. Some terminals end it with
SIGKILL instead: closing a tab in the Claude desktop app's terminal panel kills the service and
every process it started, and no handler can run. FalkorDB Lite's redis-server is daemonized, so
it survives and keeps the newest graph only in memory. This watcher is detached the same way, and
once its owner is gone it sends the server SHUTDOWN SAVE.

The owner holds the write end of a pipe whose read end the watcher waits on. However the owner
exits, the kernel closes its end and the watcher reads end-of-file. An owner that stopped or dropped
the server itself writes one byte first, and the watcher exits without touching the server, as it
does once the server's socket is gone. Process ids and start times play no part. The watcher used to
compare them, but psutil 7.2 on macOS shifts a process's create_time() whenever the system boot time
crosses a whole second, and macOS's clock corrections do that a few times a day. The watcher then
took a running service for a dead one and stopped its server.

When it stops a server, or finds it can't, the watcher appends a line saying so to `watchdog.log`
in the state folder, and writes it to the owner's stderr when that is a file, as a launchd agent's
log is.

Run as `python -m brain_engine.watchdog <pipe fd> <owner pid> <redis socket> <log file>`.
"""

from __future__ import annotations

import os
import select
import stat
import subprocess
import sys
from datetime import datetime
from pathlib import Path

POLL_SECONDS = 0.5
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
LOG_NAME = "watchdog.log"
RELEASE = b"r"


class Watchdog:
    """The owner's end of a watcher's pipe. Closing it, or exiting, tells the watcher the owner is gone."""

    def __init__(self, fd: int) -> None:
        self._fd: int | None = fd

    def release(self) -> None:
        """Let the watcher exit without touching the server, which this process stopped or dropped."""
        if self._fd is None:
            return
        try:
            os.write(self._fd, RELEASE)
        except OSError:
            pass  # the watcher already exited, because the server's socket is gone
        finally:
            os.close(self._fd)
            self._fd = None


def _stderr_if_file() -> int:
    # A launchd agent's stderr is its log file, where its owner looks. A pipe or a terminal stays
    # out: the watcher outlives its owner for a moment, and a reader such as an MCP client would
    # wait for it to see end-of-file.
    try:
        return 2 if stat.S_ISREG(os.fstat(2).st_mode) else subprocess.DEVNULL
    except OSError:
        return subprocess.DEVNULL


def start_watchdog(socket_file: str, log_file: Path) -> Watchdog:
    """Start a detached watcher for the redis-server at `socket_file`, owned by this process."""
    # Neither end is inheritable, so no other child keeps the write end open after this process exits.
    read_end, write_end = os.pipe()
    try:
        # The shell backgrounds the watcher and exits, so the watcher is not this process's descendant
        # and survives a kill of the whole process tree. It gets a session of its own, without a terminal.
        launcher = subprocess.Popen(
            [
                "/bin/sh",
                "-c",
                'exec "$0" -m brain_engine.watchdog "$1" "$2" "$3" "$4" </dev/null >/dev/null &',
                sys.executable,
                str(read_end),
                str(os.getpid()),
                socket_file,
                str(log_file),
            ],
            pass_fds=(read_end,),
            stderr=_stderr_if_file(),
            start_new_session=True,
            # Import the same brain_engine as this process, installed or not.
            env={**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(PACKAGE_ROOT), os.environ.get("PYTHONPATH")]))},
        )
        launcher.wait()
    except BaseException:
        os.close(write_end)
        raise
    finally:
        os.close(read_end)
    return Watchdog(write_end)


def _note(log_file: str, message: str) -> None:
    line = f"{datetime.now().astimezone().isoformat(timespec='seconds')} brain-engine watchdog: {message}\n"
    try:
        with open(log_file, "a", encoding="utf-8") as log:
            log.write(line)
    except OSError:
        pass
    try:
        sys.stderr.write(line)
        sys.stderr.flush()
    except (OSError, ValueError):
        pass


def _stop(owner: int, socket_file: str, log_file: str) -> None:
    from brain_engine.leftovers import connect

    gone = f"the process that opened the graph (pid {owner}) is gone"
    client = connect(socket_file)
    try:
        # SHUTDOWN reports nothing when it can't connect, so ask first whether anyone is there.
        client.ping()
        client.shutdown(save=True)
    except Exception as exc:
        _note(log_file, f"{gone}, and its redis-server at {socket_file} could not be saved and stopped: {exc}")
        return
    finally:
        client.close()
    _note(log_file, f"{gone}, so it saved and stopped the graph's redis-server at {socket_file}")


def watch(lifeline: int, owner: int, socket_file: str, log_file: str) -> None:
    socket = Path(socket_file)
    while socket.exists():
        ready, _, _ = select.select([lifeline], [], [], POLL_SECONDS)
        if not ready:
            continue
        if not os.read(lifeline, 1):  # end-of-file: the owner exited without releasing the server
            _stop(owner, socket_file, log_file)
        return


if __name__ == "__main__":
    watch(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4])
