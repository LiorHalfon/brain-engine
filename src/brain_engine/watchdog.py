"""Save and stop the graph's redis-server once the process that opened it is gone.

The service closes the graph itself on Ctrl-C, SIGTERM and SIGHUP. Some terminals end it with
SIGKILL instead: closing a tab in the Claude desktop app's terminal panel kills the service and
every process it started, and no handler can run. FalkorDB Lite's redis-server is daemonized, so
it survives and keeps the newest graph only in memory. This watcher is detached the same way. It
checks on its owner twice a second and, once the owner is gone, sends the server SHUTDOWN SAVE.
When the owner stops the server itself, the server's socket disappears and the watcher just exits.

Run as `python -m brain_engine.watchdog <owner pid> <owner start time> <redis socket>`.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

POLL_SECONDS = 0.5
PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def start_watchdog(socket_file: str) -> None:
    """Start a detached watcher for the redis-server at `socket_file`, owned by this process."""
    import psutil

    created = psutil.Process().create_time()
    # The shell backgrounds the watcher and exits, so the watcher is not this process's descendant
    # and survives a kill of the whole process tree. It gets a session of its own, without a terminal.
    launcher = subprocess.Popen(
        [
            "/bin/sh",
            "-c",
            'exec "$0" -m brain_engine.watchdog "$1" "$2" "$3" </dev/null >/dev/null 2>&1 &',
            sys.executable,
            str(os.getpid()),
            repr(created),
            socket_file,
        ],
        start_new_session=True,
        # Import the same brain_engine as this process, installed or not.
        env={**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(PACKAGE_ROOT), os.environ.get("PYTHONPATH")]))},
    )
    launcher.wait()


def _owner_alive(pid: int, created: float) -> bool:
    import psutil

    try:
        process = psutil.Process(pid)
        # Same pid but a later start time: the owner died and the pid was reused.
        return process.create_time() == created and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def watch(pid: int, created: float, socket_file: str) -> None:
    socket = Path(socket_file)
    while socket.exists():
        if not _owner_alive(pid, created):
            from brain_engine.leftovers import connect

            client = connect(socket_file)
            try:
                client.shutdown(save=True)
            except Exception:
                pass  # already stopping, or stopped between the check and the call
            finally:
                client.close()
            return
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    watch(int(sys.argv[1]), float(sys.argv[2]), sys.argv[3])
