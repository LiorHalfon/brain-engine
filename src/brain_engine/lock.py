from __future__ import annotations

import fcntl
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

LOCK_FILENAME = "service.lock"


class GraphOwnedElsewhere(RuntimeError):
    """Another process already owns the Graphiti/FalkorDB Lite state."""

    def __init__(self, owner: str, name: str = "brain-engine") -> None:
        self.owner = owner
        super().__init__(f"{name} state is already in use by: {owner or 'unknown process'}")


@contextmanager
def hold_state_lock(state_dir: Path, owner: str, name: str = "brain-engine") -> Iterator[None]:
    """Hold an exclusive lock on the derived state so only one process opens FalkorDB Lite.

    `owner` is written into the lock file so other processes can find the running service
    (an MCP URL) or report who holds it. The OS releases the lock if the process dies.
    """
    path = state_dir / LOCK_FILENAME
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        raise GraphOwnedElsewhere(path.read_text(encoding="utf-8").strip(), name) from None
    try:
        os.ftruncate(fd, 0)
        os.write(fd, owner.encode())
        yield
    finally:
        os.ftruncate(fd, 0)
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
