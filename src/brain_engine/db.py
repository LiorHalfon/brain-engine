from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def connection(path: Path) -> Iterator[sqlite3.Connection]:
    """One transaction on a SQLite file: commits, or rolls back on an exception, and always closes.

    `with sqlite3.connect(path)` does only the first part. The connection's statement cache refers
    back to it, so it keeps its file descriptor until the cyclic garbage collector runs, and a sync
    opens one per memory.
    """
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        with conn:
            yield conn
    finally:
        conn.close()
