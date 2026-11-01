from __future__ import annotations

import logging
import math
import sqlite3
import statistics
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# Report order: the MCP tools first, then the service's background graph ingest and the graph viewer.
OPERATIONS = [
    "search_memory",
    "get_memory",
    "remember",
    "remember_session",
    "edit_memory",
    "correct_memory",
    "verify_memory",
    "sync_memory_graph",
    "graph_ingest",
    "graph_view",
]
BUCKET_WIDTHS = [50, 100, 250, 500, 1000, 2500, 5000, 10000]


@dataclass
class Timing:
    """Filled in while one operation runs. Time outside `graph()` counts as store time."""

    graph_seconds: float = 0.0
    error: str | None = None
    # Set when the call turned out to do nothing worth timing, so it is not recorded.
    discard: bool = False

    @contextmanager
    def graph(self) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        except Exception as exc:
            # The service catches graph failures and falls back, so note them here.
            self.error = str(exc)
            raise
        finally:
            self.graph_seconds += time.perf_counter() - start


class Metrics:
    """How long each service operation takes, kept in `metrics.sqlite3` in the state dir.

    One row per call: total time, the part spent in Graphiti and its OpenAI calls (`graph_ms`),
    the rest (`store_ms`: Markdown, FTS and our own filtering), and how many memories the vault
    held, so a slowdown can be tied to vault growth. The file is per machine and disposable.
    """

    def __init__(self, state_dir: Path) -> None:
        self.db_path = state_dir / "metrics.sqlite3"
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS operations (
                        at TEXT NOT NULL,
                        op TEXT NOT NULL,
                        total_ms REAL NOT NULL,
                        store_ms REAL NOT NULL,
                        graph_ms REAL NOT NULL,
                        memories INTEGER,
                        chars INTEGER,
                        error TEXT
                    )
                    """
                )
        except sqlite3.Error as exc:
            logger.warning("Operation timings will not be recorded: %s", exc)

    @contextmanager
    def measure(self, op: str, *, memories: Callable[[], int], chars: int | None = None) -> Iterator[Timing]:
        timing = Timing()
        start = time.perf_counter()
        try:
            yield timing
        except Exception as exc:
            timing.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            if not timing.discard:
                self._record(op, time.perf_counter() - start, timing, memories, chars)

    def _record(
        self, op: str, total: float, timing: Timing, memories: Callable[[], int], chars: int | None
    ) -> None:
        # Recording must never fail the operation it measures.
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO operations(at,op,total_ms,store_ms,graph_ms,memories,chars,error) "
                    "VALUES (?,?,?,?,?,?,?,?)",
                    (
                        datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        op,
                        total * 1000,
                        (total - timing.graph_seconds) * 1000,
                        timing.graph_seconds * 1000,
                        memories(),
                        chars,
                        timing.error,
                    ),
                )
        except sqlite3.Error as exc:
            logger.warning("Could not record the timing of %s: %s", op, exc)

    def report(self, days: int = 30) -> str:
        since = datetime.now(timezone.utc) - timedelta(days=days)
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT * FROM operations WHERE at >= ? ORDER BY at", (since.isoformat(timespec="seconds"),)
                ).fetchall()
        except sqlite3.Error:
            rows = []
        if not rows:
            return f"No timings recorded in the last {days} days in {self.db_path}."

        by_op: dict[str, list[sqlite3.Row]] = {}
        for row in rows:
            by_op.setdefault(row["op"], []).append(row)
        ops = [op for op in OPERATIONS if op in by_op] + sorted(set(by_op) - set(OPERATIONS))

        lines = [
            f"Memory operation timings on this machine, {rows[0]['at'][:10]} to {rows[-1]['at'][:10]} "
            f"({len(rows)} calls, last {days} days)",
            "store = our Markdown/FTS code, graph = Graphiti and its OpenAI calls",
            "",
        ]
        lines += _table(
            ["operation", "calls", "p50", "p90", "max", "store p50", "graph p50", "errors"],
            [
                [
                    op,
                    str(len(op_rows)),
                    _ms(_pct([r["total_ms"] for r in op_rows], 0.5)),
                    _ms(_pct([r["total_ms"] for r in op_rows], 0.9)),
                    _ms(max(r["total_ms"] for r in op_rows)),
                    _ms(_pct([r["store_ms"] for r in op_rows], 0.5)),
                    _ms(_pct([r["graph_ms"] for r in op_rows], 0.5)),
                    str(sum(1 for r in op_rows if r["error"])),
                ]
                for op, op_rows in ((op, by_op[op]) for op in ops)
            ],
        )

        width = _bucket_width(max(r["memories"] or 0 for r in rows))
        trend = []
        for op in ops:
            buckets: dict[int, list[sqlite3.Row]] = {}
            for row in by_op[op]:
                buckets.setdefault((row["memories"] or 0) // width, []).append(row)
            for bucket, bucket_rows in sorted(buckets.items()):
                # Graphiti time grows with the size of the memory, so per KB shows whether it also
                # grows with the vault.
                per_kb = [r["graph_ms"] / (r["chars"] / 1024) for r in bucket_rows if r["chars"] and r["graph_ms"]]
                trend.append(
                    [
                        op,
                        f"{bucket * width}-{(bucket + 1) * width - 1}",
                        str(len(bucket_rows)),
                        _ms(_pct([r["total_ms"] for r in bucket_rows], 0.5)),
                        _ms(_pct([r["store_ms"] for r in bucket_rows], 0.5)),
                        _ms(_pct([r["graph_ms"] for r in bucket_rows], 0.5)),
                        _ms(statistics.median(per_kb)) if per_kb else "-",
                    ]
                )
        lines += ["", "Median by vault size, to see whether operations slow down as memories are added:", ""]
        lines += _table(["operation", "memories", "calls", "p50", "store p50", "graph p50", "graph per KB"], trend)
        return "\n".join(lines)


def _pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def _ms(value: float) -> str:
    return f"{value:.0f}ms" if value < 1000 else f"{value / 1000:.1f}s"


def _bucket_width(max_memories: int) -> int:
    # At most about eight rows per operation in the trend table.
    return next((w for w in BUCKET_WIDTHS if max_memories // w < 8), BUCKET_WIDTHS[-1])


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    widths = [max(len(cell) for cell in column) for column in zip(header, *rows)]
    return [
        "  ".join(cell.ljust(w) if i == 0 else cell.rjust(w) for i, (cell, w) in enumerate(zip(row, widths)))
        for row in [header, *rows]
    ]
