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

from brain_engine.db import connection

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
    # For a search: whether its graph facts came from the keyword index alone.
    keyword_only: bool | None = None

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
    held, so a slowdown can be tied to vault growth. A search also notes whether its facts came
    from the keyword index alone, and every row the service's resident memory (`rss_mb`), so a
    slowdown can be tied to the process growing. The file is per machine and disposable.
    """

    def __init__(self, state_dir: Path) -> None:
        self.db_path = state_dir / "metrics.sqlite3"
        self._process = _this_process()
        try:
            with connection(self.db_path) as conn:
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
                        error TEXT,
                        keyword_only INTEGER,
                        rss_mb REAL
                    )
                    """
                )
                # Files written before 0.1.5 lack the newer columns, which stay empty on their rows.
                columns = {row["name"] for row in conn.execute("PRAGMA table_info(operations)")}
                for column, kind in (("keyword_only", "INTEGER"), ("rss_mb", "REAL")):
                    if column not in columns:
                        conn.execute(f"ALTER TABLE operations ADD COLUMN {column} {kind}")
        except sqlite3.Error as exc:
            logger.warning("Operation timings will not be recorded: %s", exc)

    def _rss_mb(self) -> float | None:
        try:
            return self._process.memory_info().rss / 2**20 if self._process is not None else None
        except Exception:
            return None

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
            with connection(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO operations(at,op,total_ms,store_ms,graph_ms,memories,chars,error,keyword_only,rss_mb) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (
                        datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        op,
                        total * 1000,
                        (total - timing.graph_seconds) * 1000,
                        timing.graph_seconds * 1000,
                        memories(),
                        chars,
                        timing.error,
                        None if timing.keyword_only is None else int(timing.keyword_only),
                        self._rss_mb(),
                    ),
                )
        except sqlite3.Error as exc:
            logger.warning("Could not record the timing of %s: %s", op, exc)

    def report(self, days: int = 30) -> str:
        since = datetime.now(timezone.utc) - timedelta(days=days)
        try:
            with connection(self.db_path) as conn:
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

        searches = [r for r in by_op.get("search_memory", []) if r["keyword_only"] is not None]
        if searches:
            keyword_only = sum(r["keyword_only"] for r in searches)
            lines += [
                "",
                f"search_memory: {keyword_only} of {len(searches)} searches ({keyword_only / len(searches):.0%}) "
                "took their graph facts from the keyword index alone, because the hybrid search missed its budget or failed.",
            ]
        rss = [(r["rss_mb"], r["at"]) for r in rows if r["rss_mb"] is not None]
        if rss:
            peak, peak_at = max(rss)
            lines.append(
                f"Service memory (RSS): median {statistics.median(mb for mb, _ in rss):.0f} MB, "
                f"max {peak:.0f} MB on {peak_at[:10]}."
            )
        lines += _recent_errors(rows)

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


def _this_process():
    # psutil comes with the graph extra. Without it, rows carry no memory figure.
    try:
        import psutil

        return psutil.Process()
    except Exception:
        return None


def _recent_errors(rows: list[sqlite3.Row], days: int = 7, shown: int = 5) -> list[str]:
    # A graph that fails every call shows up here, rather than only in each tool result.
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(timespec="seconds")
    groups: dict[tuple[str, str], list[str]] = {}
    for row in rows:
        if row["error"] and row["at"] >= since:
            message = " ".join(row["error"].split())
            groups.setdefault((row["op"], message[:120] + ("…" if len(message) > 120 else "")), []).append(row["at"])
    if not groups:
        return []
    ranked = sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))
    table = _table(
        ["operation", "count", "last", "error"],
        [[op, str(len(ats)), max(ats)[:16].replace("T", " "), message] for (op, message), ats in ranked[:shown]],
        text_last=True,
    )
    more = len(ranked) - shown
    return ["", f"Errors in the last {days} days, most frequent first (UTC):", "", *table] + (
        [f"... and {more} other kinds of error"] if more > 0 else []
    )


def _pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def _ms(value: float) -> str:
    return f"{value:.0f}ms" if value < 1000 else f"{value / 1000:.1f}s"


def _bucket_width(max_memories: int) -> int:
    # At most about eight rows per operation in the trend table.
    return next((w for w in BUCKET_WIDTHS if max_memories // w < 8), BUCKET_WIDTHS[-1])


def _table(header: list[str], rows: list[list[str]], *, text_last: bool = False) -> list[str]:
    # Numbers align right. The first column, and the last one when it holds text, align left.
    widths = [max(len(cell) for cell in column) for column in zip(header, *rows)]
    last = len(header) - 1
    return [
        "  ".join(
            cell if text_last and i == last else cell.ljust(w) if i == 0 else cell.rjust(w)
            for i, (cell, w) in enumerate(zip(row, widths))
        )
        for row in [header, *rows]
    ]
