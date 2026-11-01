"""Time the store side of each memory operation on copies of the vault grown to larger sizes.

Usage: bench_store.py <vault> [size ...]. Each size gets a temporary copy of the vault's Markdown
padded with renamed duplicates of its own memories, so the real vault and index are never touched.
Graphiti is replaced by a stub that answers instantly, so the numbers are our own Markdown/FTS
cost only; `brain-engine metrics` has the Graphiti and OpenAI time measured on real calls.
"""

from __future__ import annotations

import asyncio
import random
import re
import shutil
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

from brain_engine.config import Settings
from brain_engine.graph import GraphFact
from brain_engine.service import MemoryService
from brain_engine.store import RESERVED_FILENAMES, MemoryRecord, MemoryStore

OPERATIONS = ["search_memory", "get_memory", "remember", "edit_memory", "verify_memory"]


class StubGraph:
    """Answers a search with ten facts from one memory each, like Graphiti on the real vault."""

    enabled = True

    def __init__(self, memory_ids: list[str]) -> None:
        self.memory_ids = memory_ids

    async def start(self) -> None: ...

    async def close(self) -> None: ...

    async def ingest(self, record: MemoryRecord) -> dict:
        return {"indexed": True, "memory_id": record.id}

    async def search(self, query: str, limit: int = 5, include_invalidated: bool = False) -> list[GraphFact]:
        return [GraphFact(f"fact-{i}", f"Fact {i}.", None, None, [random.choice(self.memory_ids)]) for i in range(limit)]


def grow(source: Path, target: Path, size: int) -> None:
    shutil.copytree(source, target)
    files = [p for p in target.rglob("*.md") if p.name not in RESERVED_FILENAMES]
    originals = list(files)
    copy = 0
    while len(files) < size:
        copy += 1
        for path in originals[: size - len(files)]:
            text = re.sub(r"^id: (mem_\w+)$", rf"id: \g<1>c{copy}", path.read_text(encoding="utf-8"), count=1, flags=re.M)
            clone = path.with_name(f"{path.stem}-copy{copy}.md")
            clone.write_text(f"{text}\nSynthetic copy {copy}.\n", encoding="utf-8")
            files.append(clone)


def best_ms(fn: Callable[[], object], repeats: int = 3) -> float:
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best * 1000


def bench(root: Path, settings: Settings) -> dict[str, float]:
    store = MemoryStore(
        root,
        actor="bench",
        human_id="bench",
        state_dir=root.parent / "state",
        kinds=settings.kinds,
        dated_kinds=settings.dated_kinds,
    )
    memory_ids = [record.id for record in store.iter_records(searchable_only=False)]
    service = MemoryService(store, StubGraph(memory_ids))
    target = memory_ids[len(memory_ids) // 2]
    counter = iter(range(1_000_000))
    store.ensure_index()
    return {
        "memories": len(memory_ids),
        "search_memory": best_ms(lambda: asyncio.run(service.search("office hours delivery"))),
        "get_memory": best_ms(lambda: service.get(target)),
        "remember": best_ms(lambda: asyncio.run(service.remember(title="Bench", content=f"Bench {next(counter)}."))),
        "edit_memory": best_ms(lambda: asyncio.run(service.edit(target, f"Edited {next(counter)}."))),
        "verify_memory": best_ms(lambda: service.verify(target)),
    }


def main() -> int:
    settings = Settings.load(Path(sys.argv[1]).resolve())
    source = settings.memory_dir
    current = sum(1 for p in source.rglob("*.md") if p.name not in RESERVED_FILENAMES)
    sizes = [int(arg) for arg in sys.argv[2:]] or [current, 500, 1000, 2000]
    print("Store time per operation (best of 3), Graphiti stubbed out:\n")
    print(f"{'memories':>8}" + "".join(f"{op:>15}" for op in OPERATIONS))
    for size in sorted(set(max(size, current) for size in sizes)):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "memory"
            grow(source, root, size)
            result = bench(root, settings)
        print(f"{result['memories']:>8.0f}" + "".join(f"{result[op]:>13.0f}ms" for op in OPERATIONS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
