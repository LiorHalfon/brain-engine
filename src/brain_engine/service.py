from __future__ import annotations

import asyncio
import contextlib
import logging
from contextlib import AbstractContextManager
from typing import Any

from brain_engine.config import Settings
from brain_engine.graph import GraphBackend, GraphFact, GraphitiBackend
from brain_engine.metrics import Metrics, Timing
from brain_engine.store import MemoryStore, SearchResult

logger = logging.getLogger(__name__)


class MemoryService:
    def __init__(self, store: MemoryStore, graph: GraphBackend, metrics: Metrics | None = None) -> None:
        self.store = store
        self.graph = graph
        self.metrics = metrics or Metrics(store.state_dir)
        # Set up by start(ingest_in_background=True): memory ids waiting for the graph, in write order.
        self._ingest_queue: asyncio.Queue[str] | None = None
        self._queued: set[str] = set()
        self._ingest_worker: asyncio.Task[None] | None = None

    def _measure(self, op: str, chars: int | None = None) -> AbstractContextManager[Timing]:
        # Every public operation is timed; graph awaits run inside `timing.graph()`.
        return self.metrics.measure(op, memories=self.store.count, chars=chars)

    @classmethod
    def from_settings(cls, settings: Settings) -> "MemoryService":
        store = MemoryStore(
            settings.memory_dir,
            actor=settings.actor,
            human_id=settings.human_id,
            state_dir=settings.state_dir,
            kinds=settings.kinds,
            dated_kinds=settings.dated_kinds,
        )
        graph = GraphitiBackend(
            settings.state_dir,
            enabled=settings.graph_enabled,
            name=settings.name,
            model=settings.graph_model,
            small_model=settings.graph_small_model,
        )
        return cls(store, graph)

    async def start(self, *, ingest_in_background: bool = False) -> None:
        """Open the index and the graph. A long-running server passes `ingest_in_background`, so
        writes return once Markdown and FTS have them and Graphiti ingests them afterwards."""
        self.store.ensure_index()
        try:
            await self.graph.start()
        except Exception as exc:
            # Same fallback as per-request graph failures: Markdown + FTS keep working.
            logger.warning("Graphiti unavailable, serving Markdown/FTS only: %s", exc)
            return
        if not ingest_in_background or not self.graph.enabled:
            return
        self._ingest_queue = asyncio.Queue()
        self._ingest_worker = asyncio.create_task(self._ingest_queued())
        # Catch up on what the graph missed: memories pulled from git, or still queued at the last
        # stop. Ingest skips a memory whose text the graph already has, without calling OpenAI.
        for record in self.store.iter_records(searchable_only=True):
            if record.metadata.get("status") != "superseded":  # the same memories sync_graph ingests
                self._queue_ingest(record.id)

    async def close(self) -> None:
        if self._ingest_worker is not None:
            # Don't wait for an ingest in flight: the next start queues it again.
            self._ingest_worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._ingest_worker
            self._ingest_worker = self._ingest_queue = None
            self._queued.clear()
        await self.graph.close()

    async def graph_idle(self) -> None:
        """Wait until the graph has ingested every queued memory."""
        if self._ingest_queue is not None:
            await self._ingest_queue.join()

    def _queue_ingest(self, memory_id: str) -> None:
        assert self._ingest_queue is not None
        # A memory edited again before its turn is ingested once, with its newest text.
        if memory_id not in self._queued:
            self._queued.add(memory_id)
            self._ingest_queue.put_nowait(memory_id)

    async def _ingest_queued(self) -> None:
        assert self._ingest_queue is not None
        # One ingest at a time: Graphiti resolves each episode's entities against the graph so far.
        while True:
            memory_id = await self._ingest_queue.get()
            self._queued.discard(memory_id)
            try:
                await self._ingest_in_background(memory_id)
            except Exception as exc:
                # The worker must outlive any one memory. The next start queues this one again.
                logger.warning("Graphiti could not ingest %s: %s", memory_id, exc)
            finally:
                self._ingest_queue.task_done()

    async def _ingest_in_background(self, memory_id: str) -> None:
        try:
            record = self.store.record(memory_id)
        except KeyError:
            return  # removed from the vault since it was queued
        with self._measure("graph_ingest", len(record.content)) as timing:
            try:
                with timing.graph():
                    result = await self.graph.ingest(record)
            except asyncio.CancelledError:
                timing.discard = True  # cut off by a stop, not a timing worth keeping
                raise
            timing.discard = bool(result.get("skipped"))

    async def _index(self, memory_id: str, timing: Timing) -> dict[str, Any]:
        """Hand a written memory to the graph: queued while the service runs, otherwise right away."""
        if self._ingest_queue is not None:
            self._queue_ingest(memory_id)
            return {"queued": True, "memory_id": memory_id}
        try:
            record = self.store.record(memory_id)
            with timing.graph():
                return await self.graph.ingest(record)
        except Exception as exc:
            return {"indexed": False, "error": str(exc)}

    async def search(
        self,
        query: str,
        limit: int = 5,
        kind: str | None = None,
        include_history: bool = False,
    ) -> dict[str, Any]:
        with self._measure("search_memory") as timing:
            # Keep each retrieval source compact. Agents can fetch full Markdown with get_memory.
            document_results: list[SearchResult] = self.store.search(
                query=query,
                limit=limit,
                kind=kind,
                include_superseded=include_history,
            )
            graph_error = None
            try:
                with timing.graph():
                    raw_graph_results: list[GraphFact] = await self.graph.search(
                        query=query,
                        limit=limit * 2,
                        include_invalidated=include_history,
                    )
                graph_results = []
                seen_facts: set[str] = set()
                for fact in raw_graph_results:
                    normalized = " ".join(fact.fact.casefold().split())
                    if normalized in seen_facts:
                        continue
                    seen_facts.add(normalized)

                    if not include_history and self._all_sources_superseded(fact.source_memory_ids):
                        continue

                    graph_results.append(fact)
                    if len(graph_results) >= limit:
                        break
            except Exception as exc:
                graph_results = []
                graph_error = str(exc)
                timing.error = graph_error

            return {
                "facts": [fact.__dict__ for fact in graph_results],
                "memories": [result.__dict__ for result in document_results],
                "graph_error": graph_error,
            }

    def _all_sources_superseded(self, memory_ids: list[str]) -> bool:
        # store.search() has just brought the index up to date, so its statuses are current.
        return _all_superseded(memory_ids, self.store.statuses(memory_ids))

    async def graph_snapshot(self) -> dict[str, Any]:
        """Entities and facts for the /graph viewer, with `current` decided the same way as search."""
        with self._measure("graph_view") as timing:
            try:
                with timing.graph():
                    snapshot = await self.graph.snapshot()
            except Exception as exc:
                return {"enabled": True, "nodes": [], "edges": [], "memories": {}, "error": str(exc)}
            # Read the vault once: the viewer needs the title and status of every fact's sources.
            records = {record.id: record for record in self.store.iter_records(searchable_only=False)}
            statuses = {memory_id: str(record.metadata.get("status", "draft")) for memory_id, record in records.items()}
            memories: dict[str, dict[str, Any]] = {}
            for edge in snapshot["edges"]:
                for memory_id in edge["source_memory_ids"]:
                    record = records.get(memory_id)
                    if record is None or memory_id in memories:
                        continue
                    memories[memory_id] = {
                        "title": record.metadata.get("title"),
                        "type": record.metadata.get("type"),
                        "path": record.path,
                        "status": statuses[memory_id],
                        "verified_by_user": bool(record.metadata.get("verified_by_user", False)),
                    }
                edge["current"] = edge["invalid_at"] is None and not _all_superseded(edge["source_memory_ids"], statuses)
            return {**snapshot, "memories": memories, "error": None}

    def get(self, memory_id: str, offset: int = 0, max_chars: int = 8000) -> dict[str, Any]:
        with self._measure("get_memory"):
            return self.store.get(memory_id, offset, max_chars)

    async def remember(self, **kwargs: Any) -> dict[str, Any]:
        with self._measure("remember", len(kwargs["content"])) as timing:
            result = self.store.remember(**kwargs)
            if result.get("duplicate"):
                return {**result, "graph": {"skipped": True, "reason": "exact duplicate"}}
            return {**result, "graph": await self._index(result["id"], timing)}

    async def remember_session(self, **kwargs: Any) -> dict[str, Any]:
        with self._measure("remember_session", len(kwargs["summary"])) as timing:
            result = self.store.remember_session(**kwargs)
            summary = result["summary"]
            if not summary.get("duplicate"):
                result["graph"] = await self._index(summary["id"], timing)
            else:
                result["graph"] = {"skipped": True, "reason": "exact duplicate"}
            return result

    async def edit(self, memory_id: str, content: str, reason: str | None = None) -> dict[str, Any]:
        with self._measure("edit_memory", len(content)) as timing:
            result = self.store.edit(memory_id, content, reason)
            return {**result, "graph": await self._index(memory_id, timing)}

    async def correct(
        self,
        memory_id: str,
        correction: str,
        reason: str,
        source: str | None = None,
    ) -> dict[str, Any]:
        with self._measure("correct_memory", len(correction)) as timing:
            result = self.store.correct(memory_id, correction, reason, source)
            if not result.get("corrected"):
                return {**result, "graph": {"skipped": True}}
            # Do not delete the old episode. The correction is a new episode so Graphiti can preserve
            # history and invalidate contradictory old facts rather than erasing the past.
            return {**result, "graph": await self._index(result["id"], timing)}

    def verify(self, memory_id: str) -> dict[str, Any]:
        with self._measure("verify_memory"):
            return self.store.verify(memory_id)

    async def sync_graph(self) -> dict[str, Any]:
        with self._measure("sync_memory_graph") as timing:
            indexed = 0
            skipped = 0
            errors: list[dict[str, str]] = []
            for record in self.store.iter_records(searchable_only=True):
                if record.metadata.get("status") == "superseded":
                    # Keep historical episodes already in the graph, but don't newly ingest a source
                    # already marked superseded before Graphiti was enabled.
                    skipped += 1
                    continue
                try:
                    with timing.graph():
                        result = await self.graph.ingest(record)
                    if result.get("skipped"):
                        skipped += 1
                    elif result.get("indexed"):
                        indexed += 1
                except Exception as exc:
                    errors.append({"memory_id": record.id, "error": str(exc)})
            if errors:
                timing.error = f"{len(errors)} memories failed to ingest"
            return {"indexed": indexed, "skipped": skipped, "errors": errors}


def _all_superseded(memory_ids: list[str], statuses: dict[str, str]) -> bool:
    # A local guardrail around Graphiti invalidation: if every known source for a derived
    # fact is already superseded in Markdown, the fact is not current.
    known = [statuses[memory_id] for memory_id in memory_ids if memory_id in statuses]
    return bool(known) and all(status == "superseded" for status in known)
