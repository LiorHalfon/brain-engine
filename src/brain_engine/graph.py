from __future__ import annotations

import asyncio
import logging
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from brain_engine.config import DEFAULT_GRAPH_MODEL
from brain_engine.leftovers import stop_leftover_servers
from brain_engine.watchdog import start_watchdog
from brain_engine.store import MemoryRecord, content_sha256

logger = logging.getLogger(__name__)

# Graphiti's default search first has OpenAI embed the query: about 0.25 s on a warm connection, a
# second or more on a cold one, and as long as OpenAI takes when it is struggling. Past this budget,
# search answers from the graph's keyword index, which is local and takes about 40 ms.
SEARCH_BUDGET_SECONDS = 0.5
# Saving a graph takes well under a second; past this, closing gives up rather than hang the exit.
CLOSE_TIMEOUT_SECONDS = 60


@dataclass
class GraphFact:
    uuid: str
    fact: str
    valid_at: str | None
    invalid_at: str | None
    source_memory_ids: list[str]


class GraphBackend(Protocol):
    enabled: bool

    async def start(self) -> None: ...

    async def ingest(self, record: MemoryRecord) -> dict[str, Any]: ...

    async def search(self, query: str, limit: int = 5, include_invalidated: bool = False) -> list[GraphFact]: ...

    async def remove(self, memory_id: str) -> None: ...

    async def snapshot(self) -> dict[str, Any]: ...

    async def close(self) -> None: ...


class DisabledGraphBackend:
    enabled = False

    async def start(self) -> None:
        return None

    async def ingest(self, record: MemoryRecord) -> dict[str, Any]:
        return {"enabled": False, "indexed": False, "memory_id": record.id}

    async def search(self, query: str, limit: int = 5, include_invalidated: bool = False) -> list[GraphFact]:
        return []

    async def remove(self, memory_id: str) -> None:
        return None

    async def snapshot(self) -> dict[str, Any]:
        return {"enabled": False, "nodes": [], "edges": []}

    async def close(self) -> None:
        return None


class GraphitiBackend:
    """Derived temporal graph. Markdown remains authoritative.

    The Graphiti dependency is imported lazily so the plain Markdown/FTS store remains usable
    even when Graphiti or an LLM API key is unavailable.
    """

    def __init__(
        self,
        state_dir: Path,
        *,
        enabled: bool = True,
        name: str = "brain-engine",
        model: str = DEFAULT_GRAPH_MODEL,
        small_model: str = DEFAULT_GRAPH_MODEL,
    ) -> None:
        self.state_dir = state_dir
        # Episodes record their source as `<name>:<path>`.
        self.name = name
        self.model = model
        self.small_model = small_model
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.graph_db_path = self.state_dir / "graphiti.db"
        self.state_db_path = self.state_dir / "graph-state.sqlite3"
        self.enabled = enabled
        self._client: Any = None
        self._falkor: Any = None
        # Concurrent requests must not each open their own FalkorDB Lite instance.
        self._client_lock = asyncio.Lock()
        self._memory_locks: dict[str, asyncio.Lock] = {}
        self._init_state_db()

    def _connect_state(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.state_db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_state_db(self) -> None:
        with self._connect_state() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS graph_ingestion (
                    memory_id TEXT PRIMARY KEY,
                    content_sha256 TEXT NOT NULL,
                    episode_uuid TEXT,
                    status TEXT NOT NULL,
                    last_error TEXT,
                    indexed_at TEXT
                )
                """
            )

    async def start(self) -> None:
        """Open the Graphiti/FalkorDB Lite client up front instead of on the first request."""
        if self.enabled:
            await self._get_client()

    async def _get_client(self) -> Any:
        if not self.enabled:
            raise RuntimeError("Graphiti is disabled")
        async with self._client_lock:
            if self._client is None:
                self._client = await self._create_client()
            return self._client

    async def _create_client(self) -> Any:
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is required for Graphiti ingestion/search")

        # Keep business data out of Graphiti telemetry. Their documented telemetry does not
        # include graph contents, but there is no benefit in enabling it for a local vault.
        os.environ.setdefault("GRAPHITI_TELEMETRY_ENABLED", "false")

        try:
            from graphiti_core import Graphiti
            from graphiti_core.driver.falkordb_driver import FalkorDriver
            from redislite.async_falkordb_client import AsyncFalkorDB
        except ImportError as exc:  # pragma: no cover - depends on optional runtime dependency
            raise RuntimeError(
                "Graphiti dependencies are missing. Install brain-engine with its `graph` extra."
            ) from exc

        stop_leftover_servers(self.graph_db_path)
        # redislite reattaches to the server its registry file names if that process still runs. No
        # server for this graph is left now, so the registry is stale; in a copied state dir it names
        # another vault's server.
        self.graph_db_path.with_name(self.graph_db_path.name + ".settings").unlink(missing_ok=True)
        falkor = AsyncFalkorDB(dbfilename=str(self.graph_db_path))
        self._falkor = falkor
        try:
            start_watchdog(falkor.connection.socket_file)
        except Exception as exc:
            logger.warning("No watcher for the graph's redis-server; a killed service will leave it running: %s", exc)
        driver = FalkorDriver(falkor_db=falkor)
        client = Graphiti(graph_driver=driver, llm_client=self._llm_client())
        await client.build_indices_and_constraints()
        await self._forget_missing_episodes(client)
        return client

    async def _forget_missing_episodes(self, client: Any) -> int:
        """Mark memories whose episode is no longer in the graph as not ingested.

        A redis-server left running by an earlier service could write an older graph over
        graphiti.db, dropping episodes graph-state still calls ok. Marked `missing`, they are
        ingested again by the start-up catch-up and by sync-graph.
        """
        records, _, _ = await client.driver.execute_query("MATCH (e:Episodic) RETURN e.uuid AS uuid")
        in_graph = {str(record["uuid"]) for record in records}
        with self._connect_state() as conn:
            rows = conn.execute(
                "SELECT memory_id, episode_uuid FROM graph_ingestion WHERE status='ok' AND episode_uuid IS NOT NULL"
            ).fetchall()
            missing = [row["memory_id"] for row in rows if row["episode_uuid"] not in in_graph]
            conn.executemany(
                "UPDATE graph_ingestion SET status='missing', episode_uuid=NULL, "
                "last_error='episode missing from graph' WHERE memory_id=?",
                [(memory_id,) for memory_id in missing],
            )
        if missing:
            logger.warning("Memories marked ingested whose episode is missing from the graph, to be ingested again: %d", len(missing))
        return len(missing)

    def _llm_client(self) -> Any:
        # Named explicitly: graphiti-core 0.30.2's default small model was retired by OpenAI, and
        # every ingest failed with model_not_found. The embedder and reranker keep their defaults.
        from graphiti_core.llm_client import LLMConfig, OpenAIClient

        return OpenAIClient(config=LLMConfig(model=self.model, small_model=self.small_model))

    @staticmethod
    def _reference_time(record: MemoryRecord) -> datetime:
        raw = str(record.metadata.get("created_at") or "")
        if raw:
            try:
                return datetime.fromisoformat(raw.replace("Z", "+00:00"))
            except ValueError:
                pass
        return datetime.now(timezone.utc)

    def _state(self, memory_id: str) -> sqlite3.Row | None:
        with self._connect_state() as conn:
            return conn.execute(
                "SELECT * FROM graph_ingestion WHERE memory_id=?", (memory_id,)
            ).fetchone()

    def _write_state(
        self,
        memory_id: str,
        content_hash: str,
        *,
        episode_uuid: str | None,
        status: str,
        error: str | None = None,
    ) -> None:
        now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        with self._connect_state() as conn:
            conn.execute(
                """
                INSERT INTO graph_ingestion(memory_id, content_sha256, episode_uuid, status, last_error, indexed_at)
                VALUES(?,?,?,?,?,?)
                ON CONFLICT(memory_id) DO UPDATE SET
                    content_sha256=excluded.content_sha256,
                    episode_uuid=excluded.episode_uuid,
                    status=excluded.status,
                    last_error=excluded.last_error,
                    indexed_at=excluded.indexed_at
                """,
                (memory_id, content_hash, episode_uuid, status, error, now),
            )

    async def ingest(self, record: MemoryRecord) -> dict[str, Any]:
        if not self.enabled or record.metadata.get("searchable", True) is False:
            return {"enabled": self.enabled, "indexed": False, "memory_id": record.id}

        # edit_memory and a sync can reach the same memory at once. Unserialized, both remove the
        # same old episode and each add a new one. The second waits here and then finds the
        # first one's state.
        async with self._memory_locks.setdefault(record.id, asyncio.Lock()):
            return await self._ingest(record)

    async def _ingest(self, record: MemoryRecord) -> dict[str, Any]:
        # Recompute from the body so manual Markdown edits are detected even if frontmatter
        # was not updated through the MCP server.
        content_hash = content_sha256(str(record.metadata.get("type") or "note"), record.content)
        previous = self._state(record.id)
        if previous and previous["status"] == "ok" and previous["content_sha256"] == content_hash:
            return {
                "enabled": True,
                "indexed": True,
                "skipped": True,
                "memory_id": record.id,
                "episode_uuid": previous["episode_uuid"],
            }

        client = await self._get_client()
        if previous and previous["episode_uuid"]:
            # An in-place Markdown edit is a source correction. Replace the episode rather than
            # asking Graphiti to treat both source versions as distinct life events.
            from graphiti_core.errors import NodeNotFoundError

            try:
                await client.remove_episode(previous["episode_uuid"])
            except NodeNotFoundError:
                # Graphiti 0.31 raises this from its episode lookup: the old episode is already
                # gone, so there is nothing left to replace.
                pass
            except Exception as exc:
                # Keep the old episode uuid so the next sync tries the removal again.
                self._write_state(
                    record.id,
                    content_hash,
                    episode_uuid=previous["episode_uuid"],
                    status="error",
                    error=str(exc),
                )
                raise

        try:
            from graphiti_core.nodes import EpisodeType

            result = await client.add_episode(
                name=record.id,
                episode_body=record.content,
                source=EpisodeType.text,
                source_description=f"{self.name}:{record.path}",
                reference_time=self._reference_time(record),
            )
            episode_uuid = result.episode.uuid
            self._write_state(
                record.id,
                content_hash,
                episode_uuid=episode_uuid,
                status="ok",
            )
            return {
                "enabled": True,
                "indexed": True,
                "memory_id": record.id,
                "episode_uuid": episode_uuid,
                "facts_added": len(getattr(result, "edges", []) or []),
            }
        except Exception as exc:
            self._write_state(
                record.id,
                content_hash,
                episode_uuid=None,
                status="error",
                error=str(exc),
            )
            raise

    def _memory_ids_for_episodes(self, episode_uuids: list[str]) -> list[str]:
        if not episode_uuids:
            return []
        placeholders = ",".join("?" for _ in episode_uuids)
        with self._connect_state() as conn:
            rows = conn.execute(
                f"SELECT memory_id FROM graph_ingestion WHERE episode_uuid IN ({placeholders})",
                episode_uuids,
            ).fetchall()
        return [row["memory_id"] for row in rows]

    async def search(
        self,
        query: str,
        limit: int = 5,
        include_invalidated: bool = False,
    ) -> list[GraphFact]:
        if not self.enabled:
            return []
        client = await self._get_client()
        # Graphiti's public search returns relationship/fact edges. Slice locally to avoid coupling
        # to a version-specific limit parameter.
        try:
            results = await asyncio.wait_for(client.search(query), SEARCH_BUDGET_SECONDS)
        except TimeoutError:
            results = await self._keyword_search(client, query)
        facts: list[GraphFact] = []
        for edge in results:
            invalid_at = getattr(edge, "invalid_at", None)
            if invalid_at is not None and not include_invalidated:
                continue
            episodes = list(getattr(edge, "episodes", []) or [])
            facts.append(
                GraphFact(
                    uuid=str(edge.uuid),
                    fact=str(edge.fact)[:1200],
                    valid_at=edge.valid_at.isoformat() if getattr(edge, "valid_at", None) else None,
                    invalid_at=invalid_at.isoformat() if invalid_at else None,
                    source_memory_ids=self._memory_ids_for_episodes(episodes),
                )
            )
            if len(facts) >= max(1, min(limit, 10)):
                break
        return facts

    @staticmethod
    async def _keyword_search(client: Any, query: str) -> list[Any]:
        from graphiti_core.search.search_config import (
            EdgeReranker,
            EdgeSearchConfig,
            EdgeSearchMethod,
            SearchConfig,
        )

        # The BM25 half of Graphiti's default hybrid search, ranked the same way, without the embedding.
        config = SearchConfig(
            edge_config=EdgeSearchConfig(search_methods=[EdgeSearchMethod.bm25], reranker=EdgeReranker.rrf),
            limit=10,
        )
        return (await client.search_(query, config=config)).edges

    async def remove(self, memory_id: str) -> None:
        previous = self._state(memory_id)
        if not previous or not previous["episode_uuid"]:
            return
        client = await self._get_client()
        await client.remove_episode(previous["episode_uuid"])
        with self._connect_state() as conn:
            conn.execute("DELETE FROM graph_ingestion WHERE memory_id=?", (memory_id,))

    async def snapshot(self) -> dict[str, Any]:
        """All entities and facts, for the read-only /graph viewer. Invalidated facts are included."""
        if not self.enabled:
            return {"enabled": False, "nodes": [], "edges": []}
        client = await self._get_client()
        nodes, _, _ = await client.driver.execute_query(
            "MATCH (n:Entity) RETURN n.uuid AS uuid, n.name AS name, n.summary AS summary"
        )
        edges, _, _ = await client.driver.execute_query(
            """
            MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity)
            RETURN r.uuid AS uuid, a.uuid AS source, b.uuid AS target, r.name AS name, r.fact AS fact,
                   r.valid_at AS valid_at, r.invalid_at AS invalid_at, r.episodes AS episodes
            """
        )
        return {
            "enabled": True,
            "nodes": [
                {"uuid": str(n["uuid"]), "name": str(n["name"]), "summary": str(n.get("summary") or "")}
                for n in nodes
            ],
            "edges": [
                {
                    "uuid": str(e["uuid"]),
                    "source": str(e["source"]),
                    "target": str(e["target"]),
                    "name": str(e["name"] or ""),
                    "fact": str(e["fact"] or ""),
                    "valid_at": str(e["valid_at"]) if e.get("valid_at") else None,
                    "invalid_at": str(e["invalid_at"]) if e.get("invalid_at") else None,
                    "source_memory_ids": self._memory_ids_for_episodes([str(x) for x in e.get("episodes") or []]),
                }
                for e in edges
            ],
        }

    async def close(self) -> None:
        if self._falkor is not None:
            try:
                await asyncio.wait_for(self._stop_redis(self._falkor.connection.socket_file), CLOSE_TIMEOUT_SECONDS)
            except TimeoutError:
                # Its watcher stops it once this process is gone, and the next start would too.
                logger.warning("The graph's redis-server did not stop within %ss", CLOSE_TIMEOUT_SECONDS)
            self._falkor = None
        if self._client is not None:
            await self._client.close()
            self._client = None

    @staticmethod
    async def _stop_redis(socket_file: str) -> None:
        # FalkorDB Lite runs a redis-server that saves graphiti.db only on a timer, and neither
        # Graphiti's close (it drops the connection pool) nor redislite stops it. It outlived the
        # service, or redislite SIGKILLed it at exit with its newest writes unsaved.
        from redis.asyncio import Redis
        from redis.asyncio.retry import Retry
        from redis.backoff import NoBackoff

        # No retries: the server drops the connection as it exits, and the default client then
        # spends seconds reconnecting to a socket that is gone.
        redis = Redis(unix_socket_path=socket_file, retry=Retry(NoBackoff(), 0))
        try:
            await redis.shutdown(save=True)
        except Exception as exc:
            logger.warning("Could not save and stop the graph's redis-server: %s", exc)
        finally:
            await redis.aclose()
