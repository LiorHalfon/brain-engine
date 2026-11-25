from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import sqlite3
import time
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from brain_engine.config import DEFAULT_GRAPH_MODEL, Embedding
from brain_engine.db import connection
from brain_engine.leftovers import stop_leftover_servers
from brain_engine.watchdog import LOG_NAME, Watchdog, start_watchdog
from brain_engine.store import MemoryRecord, content_sha256

logger = logging.getLogger(__name__)

# Graphiti's FalkorDB driver logs a failed query with every parameter, embedding vectors included,
# and the MCP SDK's Rich log handler then gives each number a line of its own, so one failure
# filled over a thousand lines of the service's log. The error and the query's start are enough.
MAX_LOGGED_QUERY_CHARS = 1000


class _ShortQueryErrors(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        if len(message) > MAX_LOGGED_QUERY_CHARS:
            record.msg = f"{message[:MAX_LOGGED_QUERY_CHARS]} ... [{len(message) - MAX_LOGGED_QUERY_CHARS} characters cut]"
            record.args = ()
        return True


logging.getLogger("graphiti_core.driver.falkordb_driver").addFilter(_ShortQueryErrors())

# Graphiti's default search first has OpenAI embed the query. On an open connection most searches
# finish in 0.25-0.45 s. A new connection adds a TLS handshake, and the first search after a pause
# needs one. With a 0.5 s budget, half of all searches and nearly every first one fell back. Past
# this budget, search answers from the graph's keyword index, which is local.
SEARCH_BUDGET_SECONDS = 1.5
# A hybrid search past its budget goes on in the background for up to this long. Cancelling it
# would close the connection its OpenAI request had just opened, so the next search would start
# cold as well.
LATE_SEARCH_SECONDS = 30
# How long an idle connection to OpenAI stays open. httpx's default of 5 s closed it between
# almost any two searches.
OPENAI_KEEPALIVE_SECONDS = 300
# Saving a graph takes well under a second; past this, closing gives up rather than hang the exit.
CLOSE_TIMEOUT_SECONDS = 60
# Re-embedding sends the model this many names or facts per request, and waits this long to try
# again after a failure, such as an embedding server that isn't up yet.
REEMBED_BATCH = 64
REEMBED_RETRY_SECONDS = 60
# graph-state's record of the model that made the graph's vectors, while a re-embed is unfinished.
MIXED_VECTORS = "mixed"
# What Graphiti embeds: the pattern that finds it as `x`, the property it embeds, and the vector's.
VECTORS = [
    ("entity names", "(x:Entity)", "name", "name_embedding"),
    ("facts", "()-[x:RELATES_TO]->()", "fact", "fact_embedding"),
    ("community names", "(x:Community)", "name", "name_embedding"),
]


@dataclass
class GraphFact:
    uuid: str
    fact: str
    valid_at: str | None
    invalid_at: str | None
    source_memory_ids: list[str]


class GraphFacts(list[GraphFact]):
    """A search's facts. `keyword_only` means they come from the graph's keyword index alone,
    because the hybrid search missed its budget or failed."""

    keyword_only: bool = False


class GraphBackend(Protocol):
    enabled: bool
    # Called after the graph opens again on a new redis-server, which may lack the latest writes.
    on_reopen: Callable[[], None] | None

    async def start(self) -> None: ...

    async def ingest(self, record: MemoryRecord) -> dict[str, Any]: ...

    async def search(self, query: str, limit: int = 5, include_invalidated: bool = False) -> list[GraphFact]: ...

    async def remove(self, memory_id: str) -> bool: ...

    def memory_ids(self) -> set[str]: ...

    def ingested_hashes(self) -> dict[str, str]: ...

    async def snapshot(self) -> dict[str, Any]: ...

    async def close(self) -> None: ...


class DisabledGraphBackend:
    enabled = False
    on_reopen = None

    async def start(self) -> None:
        return None

    async def ingest(self, record: MemoryRecord) -> dict[str, Any]:
        return {"enabled": False, "indexed": False, "memory_id": record.id}

    async def search(self, query: str, limit: int = 5, include_invalidated: bool = False) -> list[GraphFact]:
        return []

    async def remove(self, memory_id: str) -> bool:
        return False

    def memory_ids(self) -> set[str]:
        return set()

    def ingested_hashes(self) -> dict[str, str]:
        return {}

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
        embedding: Embedding = Embedding(),
    ) -> None:
        self.state_dir = state_dir
        # Episodes record their source as `<name>:<path>`.
        self.name = name
        self.model = model
        self.small_model = small_model
        self.embedding = embedding
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.graph_db_path = self.state_dir / "graphiti.db"
        self.state_db_path = self.state_dir / "graph-state.sqlite3"
        self.enabled = enabled
        self.on_reopen: Callable[[], None] | None = None
        self._client: Any = None
        self._falkor: Any = None
        # The redis-server this backend started: its socket, its psutil.Process, and its watcher.
        self._server_socket: str | None = None
        self._server: Any = None
        self._watchdog: Watchdog | None = None
        # Set when the server went away, until the graph is open again on a new one.
        self._reopening = False
        # Graphiti's OpenAI client and, for a model on another server, the embedding client, both
        # kept across reopens. Searches left to finish in the background.
        self._openai: Any = None
        self._embedding_server: Any = None
        self._unawaited: set[asyncio.Future[Any]] = set()
        # True while the graph holds vectors another embedding model made, until the re-embed
        # running in `_reembedding` replaces them.
        self._vectors_stale = False
        self._reembedding: asyncio.Task[None] | None = None
        # Concurrent requests must not each open their own FalkorDB Lite instance.
        self._client_lock = asyncio.Lock()
        # One writer at a time. Graphiti reads the nodes and edges an episode touches, spends seconds
        # on the LLM, then saves them back by uuid, so a second writer in between can bring back what
        # the first deleted or lose what it added. It also keeps two ingests of one memory apart: the
        # second finds the first one's state.
        self._write_lock = asyncio.Lock()
        self._init_state_db()

    def _connect_state(self) -> AbstractContextManager[sqlite3.Connection]:
        return connection(self.state_db_path)

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
            conn.execute("CREATE TABLE IF NOT EXISTS graph_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")

    def _meta(self, key: str) -> str | None:
        with self._connect_state() as conn:
            row = conn.execute("SELECT value FROM graph_meta WHERE key=?", (key,)).fetchone()
        return row["value"] if row else None

    def _set_meta(self, key: str, value: str) -> None:
        with self._connect_state() as conn:
            conn.execute(
                "INSERT INTO graph_meta(key, value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )

    async def start(self) -> None:
        """Open the Graphiti/FalkorDB Lite client up front instead of on the first request."""
        if self.enabled:
            await self._get_client()

    async def _get_client(self) -> Any:
        if not self.enabled:
            raise RuntimeError("Graphiti is disabled")
        async with self._client_lock:
            if self._client is not None and (gone := self._server_gone()):
                # Every call would fail until a restart, with "No such file or directory" for a
                # server that was stopped and "Connection refused" for one that crashed.
                logger.warning("The graph's redis-server %s. Opening the graph again on a new one.%s", gone, self._server_log_tail())
                client, self._client = self._client, None
                with contextlib.suppress(Exception):
                    await client.close()  # only drops its connections, which are dead
                self._reopening = True
            if self._client is None:
                self._client = await self._create_client()
                if self._reopening:
                    self._reopening = False
                    if self.on_reopen is not None:
                        self.on_reopen()
            return self._client

    def _server_gone(self) -> str | None:
        """Why the redis-server this backend started can't serve it any more, or None while it can."""
        if self._server_socket is None:
            return None  # nothing started here, or a stand-in client
        if self._server is not None and not self._server.is_running():
            return f"(pid {self._server.pid}) is no longer running"
        if not os.path.exists(self._server_socket):
            return f"socket {self._server_socket} is gone"
        return None

    def _server_log_tail(self) -> str:
        # redislite keeps the server's log beside its socket, in a temp folder macOS empties after
        # a few days, so quote its last line while it is there.
        if self._server_socket is None:
            return ""
        log = Path(self._server_socket).with_name("redis.log")
        try:
            lines = [line.strip() for line in log.read_text(errors="replace").splitlines() if line.strip()]
        except OSError:
            return ""
        return f" Its log {log} ends: {lines[-1]}" if lines else ""

    def _forget_server(self) -> None:
        """Forget the redis-server this backend started, and let its watcher go."""
        if self._watchdog is not None:
            self._watchdog.release()
            self._watchdog = None
        self._falkor = self._server = self._server_socket = None

    async def _create_client(self) -> Any:
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is required for Graphiti ingestion/search")

        # Keep business data out of Graphiti telemetry. Their documented telemetry does not
        # include graph contents, but there is no benefit in enabling it for a local vault.
        os.environ.setdefault("GRAPHITI_TELEMETRY_ENABLED", "false")

        try:
            from graphiti_core import Graphiti
            from graphiti_core.driver.falkordb_driver import FalkorDriver
            from graphiti_core.embedder import OpenAIEmbedder, OpenAIEmbedderConfig
            from redislite.async_falkordb_client import AsyncFalkorDB
        except ImportError as exc:  # pragma: no cover - depends on optional runtime dependency
            raise RuntimeError(
                "Graphiti dependencies are missing. Install brain-engine with its `graph` extra."
            ) from exc

        stop_leftover_servers(self.graph_db_path)
        # The server an earlier open started, gone or left half-open, is stopped now too.
        self._forget_server()
        # redislite reattaches to the server its registry file names if that process still runs. No
        # server for this graph is left now, so the registry is stale; in a copied state dir it names
        # another vault's server.
        self.graph_db_path.with_name(self.graph_db_path.name + ".settings").unlink(missing_ok=True)
        falkor = AsyncFalkorDB(dbfilename=str(self.graph_db_path))
        self._falkor = falkor
        self._server_socket = falkor.connection.socket_file
        self._server = self._server_process(falkor.connection.pid)
        try:
            self._watchdog = start_watchdog(self._server_socket, self.state_dir / LOG_NAME)
        except Exception as exc:
            logger.warning("No watcher for the graph's redis-server; a killed service will leave it running: %s", exc)
        driver = FalkorDriver(falkor_db=falkor)
        if self._openai is None:
            self._openai = self._openai_client()
        embedder = OpenAIEmbedder(
            config=OpenAIEmbedderConfig(embedding_model=self.embedding.model, embedding_dim=self.embedding.dimensions),
            client=self._embedding_client(),
        )
        client = Graphiti(graph_driver=driver, llm_client=self._llm_client(self._openai), embedder=embedder)
        await client.build_indices_and_constraints()
        await self._forget_missing_episodes(client)
        await self._check_vectors(client)
        return client

    def _embedding_client(self) -> Any:
        if self.embedding.base_url is None:
            return self._openai  # OpenAI's model shares the LLM's connections
        if self._embedding_server is None:
            key_env = self.embedding.api_key_env
            if key_env and not os.environ.get(key_env):
                raise RuntimeError(f"{key_env}, which brain.toml's embedding_api_key_env names, is not set")
            self._embedding_server = self._openai_client(
                base_url=self.embedding.base_url,
                # The OpenAI client refuses to start without a key, and a local server takes any.
                api_key=os.environ[key_env] if key_env else "unused",
            )
        return self._embedding_server

    async def _check_vectors(self, client: Any) -> None:
        """Re-embed the graph in the background if another embedding model made its vectors.

        A query vector compares only with vectors from the same model, so until the re-embed is
        done search answers from the graph's keyword index, and writes wait for it.
        """
        wanted = self.embedding.identity
        made_by = self._meta("embedding")
        counts = {}
        for what, pattern, _, vector in VECTORS:
            records, _, _ = await client.driver.execute_query(
                f"MATCH {pattern} WHERE x.{vector} IS NOT NULL RETURN count(x) AS count"
            )
            counts[what] = int(records[0]["count"])
        if made_by is None:
            # Graphs from before 0.1.6 don't say. The only model they could use was Graphiti's
            # default, and an empty graph has no vectors yet.
            made_by = Embedding().identity if sum(counts.values()) else wanted
            self._set_meta("embedding", made_by)
        self._vectors_stale = made_by != wanted
        if not self._vectors_stale:
            return
        logger.warning(
            "%s Re-embedding %s with %s in the background. Until it's done, search takes its graph facts "
            "from the keyword index and writes to the graph wait.",
            "A re-embed of the graph's vectors did not finish."
            if made_by == MIXED_VECTORS
            else f"The graph's vectors were made by {made_by}.",
            ", ".join(f"{count} {what}" for what, count in counts.items()),
            wanted,
        )
        running = self._reembedding
        if running is not None and running is asyncio.current_task():
            return  # this re-embed goes on, on the new server
        if running is not None:
            running.cancel()  # its server is gone
        self._reembedding = asyncio.create_task(self._reembed_in_background())

    async def _reembed_in_background(self) -> None:
        while self._vectors_stale:
            try:
                async with self._write_lock:
                    await self._reembed_if_stale(await self._get_client())
            except Exception as exc:
                logger.warning("Re-embedding the graph failed, trying again in %ss: %s", REEMBED_RETRY_SECONDS, exc)
                await asyncio.sleep(REEMBED_RETRY_SECONDS)

    async def _reembed_if_stale(self, client: Any) -> None:
        # The caller holds `_write_lock`.
        if self._vectors_stale:
            await self._reembed(client)
            self._vectors_stale = False

    async def _reembed(self, client: Any) -> None:
        """Have the configured model make every vector in the graph again."""
        start = time.monotonic()
        # Set first, so a re-embed that stops halfway is done again in full.
        self._set_meta("embedding", MIXED_VECTORS)
        done = []
        for what, pattern, embedded, vector in VECTORS:
            found = f"MATCH {pattern} WHERE x.{vector} IS NOT NULL"
            # Graphiti doesn't embed a blank name or fact, and an embedding API may refuse one.
            await client.driver.execute_query(f"{found} AND trim(coalesce(x.{embedded}, '')) = '' SET x.{vector} = NULL")
            records, _, _ = await client.driver.execute_query(f"{found} RETURN x.uuid AS uuid, x.{embedded} AS text")
            # Graphiti embeds a name or a fact with its newlines turned into spaces.
            rows = [(str(record["uuid"]), str(record["text"]).replace("\n", " ")) for record in records]
            for at in range(0, len(rows), REEMBED_BATCH):
                batch = rows[at : at + REEMBED_BATCH]
                vectors = await client.embedder.create_batch([text for _, text in batch])
                await client.driver.execute_query(
                    f"UNWIND $rows AS row MATCH {pattern} WHERE x.uuid = row.uuid SET x.{vector} = vecf32(row.vector)",
                    rows=[{"uuid": uuid, "vector": values} for (uuid, _), values in zip(batch, vectors, strict=True)],
                )
            done.append(f"{len(rows)} {what}")
        if self._falkor is not None:
            # The redis-server saves on a timer. Saved now, a crash can't bring back old vectors
            # under a record that says they are new.
            await self._falkor.connection.execute_command("SAVE")
        self._set_meta("embedding", self.embedding.identity)
        logger.warning("Re-embedded %s with %s in %.0f s.", ", ".join(done), self.embedding.identity, time.monotonic() - start)

    @staticmethod
    def _server_process(pid: int | None) -> Any:
        # psutil.Process.is_running() checks the pid against the process's start time since boot,
        # which a clock change doesn't move, so it can't mistake a reused pid for the server.
        import psutil

        try:
            return psutil.Process(pid) if pid else None
        except psutil.NoSuchProcess:
            return None

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

    def _llm_client(self, openai: Any) -> Any:
        # Named explicitly: graphiti-core 0.30.2's default small model was retired by OpenAI, and
        # every ingest failed with model_not_found. The embedder and reranker keep their models.
        from graphiti_core.llm_client import LLMConfig, OpenAIClient

        return OpenAIClient(config=LLMConfig(model=self.model, small_model=self.small_model), client=openai)

    @staticmethod
    def _openai_client(**options: Any) -> Any:
        """An OpenAI client for Graphiti's LLM and embedding calls, which keeps idle connections open."""
        import openai

        try:
            import httpx2 as http  # what openai>=3 sends requests with
        except ImportError:
            import httpx as http
        limits = http.Limits(max_connections=1000, max_keepalive_connections=100, keepalive_expiry=OPENAI_KEEPALIVE_SECONDS)
        return openai.AsyncOpenAI(http_client=openai.DefaultAsyncHttpxClient(limits=limits), **options)

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

        async with self._write_lock:
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
        # Graphiti compares a new episode's names and facts with the graph's by their vectors.
        await self._reembed_if_stale(client)
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
        results, keyword_only = await self._search_edges(client, query)
        facts = GraphFacts()
        facts.keyword_only = keyword_only
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

    async def _search_edges(self, client: Any, query: str) -> tuple[list[Any], bool]:
        """Graphiti's hybrid search when it answers within the budget, else the keyword index's
        edges. The second value says which."""
        if self._vectors_stale:
            # The graph holds vectors from two models until the re-embed is done.
            return await self._keyword_search(client, query), True
        hybrid = asyncio.ensure_future(asyncio.wait_for(self._hybrid_search(client, query), LATE_SEARCH_SECONDS))
        try:
            done, _ = await asyncio.wait({hybrid}, timeout=SEARCH_BUDGET_SECONDS)
        except asyncio.CancelledError:
            self._let_finish(hybrid)
            raise
        if not done:
            self._let_finish(hybrid)
        elif hybrid.exception() is None:
            return hybrid.result(), False
        else:
            # Usually OpenAI. The keyword index needs only the graph, which answers or fails here too.
            logger.info("Graphiti's hybrid search failed, so search answers from the keyword index: %s", hybrid.exception())
        return await self._keyword_search(client, query), True

    async def _hybrid_search(self, client: Any, query: str) -> list[Any]:
        from graphiti_core.search.search_config import DEFAULT_SEARCH_LIMIT
        from graphiti_core.search.search_config_recipes import EDGE_HYBRID_SEARCH_RRF

        if not query.strip():
            return []
        # Graphiti's own `search`, which embeds the query as it stands. Some models, Qwen3-Embedding
        # among them, need an instruction in front of a query to find what answers it.
        vector = await client.embedder.create(input_data=[self.embedding.query_instruction + query.replace("\n", " ")])
        config = EDGE_HYBRID_SEARCH_RRF.model_copy(update={"limit": DEFAULT_SEARCH_LIMIT})
        return (await client.search_(query, config=config, query_vector=vector)).edges

    def _let_finish(self, task: asyncio.Future[Any]) -> None:
        # Keep a reference so the task runs to the end, and take its error so none is reported unread.
        self._unawaited.add(task)
        task.add_done_callback(self._unawaited.discard)
        task.add_done_callback(lambda done: done.cancelled() or done.exception())

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

    async def remove(self, memory_id: str) -> bool:
        """Remove the memory's episode and forget the memory. Returns whether it had a row to forget.

        Graphiti deletes the facts and entities only this episode produced, drops the episode from
        the sources of facts other episodes share, and asks the LLM to summarize again the entities
        that remain. A fact this episode invalidated stays invalidated. If the removal fails, the
        memory's row stays, so the next sync tries again.
        """
        # Waits for an ingest in flight, which can't then add an episode after the removal.
        async with self._write_lock:
            previous = self._state(memory_id)
            if previous is None:
                return False
            if previous["episode_uuid"]:
                from graphiti_core.errors import NodeNotFoundError

                client = await self._get_client()
                try:
                    await client.remove_episode(previous["episode_uuid"])
                except NodeNotFoundError:
                    pass  # already gone from the graph
            with self._connect_state() as conn:
                conn.execute("DELETE FROM graph_ingestion WHERE memory_id=?", (memory_id,))
            return True

    def memory_ids(self) -> set[str]:
        """Every memory graph-state has a row for, including ingests that failed."""
        with self._connect_state() as conn:
            return {row["memory_id"] for row in conn.execute("SELECT memory_id FROM graph_ingestion")}

    def ingested_hashes(self) -> dict[str, str]:
        """The content hash of every memory whose episode is in the graph. `ingest` skips a memory
        whose text still has this hash."""
        with self._connect_state() as conn:
            return {
                row["memory_id"]: row["content_sha256"]
                for row in conn.execute("SELECT memory_id, content_sha256 FROM graph_ingestion WHERE status='ok'")
            }

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
        # A re-embed cut short is done again in full by the next start.
        pending = [*self._unawaited, *([self._reembedding] if self._reembedding is not None else [])]
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        self._reembedding = None
        if self._falkor is not None:
            if not self._server_gone():
                try:
                    await asyncio.wait_for(self._stop_redis(self._falkor.connection.socket_file), CLOSE_TIMEOUT_SECONDS)
                except TimeoutError:
                    logger.warning("The graph's redis-server did not stop within %ss", CLOSE_TIMEOUT_SECONDS)
            if not self._server_gone():
                # Keep the watcher's pipe open: it stops the server once this process is gone, and
                # the next start would too.
                self._watchdog = None
            self._forget_server()
        if self._client is not None:
            await self._client.close()
            self._client = None
        for name in ("_openai", "_embedding_server"):
            if getattr(self, name) is not None:
                await getattr(self, name).close()
                setattr(self, name, None)

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
