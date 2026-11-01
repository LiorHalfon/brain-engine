from __future__ import annotations

import asyncio
import logging
import os
import signal
import socket
import sys
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import anyio
import uvicorn
from mcp.server import MCPServer
from mcp.server.subscriptions import ListenHandler
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, PlainTextResponse, Response

from brain_engine.config import Settings
from brain_engine.lock import hold_state_lock
from brain_engine.service import MemoryService

logger = logging.getLogger(__name__)

GRAPH_VIEW = Path(__file__).with_name("graph_view.html")
SHUTDOWN_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
LOCAL_HOSTNAMES = {"127.0.0.1", "localhost", "::1"}


class PortInUse(RuntimeError):
    pass


def _ensure_port_free(host: str, port: int) -> None:
    # uvicorn reports a failed bind as SystemExit from inside its startup; check up front so
    # the user gets one clear line instead, before Graphiti is opened.
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError as exc:
            raise PortInUse(
                f"{host}:{port} is already in use by another process ({exc.strerror}). "
                "Stop it or set MEMORY_PORT."
            ) from None


class _Server(uvicorn.Server):
    """Ends every MCP listen stream before uvicorn waits for connections to close.

    Each connected client keeps a `subscriptions/listen` request open for server events. uvicorn
    would wait out `timeout_graceful_shutdown` for them, then cancel them and log a traceback per
    client. Closing them is the spec's graceful end: each client gets a final result and listens
    again once the service is back.
    """

    def __init__(self, config: uvicorn.Config, mcp: MCPServer) -> None:
        super().__init__(config)
        self.mcp = mcp

    async def shutdown(self, sockets: list[socket.socket] | None = None) -> None:
        # MCPServer registers its ListenHandler on the lowlevel server and has no public accessor.
        lowlevel = getattr(self.mcp, "_lowlevel_server", None)
        entry = lowlevel.get_request_handler("subscriptions/listen") if lowlevel is not None else None
        if entry is not None and isinstance(entry.handler, ListenHandler):
            entry.handler.close()
        await super().shutdown(sockets)

    @contextmanager
    def capture_signals(self) -> Iterator[None]:
        # `_stop_on_signals` handles them for the service's whole life instead. uvicorn's own
        # handlers re-raise SIGTERM once uvicorn is done, which ended the process before the
        # service closed the graph.
        yield


@contextmanager
def _stop_on_signals(on_signal: Callable[[int], None]) -> Iterator[None]:
    """Call `on_signal` for Ctrl-C (SIGINT), `kill` (SIGTERM) and a closed terminal (SIGHUP) alike.

    By default only Ctrl-C ended the service through its `finally`. SIGTERM and SIGHUP killed it
    and left FalkorDB Lite's redis-server running, to write an older graph over graphiti.db later.
    """
    if threading.current_thread() is not threading.main_thread():
        yield  # signal handlers can only be set from the main thread
        return
    originals = {
        sig: signal.signal(sig, lambda sig, frame: on_signal(sig))
        for sig in SHUTDOWN_SIGNALS
        # Ignored on purpose, as `nohup` does with SIGHUP: keep it that way.
        if signal.getsignal(sig) is not signal.SIG_IGN
    }
    try:
        yield
    finally:
        for sig, handler in originals.items():
            signal.signal(sig, handler)


def _cancel_soon(task: asyncio.Task) -> None:
    # A signal handler runs between two bytecodes of whatever the loop is doing; let the loop cancel.
    task.get_loop().call_soon_threadsafe(task.cancel)


def create_server(service: MemoryService, settings: Settings) -> MCPServer:
    """Build the MCP facade over one shared MemoryService. Graphiti stays behind the service.

    The server name and the instructions sent to agents come from the vault's brain.toml.
    """
    mcp = MCPServer(settings.name, instructions=settings.instructions)

    @mcp.tool()
    async def search_memory(
        query: str,
        limit: int = 5,
        kind: str | None = None,
        include_history: bool = False,
    ) -> dict:
        """Search compact graph facts and Markdown snippets without loading the whole memory vault."""
        return await service.search(query, limit, kind, include_history)

    @mcp.tool()
    def get_memory(memory_id: str, offset: int = 0, max_chars: int = 8000) -> dict:
        """Read one source memory. Large memories are paged to avoid context bloat."""
        return service.get(memory_id, offset, max_chars)

    # Read-only graph viewer for the owner's browser. These are plain HTTP routes, not MCP tools,
    # so agents still only see the memory API. The Host check blocks DNS-rebinding pages from
    # reading memories off localhost.
    @mcp.custom_route("/graph", methods=["GET"], include_in_schema=False)
    async def graph_page(request: Request) -> Response:
        if request.url.hostname not in LOCAL_HOSTNAMES:
            return PlainTextResponse("Forbidden", status_code=403)
        return HTMLResponse(GRAPH_VIEW.read_text(encoding="utf-8"))

    @mcp.custom_route("/graph.json", methods=["GET"], include_in_schema=False)
    async def graph_data(request: Request) -> Response:
        if request.url.hostname not in LOCAL_HOSTNAMES:
            return PlainTextResponse("Forbidden", status_code=403)
        return JSONResponse(await service.graph_snapshot())

    if settings.access != "read-write":
        return mcp

    @mcp.tool()
    async def remember(
        title: str,
        content: str,
        kind: str = "note",
        tags: list[str] | None = None,
        source: str | None = None,
        verified_by_user: bool = False,
        searchable: bool = True,
    ) -> dict:
        """Create a durable memory. Exact duplicates are rejected before graph ingestion."""
        return await service.remember(
            title=title,
            content=content,
            kind=kind,
            tags=tags,
            source=source,
            verified_by_user=verified_by_user,
            searchable=searchable,
        )

    @mcp.tool()
    async def remember_session(
        title: str,
        summary: str,
        transcript: str | None = None,
        tags: list[str] | None = None,
        source: str | None = None,
    ) -> dict:
        """Save a searchable summary and optionally a non-indexed verbatim transcript."""
        return await service.remember_session(
            title=title,
            summary=summary,
            transcript=transcript,
            tags=tags,
            source=source,
        )

    @mcp.tool()
    async def edit_memory(memory_id: str, content: str, reason: str | None = None) -> dict:
        """Fix the source memory in place. Use only for errors in the stored source, not changed reality."""
        return await service.edit(memory_id, content, reason)

    @mcp.tool()
    async def correct_memory(
        memory_id: str,
        correction: str,
        reason: str,
        source: str | None = None,
    ) -> dict:
        """Add a correction as a new historical episode and supersede the old Markdown memory."""
        return await service.correct(memory_id, correction, reason, source)

    @mcp.tool()
    def verify_memory(memory_id: str) -> dict:
        """Mark a memory as explicitly verified by the owner."""
        return service.verify(memory_id)

    @mcp.tool()
    async def sync_memory_graph() -> dict:
        """Ingest searchable Markdown memories not yet represented in Graphiti, and remove those whose file was deleted."""
        return await service.sync_graph()

    return mcp


async def serve_http(
    settings: Settings,
    service: MemoryService | None = None,
    *,
    stop: anyio.Event | None = None,
) -> None:
    """Run the long-lived Streamable HTTP service.

    Storage, FTS and Graphiti are initialized once here and shared by every client session.
    `stop` lets tests shut the server down cleanly.
    """
    service = service or MemoryService.from_settings(settings)
    with hold_state_lock(service.store.state_dir, settings.url, settings.name):
        _ensure_port_free(settings.host, settings.port)
        mcp = create_server(service, settings)
        app = mcp.streamable_http_app(host=settings.host)
        # _Server ends the listen streams, so the timeout only cuts off a tool call still running.
        config = uvicorn.Config(
            app, host=settings.host, port=settings.port, log_level="info", timeout_graceful_shutdown=3
        )
        server = _Server(config, mcp)
        main = asyncio.current_task()
        assert main is not None
        serving = False

        def on_signal(sig: int) -> None:
            if serving:
                server.handle_exit(sig, None)  # uvicorn's graceful shutdown; a second Ctrl-C forces it
            elif not server.should_exit:
                server.should_exit = True
                _cancel_soon(main)  # still opening the graph: stop there and close what is open

        with _stop_on_signals(on_signal):
            try:
                await service.start(ingest_in_background=True)
                async with anyio.create_task_group() as tg:
                    if stop is not None:

                        async def _watch_stop() -> None:
                            await stop.wait()
                            server.should_exit = True

                        tg.start_soon(_watch_stop)
                    serving = True
                    await server.serve()
                    serving = False
                    tg.cancel_scope.cancel()
            finally:
                serving = False
                await service.close()
                logger.info("%s stopped", settings.name)


async def serve_stdio(settings: Settings) -> None:
    """Legacy transport: one server per client process. Only one may run at a time."""
    service = MemoryService.from_settings(settings)
    loop = asyncio.get_running_loop()
    # A client often closes stdin and then signals, so the service is closed by whichever comes
    # first and the other waits for it.
    closer: asyncio.Task[None] | None = None
    stdin_closed = False

    async def close_and_exit() -> None:
        # The MCP SDK reads stdin in a worker thread that no cancellation reaches, so a cancelled
        # server would wait for the client's next message. Close here and leave instead.
        await service.close()
        logger.info("%s stopped", settings.name)
        sys.stderr.flush()
        os._exit(0)

    def start_closing() -> None:
        nonlocal closer
        if closer is None and not stdin_closed:
            closer = loop.create_task(close_and_exit())

    with (
        hold_state_lock(service.store.state_dir, "stdio", settings.name),
        _stop_on_signals(lambda sig: loop.call_soon_threadsafe(start_closing)),
    ):
        try:
            await service.start(ingest_in_background=True)
            await create_server(service, settings).run_stdio_async()
        finally:
            if closer is not None:
                await closer  # it exits the process once the service is closed
            stdin_closed = True
            await service.close()
            logger.info("%s stopped", settings.name)
