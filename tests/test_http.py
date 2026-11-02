from __future__ import annotations

import asyncio
import json
import socket
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path

import anyio
import httpx2
import pytest
from mcp import Client

from brain_engine import cli
from brain_engine.cli import _run_graph_command
from brain_engine.config import Settings
from brain_engine.graph import GraphBackend, GraphitiBackend
from brain_engine.lock import GraphOwnedElsewhere
from brain_engine.server import PortInUse, serve_http
from brain_engine.service import MemoryService
from brain_engine.store import MemoryStore
from conftest import make_vault
from test_service import FakeGraph

ALL_TOOLS = {
    "search_memory",
    "get_memory",
    "remember",
    "remember_session",
    "edit_memory",
    "correct_memory",
    "verify_memory",
    "sync_memory_graph",
}


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _settings(tmp_path: Path, access: str = "read-write") -> Settings:
    environ = {"MEMORY_ACCESS": access, "MEMORY_ACTOR": "test/http", "MEMORY_HUMAN_ID": "tester", "MEMORY_PORT": str(_free_port())}
    return Settings.load(make_vault(tmp_path), environ=environ)


def _store(settings: Settings) -> MemoryStore:
    return MemoryStore(settings.memory_dir, actor=settings.actor, human_id=settings.human_id, state_dir=settings.state_dir)


@asynccontextmanager
async def running_service(
    settings: Settings, graph: GraphBackend | None = None
) -> AsyncIterator[MemoryService]:
    store = _store(settings)
    service = MemoryService(store, graph if graph is not None else FakeGraph())
    stop = anyio.Event()
    task = asyncio.create_task(serve_http(settings, service, stop=stop))
    for _ in range(200):
        if task.done():
            task.result()
        try:
            _, writer = await asyncio.open_connection(settings.host, settings.port)
            writer.close()
            break
        except OSError:
            await asyncio.sleep(0.025)
    else:
        raise TimeoutError("service did not start")
    try:
        yield service
    finally:
        stop.set()
        await task


async def call(client: Client, tool: str, arguments: dict) -> dict:
    result = await client.call_tool(tool, arguments)
    text = "".join(block.text for block in result.content)
    assert not result.is_error, text
    return json.loads(text)


async def test_http_service_starts_and_exposes_memory_tools(tmp_path: Path) -> None:
    graph = FakeGraph()
    settings = _settings(tmp_path)
    async with running_service(settings, graph):
        async with Client(settings.url) as client:
            tools = {tool.name for tool in (await client.list_tools()).tools}
    assert tools == ALL_TOOLS
    assert graph.started == 1


async def test_read_only_service_registers_no_write_tools(tmp_path: Path) -> None:
    settings = _settings(tmp_path, access="read-only")
    async with running_service(settings):
        async with Client(settings.url) as client:
            tools = {tool.name for tool in (await client.list_tools()).tools}
    assert tools == {"search_memory", "get_memory"}


async def test_read_tools_over_http(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    async with running_service(settings) as service:
        created = service.store.remember(title="Delivery", content="Parcelco delivers Acme orders within 3 days.", kind="note")
        async with Client(settings.url) as client:
            found = await call(client, "search_memory", {"query": "delivers"})
            memory = await call(client, "get_memory", {"memory_id": created["id"]})
    assert [m["id"] for m in found["memories"]] == [created["id"]]
    assert "delivers" in memory["content"]


async def test_write_tool_over_http_writes_markdown(tmp_path: Path) -> None:
    graph = FakeGraph()
    settings = _settings(tmp_path)
    async with running_service(settings, graph) as service:
        async with Client(settings.url) as client:
            created = await call(
                client, "remember", {"title": "Order", "content": "Acme ordered ten crates.", "kind": "note"}
            )
            duplicate = await call(
                client, "remember", {"title": "Order again", "content": "Acme ordered ten crates.", "kind": "note"}
            )
        await service.graph_idle()
    assert (settings.memory_dir / created["path"]).read_text().count("Acme ordered ten crates.") == 1
    assert created["graph"] == {"queued": True, "memory_id": created["id"]}
    assert duplicate["duplicate"] is True
    assert graph.ingested == [created["id"]]


async def test_multiple_clients_share_one_service(tmp_path: Path) -> None:
    graph = FakeGraph()
    settings = _settings(tmp_path)
    async with running_service(settings, graph):
        async with Client(settings.url) as writer, Client(settings.url) as reader:
            created = await call(writer, "remember", {"title": "Office", "content": "Acme's office is in Springfield."})
            results = await asyncio.gather(
                call(reader, "search_memory", {"query": "Springfield"}),
                call(writer, "search_memory", {"query": "Springfield"}),
            )
    for found in results:
        assert [m["id"] for m in found["memories"]] == [created["id"]]
    assert graph.started == 1


class _FakeGraphitiClient:
    async def search(self, query: str) -> list:
        return []

    async def close(self) -> None:
        return None


async def test_graphiti_client_initialized_once_across_clients(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path)
    graph = GraphitiBackend(settings.state_dir, enabled=True)
    created = 0

    async def fake_create_client() -> _FakeGraphitiClient:
        nonlocal created
        created += 1
        await asyncio.sleep(0.01)
        return _FakeGraphitiClient()

    monkeypatch.setattr(graph, "_create_client", fake_create_client)
    async with running_service(settings, graph):
        async with Client(settings.url) as a, Client(settings.url) as b, Client(settings.url) as c:
            results = await asyncio.gather(
                *(call(client, "search_memory", {"query": "anything"}) for client in (a, b, c))
            )
    assert created == 1
    assert all(result["graph_error"] is None for result in results)


async def test_graphiti_disabled_falls_back_to_markdown_and_fts(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    graph = GraphitiBackend(settings.state_dir, enabled=False)
    async with running_service(settings, graph):
        async with Client(settings.url) as client:
            created = await call(client, "remember", {"title": "Mascot", "content": "Acme's mascot is a heron named Pip."})
            found = await call(client, "search_memory", {"query": "heron"})
    assert created["graph"]["indexed"] is False
    assert found["facts"] == []
    assert found["graph_error"] is None
    assert [m["id"] for m in found["memories"]] == [created["id"]]


async def test_second_service_on_same_state_is_refused(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    async with running_service(settings):
        other = replace(settings, port=_free_port())
        with pytest.raises(GraphOwnedElsewhere) as exc:
            await serve_http(other, MemoryService(_store(other), FakeGraph()))
    assert exc.value.owner == settings.url


async def test_cli_graph_commands_use_the_running_service(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    async with running_service(settings) as service:
        created = service.store.remember(title="Snacks", content="The team prefers oat crackers.")
        found = await _run_graph_command(
            settings, "search_memory", {"query": "oat crackers", "limit": 5, "include_history": False}
        )
    assert [m["id"] for m in found["memories"]] == [created["id"]]


async def test_cli_sync_graph_removes_deleted_memories_through_the_running_service(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    graph = FakeGraph()
    async with running_service(settings, graph) as service:
        service.store.remember(title="Snacks", content="The team prefers oat crackers.")
        deleted = service.store.remember(title="Drinks", content="The team prefers mint tea.")
        await _run_graph_command(settings, "sync_memory_graph", {})
        (settings.memory_dir / deleted["path"]).unlink()
        synced = await _run_graph_command(settings, "sync_memory_graph", {})
    assert synced["removed"] == 1 and synced["errors"] == []
    assert deleted["id"] not in graph.memory_ids()


async def test_cli_sync_graph_passes_the_go_ahead_to_the_running_service(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    graph = FakeGraph()
    async with running_service(settings, graph) as service:
        created = service.store.remember(title="Drinks", content="The team prefers mint tea.")
        await _run_graph_command(settings, "sync_memory_graph", {})
        (settings.memory_dir / created["path"]).unlink()
        held_back = await _run_graph_command(settings, "sync_memory_graph", {})
        gone_ahead = await _run_graph_command(settings, "sync_memory_graph", {"allow_mass_removal": True})
    assert held_back["removed"] == 0 and gone_ahead["removed"] == 1


async def test_cli_sync_graph_passes_the_go_ahead_when_no_service_runs(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path)
    store = _store(settings)
    graph = FakeGraph()
    created = store.remember(title="Drinks", content="The team prefers mint tea.")
    await graph.ingest(store.record(created["id"]))
    (settings.memory_dir / created["path"]).unlink()
    monkeypatch.setattr(MemoryService, "from_settings", staticmethod(lambda settings: MemoryService(store, graph)))

    synced = await _run_graph_command(settings, "sync_memory_graph", {"allow_mass_removal": True})

    assert synced["removed"] == 1


@pytest.mark.parametrize(
    ("flags", "arguments"),
    [([], {}), (["--allow-mass-removal"], {"allow_mass_removal": True})],
)
def test_sync_graph_sends_the_go_ahead_only_when_given(tmp_path: Path, monkeypatch, flags, arguments) -> None:
    # A service still on an older engine has no such argument, so a plain sync must not send it.
    calls = []

    async def recorded(settings: Settings, tool: str, sent: dict) -> dict:
        calls.append((tool, sent))
        return {}

    monkeypatch.setattr(cli, "_run_graph_command", recorded)
    monkeypatch.setattr(sys, "argv", ["brain-engine", "sync-graph", "--vault", str(make_vault(tmp_path)), *flags])

    cli.main()

    assert calls == [("sync_memory_graph", arguments)]


async def test_port_in_use_fails_clearly_without_starting_graph(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    graph = FakeGraph()
    with socket.socket() as occupant:
        occupant.bind((settings.host, settings.port))
        occupant.listen()
        with pytest.raises(PortInUse, match=str(settings.port)):
            await serve_http(settings, MemoryService(_store(settings), graph))
    assert graph.started == 0


async def test_graph_page_and_data_are_served_locally(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    async with running_service(settings) as service:
        async with Client(settings.url) as client:
            old = await call(client, "remember", {"title": "Delivery time", "content": "Parcelco delivers in 5 days.", "kind": "note"})
            await call(client, "correct_memory", {"memory_id": old["id"], "correction": "Parcelco delivers in 3 days.", "reason": "terms updated"})
        base = f"http://{settings.host}:{settings.port}"
        async with httpx2.AsyncClient() as http:
            page = await http.get(f"{base}/graph")
            data = (await http.get(f"{base}/graph.json")).json()
    assert page.status_code == 200 and "Memory Graph" in page.text
    assert data["error"] is None and {n["name"] for n in data["nodes"]} == {"Acme", "Parcelco"}
    by_source = {e["source_memory_ids"][0]: e for e in data["edges"]}
    # The superseded source's fact is historical, like in search; the correction's fact is current.
    assert by_source[old["id"]]["current"] is False
    assert [e["current"] for mid, e in by_source.items() if mid != old["id"]] == [True]
    assert data["memories"][old["id"]]["status"] == "superseded"


async def test_graph_routes_reject_non_local_host_header(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    async with running_service(settings):
        async with httpx2.AsyncClient() as http:
            for path in ("/graph", "/graph.json"):
                response = await http.get(
                    f"http://{settings.host}:{settings.port}{path}", headers={"Host": "evil.example"}
                )
                assert response.status_code == 403


async def test_graph_data_reports_disabled_and_failing_graph(tmp_path: Path) -> None:
    disabled = _settings(tmp_path / "disabled")
    async with running_service(disabled, GraphitiBackend(disabled.state_dir, enabled=False)):
        async with httpx2.AsyncClient() as http:
            data = (await http.get(f"http://{disabled.host}:{disabled.port}/graph.json")).json()
    assert data["enabled"] is False and data["nodes"] == []

    failing_graph = FakeGraph()
    failing_graph.fail_search = True
    failing = _settings(tmp_path / "failing")
    async with running_service(failing, failing_graph):
        async with httpx2.AsyncClient() as http:
            response = await http.get(f"http://{failing.host}:{failing.port}/graph.json")
    assert response.status_code == 200 and response.json()["error"] == "graph unavailable"


class _FakeDriver:
    async def execute_query(self, query: str, **_: object) -> tuple:
        if "MATCH (n:Entity)" in query:
            return [{"uuid": "n1", "name": "Acme", "summary": None}, {"uuid": "n2", "name": "Parcelco", "summary": "A parcel carrier"}], [], None
        return [
            {"uuid": "e1", "source": "n1", "target": "n2", "name": "SHIPS_WITH", "fact": "Acme ships orders with Parcelco.",
             "valid_at": "2026-09-01T00:00:00+00:00", "invalid_at": None, "episodes": ["ep-1", "ep-unknown"]}
        ], [], None


async def test_graphiti_snapshot_maps_episodes_to_memory_ids(tmp_path: Path, monkeypatch) -> None:
    graph = GraphitiBackend(tmp_path / ".brain", enabled=True)
    graph._write_state("mem_1", "hash", episode_uuid="ep-1", status="ok")
    fake_client = _FakeGraphitiClient()
    fake_client.driver = _FakeDriver()

    async def fake_create_client() -> _FakeGraphitiClient:
        return fake_client

    monkeypatch.setattr(graph, "_create_client", fake_create_client)
    snapshot = await graph.snapshot()
    assert snapshot["nodes"][0] == {"uuid": "n1", "name": "Acme", "summary": ""}
    assert snapshot["edges"][0]["source_memory_ids"] == ["mem_1"]
    assert snapshot["edges"][0]["valid_at"] == "2026-09-01T00:00:00+00:00"


async def test_shutdown_does_not_wait_for_open_client_streams(tmp_path: Path) -> None:
    # MCP clients keep a subscriptions/listen request open for the whole session. Graceful
    # shutdown used to wait on it forever, so Ctrl+C appeared to do nothing.
    settings = _settings(tmp_path)
    stop = anyio.Event()
    service = MemoryService(_store(settings), FakeGraph())
    server = asyncio.create_task(serve_http(settings, service, stop=stop))

    async def hold_listen() -> None:
        async with Client(settings.url) as client:
            async with client.listen(tools_list_changed=True):
                await asyncio.sleep(3600)

    for _ in range(200):
        try:
            _, writer = await asyncio.open_connection(settings.host, settings.port)
            writer.close()
            break
        except OSError:
            await asyncio.sleep(0.025)
    holder = asyncio.create_task(hold_listen())
    await asyncio.sleep(1)
    stop.set()
    done, _ = await asyncio.wait({server}, timeout=8)
    holder.cancel()
    assert server in done, "service did not stop while a client stream was open"


async def test_shutdown_ends_listen_streams_gracefully(tmp_path: Path) -> None:
    # Ending each listen stream first leaves uvicorn nothing to cancel after its graceful timeout,
    # so there is no traceback per connected client, and clients get the final frame that tells
    # them the stream ended on purpose.
    settings = _settings(tmp_path)
    stop = anyio.Event()
    service = MemoryService(_store(settings), FakeGraph())
    server = asyncio.create_task(serve_http(settings, service, stop=stop))
    opened = asyncio.Event()

    async def listen_until_closed() -> str:
        async with Client(settings.url) as client:
            async with client.listen(tools_list_changed=True) as subscription:
                opened.set()
                async for _ in subscription:
                    pass
        return "ended gracefully"

    for _ in range(200):
        try:
            _, writer = await asyncio.open_connection(settings.host, settings.port)
            writer.close()
            break
        except OSError:
            await asyncio.sleep(0.025)
    listener = asyncio.create_task(listen_until_closed())
    try:
        await asyncio.wait_for(opened.wait(), 5)
        stop.set()
        done, _ = await asyncio.wait({server}, timeout=2)  # under uvicorn's 3s graceful timeout
        assert server in done, "shutdown waited for the listen stream instead of ending it"
        assert await asyncio.wait_for(listener, 2) == "ended gracefully"
    finally:
        stop.set()
        listener.cancel()
        await asyncio.wait({server, listener}, timeout=8)


def test_ctrl_c_stops_serve_without_a_traceback(tmp_path: Path, monkeypatch) -> None:
    # uvicorn re-raises the Ctrl+C it handled once its graceful shutdown is done.
    async def interrupted(settings: Settings) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "serve_http", interrupted)
    monkeypatch.setattr(sys, "argv", ["brain-engine", "serve", "--vault", str(make_vault(tmp_path))])

    try:
        cli.main()
    except KeyboardInterrupt:
        pytest.fail("Ctrl+C escaped `serve` as a KeyboardInterrupt traceback")
