from __future__ import annotations

import argparse
import asyncio
import json
import os
from typing import Any

from brain_engine.config import ConfigError, Settings, find_vault
from brain_engine.lock import GraphOwnedElsewhere, hold_state_lock
from brain_engine.server import PortInUse, serve_http, serve_stdio
from brain_engine.service import MemoryService


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


async def _call_service(url: str, tool: str, arguments: dict[str, Any]) -> Any:
    from mcp import Client

    async with Client(url) as client:
        result = await client.call_tool(tool, arguments)
    text = "".join(getattr(block, "text", "") for block in result.content)
    if result.is_error:
        raise SystemExit(f"{tool} failed: {text}")
    return json.loads(text)


async def _run_graph_command(settings: Settings, tool: str, arguments: dict[str, Any]) -> Any:
    """Run a Graphiti-touching command without opening a second FalkorDB Lite instance.

    If the service is running, the command goes through its MCP endpoint. Otherwise it runs
    locally while holding the same state lock, so the service cannot start mid-command.
    """
    service = MemoryService.from_settings(settings)
    try:
        with hold_state_lock(service.store.state_dir, "cli", settings.name):
            await service.start()
            try:
                if tool == "sync_memory_graph":
                    return await service.sync_graph()
                return await service.search(**arguments)
            finally:
                await service.close()
    except GraphOwnedElsewhere as exc:
        if not exc.owner.startswith("http"):
            raise SystemExit(f"{exc}. Stop it and retry.") from None
        return await _call_service(exc.owner, tool, arguments)


def main() -> None:
    vault_arg = argparse.ArgumentParser(add_help=False)
    vault_arg.add_argument(
        "--vault", help="the vault folder, which holds brain.toml (default: $MEMORY_VAULT, then the current directory)"
    )
    parser = argparse.ArgumentParser(prog="brain-engine")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("serve", parents=[vault_arg], help="run the long-lived Streamable HTTP MCP service")
    sub.add_parser("stdio", parents=[vault_arg], help="legacy: run a single-client MCP server over stdio")
    sub.add_parser("reindex", parents=[vault_arg], help="rebuild the FTS5 index from Markdown")
    sub.add_parser("sync-graph", parents=[vault_arg], help="ingest new/edited Markdown into Graphiti")
    search = sub.add_parser("search", parents=[vault_arg], help="search memory, through the service if it runs")
    search.add_argument("query")
    search.add_argument("--limit", type=int, default=5)
    search.add_argument("--include-history", action="store_true")
    metrics = sub.add_parser(
        "metrics", parents=[vault_arg], help="report how long memory operations have taken on this machine"
    )
    metrics.add_argument("--days", type=int, default=30)
    args = parser.parse_args()

    try:
        settings = Settings.load(find_vault(args.vault, os.environ))
    except ConfigError as exc:
        raise SystemExit(str(exc)) from None
    try:
        if args.command in ("serve", "stdio"):
            try:
                asyncio.run(serve_http(settings) if args.command == "serve" else serve_stdio(settings))
            except (KeyboardInterrupt, asyncio.CancelledError):
                # A signal that arrived before the server was up cancels start-up; the service
                # has closed by the time it gets here.
                pass
        elif args.command == "reindex":
            # FTS only; safe to run while the service is up.
            print(MemoryService.from_settings(settings).store.reindex())
        elif args.command == "metrics":
            # Reads the service's timings file; safe to run while the service is up.
            print(MemoryService.from_settings(settings).metrics.report(args.days))
        elif args.command == "sync-graph":
            _print(asyncio.run(_run_graph_command(settings, "sync_memory_graph", {})))
        elif args.command == "search":
            arguments = {"query": args.query, "limit": args.limit, "include_history": args.include_history}
            _print(asyncio.run(_run_graph_command(settings, "search_memory", arguments)))
    except GraphOwnedElsewhere as exc:
        raise SystemExit(f"{exc}. Only one {settings.name} service may run at a time.") from None
    except PortInUse as exc:
        raise SystemExit(str(exc)) from None


if __name__ == "__main__":
    main()
