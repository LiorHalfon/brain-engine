"""Find and stop FalkorDB Lite redis-servers that an earlier service left running.

FalkorDB Lite runs the graph in a redis-server process of its own. A service that died without
stopping it, from `kill` or a closed terminal before the engine handled those, left it running.
Such a server keeps writing its in-memory graph over graphiti.db on its save timer and at logout,
and that graph can be older than the one on disk. The service owns the state lock while this runs,
so any server pointing at its graphiti.db belongs to no running service.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

REDISLITE_SERVER = "redislite/bin/redis-server"


class LeftoverServersError(RuntimeError):
    """More than one leftover server holds unsaved writes, so which one is newest is unknown."""


@dataclass
class RedisServer:
    pid: int
    socket: str
    unsaved_changes: int


def _socket_path(process) -> str | None:
    # redis-server rewrites its process title to "<path>/redis-server unixsocket:<socket>".
    match = re.search(r"unixsocket:(\S+)", " ".join(process.info["cmdline"] or []))
    if match:
        return match.group(1)
    try:
        sockets = [conn.laddr for conn in process.net_connections(kind="unix") if conn.laddr]
    except Exception:
        return None
    return sockets[0] if sockets else None


def connect(socket: str):
    from redis import Redis
    from redis.retry import Retry
    from redis.backoff import NoBackoff

    # No retries: a server that exits drops the connection, and retrying a gone socket takes seconds.
    return Redis(unix_socket_path=socket, retry=Retry(NoBackoff(), 0), socket_timeout=5)


def find_servers(graph_db: Path) -> list[RedisServer]:
    """Every running FalkorDB Lite redis-server whose database file is `graph_db`."""
    import psutil

    target = graph_db.resolve()
    servers = []
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        cmdline = process.info["cmdline"] or []
        if not cmdline or REDISLITE_SERVER not in cmdline[0]:
            continue
        socket = _socket_path(process)
        if socket is None:
            continue
        client = connect(socket)
        try:
            directory = client.config_get("dir")["dir"]
            filename = client.config_get("dbfilename")["dbfilename"]
            if (Path(directory) / filename).resolve() != target:
                continue
            unsaved = int(client.info("persistence")["rdb_changes_since_last_save"])
        except Exception as exc:
            logger.warning("Could not ask redis-server %s (%s) which graph it holds: %s", process.info["pid"], socket, exc)
            continue
        finally:
            client.close()
        servers.append(RedisServer(pid=process.info["pid"], socket=socket, unsaved_changes=unsaved))
    return servers


def _shutdown(server: RedisServer, *, save: bool) -> None:
    import psutil

    client = connect(server.socket)
    try:
        client.shutdown(save=save, nosave=not save)
    finally:
        client.close()
    # SHUTDOWN answers once the file is written; wait for the process too, so its socket and
    # pidfile are gone before a new server starts.
    try:
        psutil.Process(server.pid).wait(10)
    except psutil.NoSuchProcess:
        pass


def stop_leftover_servers(graph_db: Path) -> list[RedisServer]:
    """Stop every leftover server for `graph_db`, keeping the one with unsaved writes if there is one.

    With no unsaved writes the file on disk is at least as new as the server's memory, so each is
    stopped without saving. One server with unsaved writes holds a dead service's newest graph, so
    it saves first. More than one means guessing which is newest; refuse instead.
    """
    servers = find_servers(graph_db)
    unsaved = [server for server in servers if server.unsaved_changes]
    if len(unsaved) > 1:
        listed = "\n".join(
            f"  pid {server.pid}: {server.unsaved_changes} unsaved changes; inspect with "
            f"`redis-cli -s {server.socket} INFO persistence`"
            for server in unsaved
        )
        raise LeftoverServersError(
            f"{len(unsaved)} leftover redis-servers hold unsaved writes to {graph_db}:\n{listed}\n"
            "Decide which to keep, `SAVE` it and stop the others with `SHUTDOWN NOSAVE`, then start again."
        )
    # The one with unsaved writes goes last, so its save is the write that stays on disk.
    for server in sorted(servers, key=lambda server: server.unsaved_changes):
        _shutdown(server, save=bool(server.unsaved_changes))
        logger.warning(
            "Stopped a leftover graph redis-server (pid %s, %s)",
            server.pid,
            f"saved its {server.unsaved_changes} unsaved changes" if server.unsaved_changes else "nothing unsaved",
        )
    return servers
