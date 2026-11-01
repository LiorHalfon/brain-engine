"""Find and stop FalkorDB Lite redis-servers that an earlier service left running.

FalkorDB Lite runs the graph in a redis-server process of its own. A service that died without
stopping it, from `kill` or a closed terminal before the engine handled those, left it running.
Such a server keeps writing its in-memory graph over graphiti.db on its save timer and at logout,
and that graph can be older than the one on disk. The service owns the state lock while this runs,
so any server pointing at its graphiti.db belongs to no running service.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

REDISLITE_SERVER = "redislite/bin/redis-server"
# How long a server for this graph gets to say what it holds before start-up gives up on it.
QUERY_TIMEOUT_SECONDS = 5.0


class LeftoverServersError(RuntimeError):
    """A leftover server can't be stopped safely: several hold unsaved writes, or one doesn't answer."""


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


def connect(socket: str, timeout: float | None = None):
    from redis import Redis
    from redis.retry import Retry
    from redis.backoff import NoBackoff

    # No retries: a server that exits drops the connection, and retrying a gone socket takes seconds.
    return Redis(unix_socket_path=socket, retry=Retry(NoBackoff(), 0), socket_timeout=timeout)


def _works_in(process_cwd: str | None, folder: Path) -> bool:
    # samefile rather than comparing paths: macOS paths ignore case, /var is /private/var, and a
    # server whose folder was deleted reports an empty working directory.
    try:
        return bool(process_cwd) and os.path.samefile(process_cwd, folder)
    except OSError:
        return False


def find_servers(graph_db: Path) -> list[RedisServer]:
    """Every running FalkorDB Lite redis-server whose database file is `graph_db`.

    redis-server works in its database's folder, so its working directory picks out the servers for
    this graph without asking any other server anything. One of those that doesn't answer may still
    be saving, and starting another server beside it could let it overwrite the graph later, so that
    raises LeftoverServersError.
    """
    import psutil

    servers = []
    for process in psutil.process_iter(["pid", "cmdline", "cwd"]):
        cmdline = process.info["cmdline"] or []
        if not cmdline or REDISLITE_SERVER not in cmdline[0] or not _works_in(process.info["cwd"], graph_db.parent):
            continue
        pid = process.info["pid"]
        socket = _socket_path(process)
        if socket is None:
            raise LeftoverServersError(
                f"A redis-server (pid {pid}) works in {graph_db.parent} but its socket can't be found. "
                f"Check what it is with `ps -p {pid}`, stop it, then start again."
            )
        client = connect(socket, QUERY_TIMEOUT_SECONDS)
        try:
            filename = client.config_get("dbfilename")["dbfilename"]
            unsaved = int(client.info("persistence")["rdb_changes_since_last_save"])
        except Exception as exc:
            raise LeftoverServersError(
                f"A redis-server (pid {pid}) works in {graph_db.parent} but doesn't answer ({exc}). It may still "
                f"be saving the graph. Wait and start again, or inspect it with `redis-cli -s {socket} INFO persistence`."
            ) from None
        finally:
            client.close()
        if filename == graph_db.name:
            servers.append(RedisServer(pid=pid, socket=socket, unsaved_changes=unsaved))
    return servers


def _shutdown(server: RedisServer, *, save: bool) -> None:
    import psutil

    # No timeout: saving a large graph can take a while, and giving up halfway would let a new
    # server start beside one that is still writing the file.
    client = connect(server.socket)
    try:
        client.shutdown(save=save, nosave=not save)
    finally:
        client.close()
    # SHUTDOWN answers once the file is written; wait for the process too, so its socket and
    # pidfile are gone before a new server starts.
    try:
        psutil.Process(server.pid).wait(60)
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
