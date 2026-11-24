# Changelog

## 0.1.4

- On macOS the graph's watcher no longer stops the redis-server of a service that is still running. It identified the service by its start time from psutil. psutil 7.2 shifts `create_time()` by a second whenever the system boot time crosses a whole second, and macOS's clock corrections cause that a few times a day. The watcher then took the service for dead and sent its server `SHUTDOWN SAVE`. Every graph call after that failed with `Error 2 connecting to <socket>. No such file or directory.` until the service restarted. The watcher now waits on a pipe whose write end the service holds open. The kernel closes that end however the service exits, so neither clock changes nor reused pids matter.
- When the watcher stops a server, or fails to, it says so in `watchdog.log` in the state folder. When the service's stderr is a file, as a launchd agent's log is, the same line goes there too.
- A service whose graph server is gone opens the graph again on the next graph call, instead of failing every call until a restart. That covers a server that was stopped and one that crashed: a FalkorDB Lite redis-server can segfault, and the socket it leaves behind refuses connections. The service logs "Opening the graph again on a new one" with the last line of the server's log. It stops any server left for the graph and starts a new one. Then it queues every memory, so the ingests that failed meanwhile and the episodes the old server never saved go in again.
- A service started on an older tag keeps its old watcher until it restarts. No migration: the state format is unchanged.

## 0.1.3

- Every SQLite connection is closed as soon as its transaction ends. `with sqlite3.connect(path)` only commits, so each one kept its file descriptor until Python's cyclic garbage collector ran, and a sync opens one per memory. On a vault of about 400 memories a service spiked to about 250 open files on every sync and held up to 200 while idle. Under launchd's default limit of 256 that failed with `Too many open files`, or `unable to open database file`. The same load now peaks at about 55. No migration: the state format is unchanged.

## 0.1.2

- `sync-graph` and `sync_memory_graph` remove the graph episode of every memory whose Markdown file is gone, and count them as `removed` next to `indexed`. Graphiti deletes the facts and entities only that memory produced. The first sync after upgrading also removes memories deleted before it. No migration: the state format is unchanged.
- A sync keeps the graph, and reports it as one `errors` entry with no `memory_id`, when a Markdown file doesn't parse, when the folder holds no memories, or when more than 20 memories look deleted at once. `sync-graph --allow-mass-removal`, or `allow_mass_removal=true` on the tool, goes past the last two.
- `search_memory` and `/graph` no longer treat a fact as current once its every source memory is deleted, even before a sync removes its episode.
- A YAML error in one file's frontmatter no longer stops every call that reads the index. The file now counts as one that doesn't parse.
- Graph writes run one at a time, so a removal or a sync's ingest can't overlap a background ingest and bring back or lose facts. `sync-graph` waits its turn while the queue is busy.

## 0.1.1

- `skills/brain-engine/` is a skill for the agents that set up, move, upgrade or troubleshoot a vault. The README's "Agent skill" section shows how a vault installs it.
- The leftover-servers error says to stop the servers you don't keep with `SHUTDOWN NOSAVE` and start again, so the engine saves the one left. It used to say to `SAVE` the kept one first, which let another server's timed save overwrite it.

## 0.1.0

The first release, extracted from a vault's own memory server.

- Each vault configures the engine in `brain.toml`: MCP server name and instructions, port, Markdown and state folders, memory kinds and dated kinds, Graphiti models. Environment variables override it, and blank ones count as unset.
- The CLI is `brain-engine`, and every command takes `--vault`, falling back to `MEMORY_VAULT` and then the current directory.
- `MEMORY_ROOT` now moves only the Markdown folder. The state folder stays relative to the vault.
- Graphiti's LLM models are set explicitly and default to `gpt-6-luna`, graphiti-core 0.31.2's own default.
- SIGTERM and SIGHUP stop the service the way Ctrl-C does, closing the graph and its redis-server.
- A detached watcher saves and stops the graph's redis-server when the service is killed outright, for example by closing a tab in the Claude desktop app's terminal panel.
- On start the engine stops FalkorDB Lite redis-servers an earlier service left running for its graph, keeping the one with unsaved writes, and refuses to start if more than one has them.
- Memories marked ingested whose episode is missing from the graph are ingested again.
