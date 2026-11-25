# Changelog

## 0.1.6

- The graph's embedding model is set in `brain.toml`: `embedding_model`, `embedding_base_url` for any OpenAI-compatible server such as Ollama, `embedding_dimensions`, `embedding_query_instruction` and `embedding_api_key_env`. The default stays OpenAI's `text-embedding-3-small` at 1024 dimensions, so a vault that sets none of them keeps its vectors. The README's "A local embedding model" sets up Qwen3-Embedding-0.6B through Ollama. On two vaults' graphs it cut the median search from 266 and 255 ms to 32 and 65 ms, and found the facts behind a memory as often as OpenAI's model.
- Search embeds the query itself, with the query instruction in front, and hands the vector to Graphiti's hybrid search with the config Graphiti's own `search` uses. With no instruction the query and its results are the same as before.
- graph-state records which model made the graph's vectors, in a new `graph_meta` table. Migration: a graph from before 0.1.6 is recorded as `text-embedding-3-small` at 1024 dimensions when it first opens, since that is the only model older engines used. It re-embeds and ingests nothing.
- When `brain.toml` names another model, opening the graph re-embeds every entity name, fact and community name in the background. It makes no LLM calls and ingests nothing again. Until it's done, search takes its graph facts from the keyword index and writes to the graph wait. If a re-embed stops halfway, the next open does it again in full, and one that fails tries again every minute. The redis-server saves before graph-state records the new model, so a crash can't leave old vectors under it.
- Engines before 0.1.6 refuse `brain.toml` files with the new fields, so a vault on a local model can't run on an older tag by mistake, which would compare OpenAI query vectors with local ones.

## 0.1.5

- Search gets graph facts from Graphiti's hybrid search far more often. On two vaults' metrics, half of all searches, and nearly every first search after a pause of 5 minutes or more, missed the 0.5 s budget and answered from the keyword index alone. Three things caused it. httpx closed an idle connection to OpenAI after 5 s, so most searches paid for a new TLS handshake. The budget left no room for that handshake. And cancelling a late search closed the connection it had just opened. Now idle OpenAI connections stay open for 5 minutes, and Graphiti's LLM and embedder share them. The budget is 1.5 s, and a search past it still finishes in the background. A hybrid search that fails, usually on OpenAI, also falls back to the keyword index instead of returning no facts.
- `metrics.sqlite3` records whether each search's graph facts came from the keyword index alone (`keyword_only`), and the service's resident memory on every row (`rss_mb`). An older file gets the two columns when it opens, and its rows keep them empty. `brain-engine metrics` reports the share of keyword-only searches, the median and peak memory, and the most frequent errors of the last 7 days. A graph that had failed every call for days was visible only by reading the rows.
- `sync-graph`, `sync_memory_graph` and the start-up catch-up compare the index's content hashes with graph-state's, and read only the files of memories the graph lacks. They used to parse every Markdown file: on a vault of 860 memories a sync spent about 0.8 s doing that, and the cost grew with the vault. The counts they return are unchanged.
- Graphiti's FalkorDB driver logs each failed query with its parameters, including the query's embedding. Under the MCP SDK's log handler that took over a thousand lines per failure, and one vault's log reached 18 MB. Those messages are now cut to 1,000 characters.
- No migration: the state format is unchanged.

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
