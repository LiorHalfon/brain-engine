# AGENTS.md

brain-engine is a memory server that vaults install by pinning a git tag. This repo is public. Read `README.md` for what the engine does and how vaults use it. This file is for agents changing the engine.

## Rules

- **No private content.** No memory text, business facts, prices, product or vendor names, people's names, emails, ids from a real vault, keys or a vault's instructions, anywhere: code, tests, fixtures, docs, commit messages. Test fixtures use invented neutral facts like "Acme ships orders with Parcelco". Vault-specific values belong in that vault's `brain.toml`.
- **Tests need no network and no API key.** Fake Graphiti where you can. Start a real FalkorDB Lite server (with `OPENAI_API_KEY=unused`, since opening the graph calls no API) only when the redis-server itself is under test. Run `just test` before every commit.
- **The memory rules below are part of the API.** A change that breaks one breaks every vault.
- **The state format is part of the API too.** A vault upgrading must not lose its index or graph, or re-ingest memories it already ingested: that costs OpenAI calls and can change the facts Graphiti extracts. Keep the `graph_ingestion` table, `normalize_content`, `content_sha256(type, body)` and Graphiti's default graph (no `group_id`) as they are, or write a migration and say so in `CHANGELOG.md`.

## Memory rules the code enforces

- Markdown is authoritative. FTS and Graphiti are derived and can be rebuilt from it.
- Duplicate detection hashes normalized `type + content` (sha256) before any file write or graph ingest. An exact duplicate is rejected.
- When reality changes, agents create a new memory. `edit_memory` fixes the stored text in place, clears verification and replaces the graph episode. `correct_memory` writes a new memory with `corrects:`, marks the old one `status: superseded` with `superseded_by:`, and keeps the old episode so Graphiti can model the history.
- Only a person confirms a memory. `verify_memory` writes `verified.by: human:<MEMORY_HUMAN_ID>`, and any edit clears it.
- Transcripts from `remember_session` are saved with `searchable: false` and never indexed.
- Search hides superseded memories, graph facts with `invalid_at`, and graph facts whose every source memory is superseded, unless `include_history=true`.

## Code map

- `config.py` (`Settings`) loads the vault's `brain.toml`, applies environment overrides (blank means unset) and engine defaults, and validates everything. `find_vault` picks the vault from `--vault`, `MEMORY_VAULT` or the current directory. `DEFAULT_KINDS` are the kinds the engine's own tools write; `DEFAULT_GRAPH_MODEL` pins Graphiti's models.
- `store.py` (`MemoryStore`) owns the Markdown vault and the FTS5 index (`index.sqlite3` in the state dir). Its `documents` table has one row per Markdown file with the mtime and size it was indexed at, plus the id, status and body hash. `ensure_index()` stats every file and re-parses only those whose stat changed, so a `git pull` or a hand edit is picked up on the next call. `get`, the duplicate check and search's superseded filter read the table instead of parsing the vault. `_load` trusts the table only after reading the file it points to, and refreshes the index on a miss. Bump `INDEX_SCHEMA` when the tables change; the old index is then dropped and rebuilt. The store's `kinds` map memory kinds to folders, and `dated_kinds` get a date prefix in the filename. `index.md` and `log.md` are reserved names and never indexed.
- `graph.py` has the `GraphBackend` protocol, with `GraphitiBackend` and `DisabledGraphBackend`. It keeps memory id to episode uuid, with content hashes, in `graph-state.sqlite3`. The hash comes from the file body, so hand edits and pulled memories are noticed. Re-ingesting a changed memory removes the old episode and adds a new one; ingests of one memory are serialized. `_create_client` stops leftover redis-servers, deletes redislite's stale registry file, opens FalkorDB Lite, starts its watcher, passes Graphiti the configured LLM models, and marks memories whose episode left the graph as `missing`. `search` gives Graphiti's hybrid search `SEARCH_BUDGET_SECONDS`, then falls back to `_keyword_search`, its BM25 half. `close()` sends the redis-server `SHUTDOWN SAVE` on a connection without retries.
- `watchdog.py` starts, for each opened graph, a detached `python -m brain_engine.watchdog` that saves and stops the redis-server once the opening process is gone, even after a SIGKILL of its whole process tree. It identifies the owner by pid and start time, and exits when the server's socket disappears.
- `leftovers.py` finds FalkorDB Lite redis-servers for one `graphiti.db` (working directory `samefile` the state folder, then `CONFIG GET dbfilename`, over the `unixsocket:` in their process title) and stops them: no unsaved changes, `NOSAVE`; exactly one with unsaved changes, `SAVE`; more than one, or one that doesn't answer, refuse. `MemoryService.start` lets that refusal stop the service instead of falling back to FTS. Running the tests sends read-only `CONFIG GET` and `INFO` only to servers working in the tests' own folders.
- `service.py` (`MemoryService`) coordinates the store and the graph. It always writes to the store first. Graph failures come back in the result (`graph_error`, `{"indexed": False, "error": ...}`) and search falls back to FTS. `start(ingest_in_background=True)`, which both servers use, starts one worker (`_ingest_queued`) and queues what the graph is missing; writes then queue their memory instead of awaiting the ingest. `close()` cancels the worker before closing the graph. `graph_idle()` waits for the queue to drain, for tests.
- `server.py` is the MCP layer. `create_server(service, settings)` registers the tools under the vault's `name` and `instructions`; write tools exist only with `MEMORY_ACCESS=read-write`. `/graph` and `/graph.json` are plain HTTP routes for the viewer (`graph_view.html`) that refuse non-localhost `Host` headers. `serve_http` takes the state lock, opens the service, then runs uvicorn through `_Server`, which ends the SDK's `ListenHandler` streams at the start of shutdown. `_stop_on_signals` handles SIGINT, SIGTERM and SIGHUP for the service's whole life, leaving alone any it inherited as ignored: during start-up a signal cancels the start, while serving it triggers uvicorn's graceful shutdown. `serve_stdio` closes the service on the first signal and exits with `os._exit`, because the SDK reads stdin in a thread no cancellation reaches. uvicorn's own signal handling is switched off, since it re-raised SIGTERM before the service could close.
- `metrics.py` (`Metrics`) records each operation's timing in `metrics.sqlite3` and builds the `metrics` report. `MemoryService` wraps every public method in `self._measure(op)` and every graph await in `timing.graph()`.
- `lock.py` keeps FalkorDB Lite single-process. Whoever opens the graph holds `service.lock` and writes its owner (the service URL) into it. `cli.py`'s `search` and `sync-graph` call the running service over MCP when the lock holds a URL.

Tests inject `FakeGraph` (in `tests/test_service.py`) in place of Graphiti. `tests/test_http.py` runs the real HTTP service on a free port through `running_service()` and talks to it with `mcp.Client`. `tests/test_signals.py` and `tests/test_leftovers.py` start real processes. `tests/conftest.py` has `make_vault`, which writes a `brain.toml`.

## Releasing

1. Add the version's entry to `CHANGELOG.md` and bump `version` in `pyproject.toml`.
2. Commit, push, and wait for the test workflow to pass on `main`.
3. `git tag vX.Y.Z && git push origin vX.Y.Z`. Never move a tag that vaults may have pinned; release a new one.
