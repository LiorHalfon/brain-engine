# brain-engine

A memory server for AI agents. A vault is a folder of Markdown memories with YAML frontmatter. brain-engine indexes it with SQLite FTS5, builds a temporal knowledge graph of it with [Graphiti](https://github.com/getzep/graphiti), and serves both to agents over MCP.

```text
                 Markdown files, committed to git
                       SOURCE OF TRUTH
                             │
              ┌──────────────┴──────────────┐
              │                             │
            FTS5                        Graphiti
     document retrieval        entities, facts and time
              │                             │
              └──────────────┬──────────────┘
                             │
             one long-lived local MCP service
           (Streamable HTTP, 127.0.0.1:<port>/mcp)
                             │
       ┌─────────────┬───────┴──────┬──────────────┐
  Claude Code      Cursor       Gemini CLI    ChatGPT (later)
```

Every client connects to the same running service. It is the only process that opens Graphiti's embedded FalkorDB Lite database. Graphiti and SQLite are indexes you can throw away and rebuild. The Markdown is what counts.

It runs on macOS and Linux with Python 3.12 or later.

A vault holds no server code. It has a `brain.toml`, its `memory/` folder, a `pyproject.toml` that pins an engine tag, and whatever scripts it wants. A fix lands here once and reaches a vault when that vault bumps its pin.

## Vault layout

```text
my-vault/
├── brain.toml              # name, instructions, port, kinds; see below
├── pyproject.toml          # pins brain-engine
├── memory/                 # the Markdown, committed to git
│   ├── notes/
│   ├── sessions/
│   │   └── transcripts/
│   └── <one folder per kind you add>
└── .brain/                 # derived, per machine, ignored by git
    ├── index.sqlite3       # FTS5, plus one row per file (path, stat, id, status, hash)
    ├── graphiti.db         # embedded FalkorDB Lite
    ├── graph-state.sqlite3 # memory id <-> Graphiti episode, with content hashes
    ├── metrics.sqlite3     # how long each operation took on this machine
    └── service.lock        # held by the one process that owns the graph
```

Add the state folder to the vault's `.gitignore`. `examples/vault/` has a starting point.

## brain.toml

```toml
name = "my-memory"
instructions = """
My long-term memory, shared by every agent I work with. Call search_memory first. ...
"""
port = 8766
state_dir = ".brain"
dated_kinds = ["decision", "session", "transcript"]

[kinds]
decision = "decisions"
person = "people"
project = "projects"
```

| Field | Default | Meaning |
| --- | --- | --- |
| `name` | required | The MCP server name agents see. Also used in the state-lock message, CLI messages, each graph episode's source (`<name>:<path>`) and the default actor (`<name>/local`). |
| `instructions` | required | The text the MCP server sends agents when they connect. The engine ships none. |
| `port` | `8766` | Service port. |
| `memory_dir` | `memory` | The Markdown folder, relative to the vault. |
| `state_dir` | `.brain` | Index, graph, lock and metrics, relative to the vault. |
| `kinds` | see below | Memory kind to folder under `memory_dir`. `remember` refuses kinds not listed. |
| `dated_kinds` | `[]` | Kinds whose new filenames start with the date, for memories that record an event. |
| `graph_model`, `graph_small_model` | `gpt-6-luna` | The OpenAI models Graphiti uses to extract entities and facts. |

`remember` defaults to kind `note`, and `remember_session` writes `session` and `transcript`. If `kinds` leaves one of those out, the engine adds it: `note` to `notes/`, `session` to `sessions/`, `transcript` to `sessions/transcripts/`. Map them yourself to use other folders. A misspelled field, a dated kind with no folder or a missing `name` stops the engine with an error naming it.

The graph models are set explicitly so that a graphiti-core upgrade can't switch them without anyone noticing. graphiti-core 0.30.2 defaulted to `gpt-4.1-nano`, and once OpenAI retired it every ingest failed with `model_not_found`. If you pick another model, check that the pinned graphiti-core sends it valid parameters. Some versions sent `reasoning.effort: minimal` to models that reject it.

## Environment variables

The environment overrides `brain.toml`, which overrides the engine defaults. A blank value counts as unset. The engine reads only the process environment and never loads `.env` itself. Let your task runner do that, with `set dotenv-load` in a `justfile` for example.

| Variable | Default | Meaning |
| --- | --- | --- |
| `MEMORY_VAULT` | current directory | The vault folder, when `--vault` isn't given. |
| `MEMORY_ROOT` | `<vault>/<memory_dir>` | The Markdown folder. It moves only the Markdown; the state stays in the vault. |
| `MEMORY_HOST` | `127.0.0.1` | Bind address. Keep it on localhost, since the service has no authentication. |
| `MEMORY_PORT` | `port` | Service port. |
| `MEMORY_ACCESS` | `read-write` | `read-only` serves only `search_memory` and `get_memory`. |
| `MEMORY_ACTOR` | `<name>/local` | Written to each new memory's `generated.by`. |
| `MEMORY_HUMAN_ID` | `owner` | Written to `verified.by` as `human:<id>` when a memory is verified. |
| `MEMORY_GRAPH_ENABLED` | `true` | `false` serves Markdown and FTS only. |
| `MEMORY_GRAPH_MODEL`, `MEMORY_GRAPH_SMALL_MODEL` | `graph_model`, `graph_small_model` | Graphiti's models. |
| `OPENAI_API_KEY` | unset | Graphiti's LLM and embedding calls. Without it the service logs a warning and serves Markdown and FTS. |

Access mode and actor belong to the service, so every connected client shares them.

## Installing in a vault

brain-engine isn't on PyPI. A vault installs a tag straight from GitHub with [uv](https://docs.astral.sh/uv/):

```toml
[project]
name = "my-vault"
version = "0"
requires-python = ">=3.12"
dependencies = ["brain-engine[graph] @ git+https://github.com/LiorHalfon/brain-engine@v0.1.0"]

[tool.uv]
package = false
```

```bash
uv sync
uv run brain-engine serve
```

The `graph` extra brings Graphiti and FalkorDB Lite. Without it the service runs on Markdown and FTS alone.

To upgrade, change the tag, run `uv lock`, and commit both files. Tags follow `vX.Y.Z`, and `CHANGELOG.md` says what each one changed.

### Running a vault against a local checkout

To try an engine change in a real vault before tagging it, layer your checkout over the vault's environment for one command:

```bash
uv run --project ~/my-vault --with-editable ~/git/brain-engine brain-engine serve --vault ~/my-vault
```

The vault's `pyproject.toml` and `uv.lock` don't change, so there is nothing to undo.

## Running the service

```bash
brain-engine serve          # Streamable HTTP on http://127.0.0.1:<port>/mcp
brain-engine search "query" # through the service if it runs, otherwise locally
brain-engine sync-graph     # ingest new and edited Markdown into Graphiti
brain-engine reindex        # rebuild the FTS5 index from the Markdown
brain-engine metrics        # how long operations took on this machine; --days N
```

Each command takes `--vault DIR`, then falls back to `MEMORY_VAULT`, then the current directory.

### One process owns the graph

The service opens the Markdown store, the FTS5 index and FalkorDB Lite once at start-up and shares them across all clients. It holds `service.lock` in the state folder while it runs, so:

- a second `serve` on the same vault refuses to start;
- `search` and `sync-graph` find the running service through the lock and call it over MCP. With no service running, they work locally and hold the lock while they do;
- `reindex` touches only SQLite and is safe at any time.

### Writes don't wait for the graph

A Graphiti ingest takes about 20 seconds of OpenAI calls, against about 40 ms for the Markdown write and FTS. While the service runs, `remember`, `remember_session`, `edit_memory` and `correct_memory` return once the Markdown file and FTS have the memory, with `"graph": {"queued": true, "memory_id": ...}`. One background worker ingests queued memories in write order, one at a time, because Graphiti resolves each episode's entities against the graph built so far. A memory edited again before its turn is ingested once, with its newest text.

- For about 20 seconds after a write, search finds the memory through FTS but not yet through graph facts.
- At start-up the service queues every memory `sync_memory_graph` would ingest. A memory whose text the graph already has is skipped without calling OpenAI, so this costs a few milliseconds per memory. It picks up memories pulled from git and anything still queued at the last stop.
- Stopping the service doesn't wait for an ingest in flight. The memory is already in Markdown and FTS, and the next start queues it again.
- A failed ingest is logged and recorded in graph-state and `metrics`, and the queue moves on. The next start retries it.
- If Graphiti didn't start, writes ingest right away and return the error, rather than queueing work that would fail where nobody sees it.
- `sync_memory_graph`, and the CLI when no service runs, ingest right away and return counts.

### Stopping

Ctrl-C, `kill <pid>` (SIGTERM) and closing the terminal that runs the service (SIGHUP) all stop it the same way, for `serve` and `stdio` alike. The service logs `<name> stopped` when it is done. A service started with SIGHUP ignored, as `nohup` does, keeps ignoring it.

- Each connected client holds a `subscriptions/listen` request open. The service ends those streams first, which is the spec's graceful close, and clients listen again when the service is back. uvicorn's 3-second timeout then only cuts off a tool call still running.
- FalkorDB Lite runs a `redis-server` process that writes `graphiti.db` only on a timer. Closing the graph sends it `SHUTDOWN SAVE`, so the file is current and the process exits.
- A signal that arrives while the service is still opening the graph cancels the start and closes what was opened.
- Some terminals don't send a signal the service can catch. Closing a tab in the Claude desktop app's terminal panel SIGKILLs the service and every process it started. FalkorDB Lite's redis-server is detached from the service, so it survives that. When the graph opens, the engine starts a small watcher, detached the same way, that checks on the service twice a second. Once the service is gone the watcher sends the redis-server `SHUTDOWN SAVE`, so `graphiti.db` is saved and nothing is left running about a second later. When the service stops the server itself, the watcher just exits.

Older servers stopped cleanly only on Ctrl-C. After a `kill` or a closed terminal their redis-server kept running, and whatever killed it later, a logout for example, made it write its in-memory graph over `graphiti.db`. That graph could be older than the one on disk. Two things clean up after them:

- On start, before opening the graph, the engine looks for running FalkorDB Lite redis-servers working in its state folder with its `graphiti.db` as their database. One with no unsaved changes is stopped without saving. If exactly one has unsaved changes, it holds writes from a service that died, so it saves and then stops. If more than one has unsaved changes, the engine refuses to start and names them, since guessing would lose data. It also refuses when a server for its graph doesn't answer, since it may still be saving. The log says what was stopped.
- Opening the graph checks graph-state against the episodes actually in the graph. A memory marked ingested whose episode is gone is marked `missing`, and the start-up catch-up and `sync-graph` ingest it again. The log counts them.

The engine also deletes FalkorDB Lite's `graphiti.db.settings` before opening, so it never reattaches to a server it didn't start. In a copied state folder that file names the original vault's server.

### Graph viewer

While the service runs, <http://127.0.0.1:PORT/graph> shows the Graphiti graph.

- Entities are circles sized by how many facts touch them. One line joins each pair of entities and carries every fact between them. The arrow shows the direction, and a line without an arrowhead has facts both ways.
- Only labels that fit are drawn, so zoom in to see more. Hover a node to light up its neighbours and the names of its facts.
- The chips at the top count entity kinds. Click one to highlight it. The relationship menu highlights the facts with one name.
- Search finds an entity. Enter selects the best match and frames it with its neighbours.
- Click a node or line to inspect it. A line lists each fact with its dates and the Markdown memories it came from.
- Double-click a node to show only it and its neighbours. Drag a node to move it.
- **Show history** adds invalidated and superseded facts as fainter lines.

The page is read-only and answers only when the `Host` header is localhost, which blocks DNS-rebinding pages from reading memories. It is a plain web page, not an MCP tool, so agents still see only the memory API. It renders with [Sigma.js](https://www.sigmajs.org/) and lays the graph out with ForceAtlas2 from graphology, loaded as pinned ES modules from jsDelivr. The graph data never leaves the machine.

### Timings

The service writes one row per operation to `metrics.sqlite3`: the operation (the MCP tool name, `graph_ingest` for one background ingest, or `graph_view` for `/graph.json`), the total time, the part spent in Graphiti and its OpenAI calls (`graph_ms`), the rest (`store_ms`), the number of searchable memories at the time, and for writes the content size. An operation that raised, or whose graph step failed and fell back, keeps the error. Recording never fails the operation. A background ingest that skipped an unchanged memory isn't recorded.

`brain-engine metrics` reports the last 30 days. It gives p50, p90 and max per operation with the store and graph split, then the medians grouped by vault size, so a slowdown that comes with growth shows up as rising rows. Ingests also get Graphiti time per KB, because ingest time grows with the memory's size and only a rise per KB means the graph itself is slowing down.

`scripts/bench_store.py <vault> [sizes...]` answers the same question ahead of time. It copies the vault's Markdown into a temporary folder, pads it with renamed copies of its own memories up to each size, and times search, get, remember, edit and verify with Graphiti stubbed out. It never touches the vault and makes no OpenAI calls.

## Connecting clients

### Claude Code

Add a `.mcp.json` to the vault, or to any project that should see the memory:

```json
{
  "mcpServers": {
    "my-memory": {
      "type": "http",
      "url": "http://127.0.0.1:8766/mcp"
    }
  }
}
```

Claude Code asks once to approve it. For every project, use `claude mcp add --scope user --transport http my-memory http://127.0.0.1:8766/mcp`.

### Cursor

`.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "my-memory": {
      "url": "http://127.0.0.1:8766/mcp"
    }
  }
}
```

Enable it once under Cursor Settings, MCP.

### Gemini CLI

```bash
gemini mcp add --scope user --transport http my-memory http://127.0.0.1:8766/mcp
```

### ChatGPT

Not supported yet. ChatGPT can't reach `127.0.0.1`, so it needs the service behind a tunnel or another remote MCP route, and that needs authentication first.

### stdio

For a client that only speaks stdio:

```bash
uv run --project /path/to/vault brain-engine stdio --vault /path/to/vault
```

This starts a separate server inside the client's process. It takes the same lock, so it can't run alongside `serve` or another stdio client. Prefer the HTTP service.

## The memory model

### Memory format

```markdown
---
type: decision
id: mem_abc123
title: Memory architecture
tags:
  - memory
status: draft
created_at: 2026-09-19T20:00:00Z
generated:
  by: claude-code/local
  at: 2026-09-19T20:00:00Z
verified_by_user: false
searchable: true
content_sha256: ...
sources:
  - resource: conversation:123
---

Markdown remains the source of truth. Graphiti is a derived temporal index.
```

### Retrieval

Agents search first and read second. `search_memory` returns compact current facts from the graph plus short Markdown snippets from FTS. The agent picks the one to three memories that matter and reads them with `get_memory`, which pages large memories so a long transcript can't enter the prompt in one call. Transcripts saved through `remember_session` are `searchable: false`. They are kept, but out of normal search.

### Duplicates

The store hashes normalized `type + content` before writing. An exact duplicate creates no file, no FTS row and no graph episode. Graph sync hashes the file body again, so hand edits are noticed even when the frontmatter hash is stale.

Graphiti resolves entities and deduplicates facts across episodes. The service also drops exact-duplicate facts before returning them.

### Contradictions and time

Don't overwrite a memory because reality changed. "The office opens at 9" in September and "the office opens at 10" in October are two memories and two episodes, and Graphiti marks the first fact as no longer valid. Three tools keep this straight:

- `remember` records something new: an event, a decision, a changed fact.
- `edit_memory` fixes the stored text itself, like a typo. It edits the file in place, clears verification and replaces the graph episode.
- `correct_memory` records that a memory was wrong. It writes a new memory with `corrects: <old id>` and marks the old one `status: superseded`, `superseded_by: <new id>`. The old episode stays in the graph so the history survives.

By default `search_memory` leaves out graph facts with `invalid_at` set, superseded memories, and graph facts whose every source memory is superseded, so a missed invalidation can't make an old source look current. `include_history=true` brings them back.

### Provenance and verification

Each memory records who wrote it (`generated.by`) and, optionally, where it came from (`sources`). `verified_by_user` stays false until a person confirms the memory and an agent calls `verify_memory`, which writes `verified.by: human:<MEMORY_HUMAN_ID>`. Any edit clears it.

### MCP tools

Always:

- `search_memory(query, limit=5, kind=None, include_history=false)`. Graph facts come from Graphiti's hybrid search, which needs OpenAI to embed the query. If that takes longer than 0.5 s, the facts come from the graph's keyword index alone, which is local and takes about 40 ms. If Graphiti is down, search returns FTS results and says why in `graph_error`.
- `get_memory(memory_id, offset=0, max_chars=8000)`.

With `MEMORY_ACCESS=read-write`: `remember`, `remember_session`, `edit_memory`, `correct_memory`, `verify_memory`, `sync_memory_graph`. Agents never get file-system write tools.

### Suggested agent rules

Put rules like these in the vault's `instructions` or in the agent's own instructions:

1. Search memory when history or decisions can help. Never load the whole vault.
2. Prefer current facts unless the user asks about history.
3. Fetch full memories only after search points to them.
4. When something new happens, call `remember` rather than rewriting history. Use `edit_memory` only for errors in the text.
5. Treat `verified_by_user=false` as useful but not confirmed.
6. Ask before storing sensitive material. Suggest candidate memories instead of bulk-importing.

## Changes from the per-vault servers

brain-engine replaces vault-specific servers that each ran their own copy of this code. If you ran one:

- Its own command is gone. Use `brain-engine <command> --vault <dir>`. MCP clients that connect over HTTP need no change, since the name, port and URL come from `brain.toml`.
- `MEMORY_ROOT` used to move both the Markdown and the state folder, which sat next to it. It now moves only the Markdown, and `state_dir` stays relative to the vault.
- The old servers found `memory/` relative to their own source files. The engine is installed as a package, so it finds the vault through `--vault`, `MEMORY_VAULT` or the current directory.
- The state folder's files keep their names and formats. A vault that points `state_dir` at its old folder keeps its index, graph and timings without re-ingesting anything. Keep the old server's name as `name` too, so clients see the same MCP server.

## Development

```bash
just install   # uv sync --extra graph --extra dev
just test
```

The tests need no network and no API key. They replace Graphiti with fakes where they can and start real FalkorDB Lite servers where the redis-server itself is under test. `AGENTS.md` has the code map.

## License

MIT
