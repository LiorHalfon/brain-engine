---
name: brain-engine
description: Use when working on a brain-engine memory vault (a folder with a brain.toml) or its service - setting up a vault, moving a vault's own memory server onto brain-engine, upgrading the pinned engine tag, `brain-engine serve` refusing to start, or cleaning up or rebuilding the vault's graph.
---

# brain-engine vaults

A vault is a folder with a `brain.toml`, its Markdown memories, and a `pyproject.toml` that pins a brain-engine tag. The reference is the engine's README at that tag, `https://github.com/LiorHalfon/brain-engine/blob/<tag>/README.md`. `CHANGELOG.md` says what each tag changed, and `examples/vault/` is a starting vault. Where the vault wraps `brain-engine <command> --vault <dir>` in its own task-runner recipes, use the recipes.

## Rules for every job

- **The vault's service belongs to the user.** `brain-engine serve` runs for as long as the user works, in a terminal they own. Give them the command to run, and ask before stopping it. A service started in an agent's own shell stops when the agent's session ends, so only a test copy on a spare port runs there.
- **Check with reads.** Use the proof checks below. A memory written to test a real vault stays in its history and its graph.
- **A graph rebuild costs time and money.** The Markdown is the vault and the state folder is derived, but ingesting one memory into the graph takes about 20 seconds of OpenAI calls. Deleting `graphiti.db` or `graph-state.sqlite3` re-ingests every memory. Tell the user the memory count and the hours it will take, and let them decide.
- **`service.lock` is a live OS lock (`flock`).** The kernel releases it when its process dies, so "state is already in use by: <owner>" always means a running process holds it. `lsof <state_dir>/service.lock` finds that process. The owner is the service's URL, `stdio` for a server an MCP client started, or `cli` for a `search` or `sync-graph` running locally. With the user's OK, stop that process with Ctrl-C in its terminal or `kill <pid>`. Deleting the lock file lets a second process open the same graph.

## Proof checks

After anything that changes a working vault's code or state, with the service running:

```bash
uv run brain-engine sync-graph --vault <vault>                         # indexed: 0, removed: 0, errors: []
uv run brain-engine search "<a topic the vault knows>" --vault <vault>  # facts, graph_error: null
```

`indexed` counts memories ingested just now. A few means memories written since the last ingest. A count near the vault's size means the graph state didn't carry over: stop the service and find out why before going on.

`removed` counts memories whose Markdown file is gone, removed from the graph just now. A few means files deleted since the last sync, here or in a `git pull`. The first sync on an engine newer than 0.1.1 also removes the memories deleted before it.

A `not removed` error means the sync kept the graph because the Markdown folder looks wrong:

- "not removed: <folder> holds no memories" or "not removed: <n> memories look deleted": check the vault's Markdown folder, `MEMORY_ROOT` and the checkout's branch before anything else. Run `sync-graph --allow-mass-removal` only once the user confirms those memories really were deleted.
- "not removed: these files don't parse": fix the files it names, or delete them if they hold no memory. The next sync carries on, and a fixed file needs no new ingest.

## Setting up a vault

Copy `examples/vault/` at the tag, then follow the README's "Installing in a vault" and "Connecting clients".

- Give the vault a `name` and `port` no other vault on the machine uses. `lsof -nP -iTCP:<port> -sTCP:LISTEN` prints nothing for a free port.
- The engine never loads `.env`. Pass `OPENAI_API_KEY` in through the vault's task runner, with `set dotenv-load` in a `justfile` for example.
- An MCP client reads its config when a session starts, so the agent that writes `.mcp.json` sees the new server from its next session.

## Moving a vault's own memory server onto the engine

The README's "Changes from the per-vault servers" lists what changes for the vault.

1. Carry the settings over exactly: the old server name as `name`, its port, its state folder as `state_dir`, its kinds, dated kinds and instructions text. Compare them with the old code by script. A new `name` renames the MCP server in every client, and a new `state_dir` re-ingests every memory.
2. Pin the tag, point the vault's commands at `brain-engine <command> --vault <dir>`, and remove the old server code.
3. Test on a copy while the old service keeps serving:
   - Ask the user to stop the old service with Ctrl-C, since older servers save their graph only then. Copy the new `brain.toml`, the Markdown folder and the state folder into a scratch folder, leaving out `service.lock` and `graphiti.db.settings`. Then ask them to start the old service again. It is down for seconds.
   - From the vault with the new pin installed and `OPENAI_API_KEY` in the environment, run `MEMORY_PORT=<spare port> uv run brain-engine serve --vault <copy>`. Before any other command against the copy, check that the copy's `service.lock` holds the spare port's URL.
   - Run the proof checks against the copy. Stop it with Ctrl-C, then check that `lsof +D <copy's state folder>` lists nothing, which means its redis-server is gone too.
4. Switch: the user stops the old service, runs `uv sync`, and starts `brain-engine serve`. Then run the proof checks against the vault.

## Upgrading the engine tag

1. Read `CHANGELOG.md` from the current tag to the new one, and do any migration it names.
2. Change the tag in `pyproject.toml`, run `uv lock`, and commit both files.
3. The running service keeps the old code until it restarts. Ask the user to restart it, then run the proof checks.
4. Copy this skill again from the new tag, as the README's "Agent skill" shows.

## When `serve` won't start

- **"state is already in use by"**: see the `service.lock` rule.
- **"leftover redis-servers hold unsaved writes"**: either choice keeps every memory. The start-up repair re-ingests the graph episodes the dropped servers held, and the log counts them ("to be ingested again"). Keep the server with the most unsaved changes. Stop each of the others with `redis-cli -s <socket> SHUTDOWN NOSAVE`, then start again: with one server left holding unsaved writes, the engine saves and stops it itself. FalkorDB Lite ships `redis-cli` beside its `redis-server`, whose path `ps -o command= -p <pid>` prints. Tell the user which server you kept.
- **A redis-server that "doesn't answer"**: it may still be saving. Wait a minute and start again.
- **"Graphiti unavailable, serving Markdown/FTS only"** in the log: the service is up without graph facts. The rest of that line gives the reason, usually a missing `OPENAI_API_KEY`.

## Cleaning up the graph

For a memory that was wrong, `correct_memory` keeps the history and hides facts whose every source is superseded. For a memory that must not be kept at all, delete its Markdown file and run `sync-graph`. Search stops returning the facts only that memory supports as soon as the file is gone. The sync removes the memory's episode with those facts and entities, and counts it in `removed`. Every machine's next sync does the same, so a deletion pulled from git cleans up there too. The README's "Deleting a memory" lists what a removal leaves behind.

A fact the deleted memory had invalidated stays invalidated. Only a rebuild brings it back: stop the service, move `graphiti.db` and `graph-state.sqlite3` aside, and start it again. It ingests every memory, so the rebuild rule above applies.
