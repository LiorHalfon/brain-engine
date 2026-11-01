# Changelog

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
