from __future__ import annotations

import re
from pathlib import Path

import pytest

from brain_engine.config import DEFAULT_GRAPH_MODEL, DEFAULT_KINDS, ConfigError, Settings, find_vault
from brain_engine.lock import GraphOwnedElsewhere, hold_state_lock
from brain_engine.store import MemoryStore
from conftest import make_vault


def test_defaults(tmp_path: Path) -> None:
    vault = make_vault(tmp_path / "vault")

    settings = Settings.load(vault, environ={})

    assert settings.name == "test-vault" and settings.instructions == "Test vault."
    assert settings.port == 8766
    assert settings.memory_dir == vault / "memory"
    assert settings.state_dir == vault / ".brain"
    assert settings.kinds == DEFAULT_KINDS
    assert settings.dated_kinds == frozenset()
    assert settings.actor == "test-vault/local"
    assert settings.human_id == "owner" and settings.access == "read-write" and settings.graph_enabled
    assert settings.graph_model == settings.graph_small_model == DEFAULT_GRAPH_MODEL == "gpt-6-luna"
    assert settings.url == "http://127.0.0.1:8766/mcp"


def test_toml_values_and_engine_kinds_added(tmp_path: Path) -> None:
    vault = make_vault(
        tmp_path / "vault",
        port=9001,
        memory_dir="notes-vault",
        state_dir=".x",
        dated_kinds=["decision"],
        graph_model="model-a",
        graph_small_model="model-b",
        kinds={"decision": "decisions", "note": "inbox"},
    )

    settings = Settings.load(vault, environ={})

    assert settings.port == 9001
    assert settings.memory_dir == vault / "notes-vault" and settings.state_dir == vault / ".x"
    # The vault's own folder for `note` wins; the engine adds the kinds its tools write.
    assert settings.kinds == {
        "decision": "decisions",
        "note": "inbox",
        "session": "sessions",
        "transcript": "sessions/transcripts",
    }
    assert settings.dated_kinds == {"decision"}
    assert (settings.graph_model, settings.graph_small_model) == ("model-a", "model-b")


def test_env_overrides_toml(tmp_path: Path) -> None:
    vault = make_vault(tmp_path / "vault", port=9001, graph_model="model-a")
    elsewhere = tmp_path / "elsewhere"

    settings = Settings.load(
        vault,
        environ={
            "MEMORY_PORT": "9100",
            "MEMORY_HOST": "0.0.0.0",
            "MEMORY_GRAPH_MODEL": "m1",
            "MEMORY_GRAPH_SMALL_MODEL": "m2",
            "MEMORY_ROOT": str(elsewhere),
            "MEMORY_ACTOR": "agent/1",
            "MEMORY_HUMAN_ID": "tester",
            "MEMORY_ACCESS": "read-only",
            "MEMORY_GRAPH_ENABLED": "false",
        },
    )

    assert settings.port == 9100 and settings.host == "0.0.0.0"
    assert (settings.graph_model, settings.graph_small_model) == ("m1", "m2")
    assert settings.memory_dir == elsewhere.resolve()
    # MEMORY_ROOT moves only the Markdown; the state stays with the vault.
    assert settings.state_dir == vault / ".brain"
    assert (settings.actor, settings.human_id, settings.access) == ("agent/1", "tester", "read-only")
    assert settings.graph_enabled is False


def test_blank_env_falls_back(tmp_path: Path) -> None:
    vault = make_vault(tmp_path / "vault", port=9001, graph_model="model-a")

    settings = Settings.load(
        vault, environ={"MEMORY_PORT": "", "MEMORY_GRAPH_MODEL": "  ", "MEMORY_GRAPH_SMALL_MODEL": "", "MEMORY_ROOT": ""}
    )

    assert settings.port == 9001
    assert settings.graph_model == "model-a" and settings.graph_small_model == DEFAULT_GRAPH_MODEL
    assert settings.memory_dir == vault / "memory"


def test_missing_brain_toml(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match=r"No brain\.toml in .*--vault"):
        Settings.load(tmp_path, environ={})


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"dated_kind": ["note"]}, "dated_kind"),
        ({"name": ""}, "name"),
        ({"instructions": 3}, "instructions"),
        ({"dated_kinds": ["decision"]}, "decision"),
        ({"port": "eighty"}, "port"),
        ({"kinds": {"Decision": "decisions"}}, "Decision"),
    ],
)
def test_bad_brain_toml_is_named(tmp_path: Path, fields: dict, message: str) -> None:
    vault = make_vault(tmp_path / "vault", **fields)

    with pytest.raises(ConfigError, match=re.escape(message)):
        Settings.load(vault, environ={})


def test_missing_name_is_named(tmp_path: Path) -> None:
    (tmp_path / "brain.toml").write_text('instructions = "x"\n', encoding="utf-8")

    with pytest.raises(ConfigError, match="name"):
        Settings.load(tmp_path, environ={})


def test_bad_access_is_refused(tmp_path: Path) -> None:
    vault = make_vault(tmp_path / "vault")

    with pytest.raises(ConfigError, match="MEMORY_ACCESS"):
        Settings.load(vault, environ={"MEMORY_ACCESS": "admin"})


def test_find_vault_order(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()

    assert find_vault("a", {"MEMORY_VAULT": str(tmp_path / "b")}) == (tmp_path / "a").resolve()
    assert find_vault(None, {"MEMORY_VAULT": str(tmp_path / "b")}) == (tmp_path / "b").resolve()
    assert find_vault(None, {"MEMORY_VAULT": " "}) == tmp_path.resolve()
    assert find_vault(None, {}) == tmp_path.resolve()


def test_store_uses_vault_kinds_and_dated_kinds(tmp_path: Path) -> None:
    store = MemoryStore(
        tmp_path / "memory",
        actor="test/1",
        human_id="tester",
        kinds={**DEFAULT_KINDS, "decision": "decisions"},
        dated_kinds={"decision"},
    )

    dated = store.remember(title="Office hours", content="The office opens at 9.", kind="decision")
    undated = store.remember(title="Carrier", content="Acme ships orders with Parcelco.", kind="note")

    assert re.fullmatch(r"decisions/\d{4}-\d{2}-\d{2}-office-hours-\w{6}\.md", dated["path"])
    assert re.fullmatch(r"notes/carrier-\w{6}\.md", undated["path"])
    with pytest.raises(ValueError, match="Allowed: .*decision"):
        store.remember(title="Unknown", content="No folder for this kind.", kind="vendor")


def test_store_state_dir_defaults_next_to_the_memory_folder(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory", actor="test/1", human_id="tester")

    assert store.state_dir == tmp_path / ".brain"
    assert (store.state_dir / "index.sqlite3").exists()


def test_lock_message_uses_the_vault_name(tmp_path: Path) -> None:
    with hold_state_lock(tmp_path, "http://127.0.0.1:1/mcp", name="test-vault"):
        with pytest.raises(GraphOwnedElsewhere, match=r"^test-vault state is already in use by: http://127\.0\.0\.1:1/mcp"):
            with hold_state_lock(tmp_path, "cli", name="test-vault"):
                pass
