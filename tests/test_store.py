import sqlite3
from pathlib import Path

import pytest

from brain_engine import store as store_module
from brain_engine.store import MemoryStore


def test_remember_search_get_update_verify(tmp_path: Path) -> None:
    root = tmp_path / "memory"
    store = MemoryStore(root, actor="test-agent/1", human_id="tester")
    created = store.remember(
        title="Opening hours decision",
        content="We decided that the Springfield office opens at 9.",
        kind="note",
        tags=["acme", "hours"],
        source="conversation:test",
    )

    results = store.search("office opens hours")
    assert len(results) == 1
    assert results[0].id == created["id"]
    assert results[0].verified_by_user is False

    doc = store.get(created["id"])
    assert "opens at 9" in doc["content"]
    assert doc["metadata"]["generated"]["by"] == "test-agent/1"

    store.update(created["id"], "The Springfield office opens at 10.", reason="Hours changed")
    updated = store.get(created["id"])
    assert updated["metadata"]["verified_by_user"] is False
    assert updated["metadata"]["status"] == "draft"

    store.verify(created["id"])
    verified = store.get(created["id"])
    assert verified["metadata"]["verified_by_user"] is True
    assert verified["metadata"]["verified"]["by"] == "human:tester"


def test_large_memory_is_paged(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory", actor="test/1", human_id="owner")
    created = store.remember(title="Long transcript", content="x" * 30000, kind="transcript", searchable=False)
    chunk = store.get(created["id"], max_chars=1000)
    assert chunk["returned_chars"] == 1000
    assert chunk["truncated"] is True
    assert chunk["next_offset"] == 1000


def test_session_transcript_not_search_indexed(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory", actor="test/1", human_id="owner")
    created = store.remember_session(
        title="Memory architecture discussion",
        summary="We chose SQLite FTS5 for the first version.",
        transcript="SECRETVERBATIMTOKEN appeared only in the raw transcript.",
    )
    assert created["transcript"] is not None
    assert store.search("SQLite FTS5")
    assert store.search("SECRETVERBATIMTOKEN") == []


def _count_parses(monkeypatch) -> list[Path]:
    parsed: list[Path] = []
    real_parse = store_module.parse_markdown
    monkeypatch.setattr(store_module, "parse_markdown", lambda path: parsed.append(path) or real_parse(path))
    return parsed


def _store_with(tmp_path: Path, n: int) -> tuple[MemoryStore, list[str]]:
    store = MemoryStore(tmp_path / "memory", actor="test/1", human_id="owner")
    ids = [store.remember(title=f"Fact {i}", content=f"Fact number {i}.", kind="note")["id"] for i in range(n)]
    return store, ids


def test_reads_parse_only_the_memory_asked_for(tmp_path: Path, monkeypatch) -> None:
    store, ids = _store_with(tmp_path, 20)
    parsed = _count_parses(monkeypatch)

    store.get(ids[7])
    store.search("fact number")

    # Each used to parse every file in the vault: 2s per call at 2000 memories.
    assert len(parsed) == 1


def test_writes_parse_only_the_files_they_change(tmp_path: Path, monkeypatch) -> None:
    store, ids = _store_with(tmp_path, 20)
    parsed = _count_parses(monkeypatch)

    store.remember(title="New", content="A new fact.", kind="note")
    assert len(parsed) == 1
    parsed.clear()
    store.edit(ids[3], "Fact number three, edited.")
    assert len(parsed) == 2  # read it, then index the new text
    parsed.clear()
    store.verify(ids[4])
    assert len(parsed) == 2


def test_index_follows_files_changed_outside_the_server(tmp_path: Path) -> None:
    store, ids = _store_with(tmp_path, 3)
    edited, removed = store.record(ids[0]), store.record(ids[1])

    # A git pull: one memory edited by hand, one deleted, one added.
    edited_path = store.root / edited.path
    edited_path.write_text(edited_path.read_text().replace("Fact number 0.", "Pulled zebra text."))
    (store.root / removed.path).unlink()
    added = (store.root / "notes" / "pulled-note.md")
    added.write_text("---\ntype: note\nid: mem_pulled\ntitle: Pulled\n---\n\nPulled giraffe text.\n")

    assert [r.id for r in store.search("zebra")] == [ids[0]]
    assert store.search("Fact number 1") and ids[1] not in {r.id for r in store.search("Fact number 1")}
    assert store.get("mem_pulled")["content"] == "Pulled giraffe text.\n"
    with pytest.raises(KeyError):
        store.get(ids[1])
    assert store.remember(title="Old", content="Fact number 0.", kind="note")["duplicate"] is False
    assert store.remember(title="Same", content="Pulled zebra text.", kind="note")["id"] == ids[0]


def test_an_index_from_before_the_documents_table_is_rebuilt(tmp_path: Path) -> None:
    root = tmp_path / "memory"
    state = tmp_path / ".brain"
    state.mkdir()
    with sqlite3.connect(state / "index.sqlite3") as conn:
        conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        conn.execute(
            "CREATE VIRTUAL TABLE memories_fts USING fts5(id UNINDEXED, title, body, tags, type, "
            "path UNINDEXED, verified UNINDEXED, status UNINDEXED)"
        )
        conn.execute("INSERT INTO memories_fts VALUES ('mem_ghost','Ghost','ghost row','','note','x.md','0','draft')")

    store = MemoryStore(root, actor="test/1", human_id="owner")
    created = store.remember(title="Real", content="A real ghost story.", kind="note")

    assert [r.id for r in store.search("ghost")] == [created["id"]]


def test_a_yaml_error_in_one_file_does_not_stop_the_others(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory", actor="test/1", human_id="owner")
    created = store.remember(title="Carrier", content="Acme ships orders with Parcelco.", kind="note")
    conflicted = "---\nid: note-1\n<<<<<<< HEAD\ntitle: Ours\n=======\ntitle: Theirs\n>>>>>>> main\n---\n\nText.\n"
    (store.root / "conflicted.md").write_text(conflicted, encoding="utf-8")

    assert [result.id for result in store.search("Parcelco")] == [created["id"]]
    assert store.unparsed_paths() == ["conflicted.md"]
    assert [record.id for record in store.iter_records()] == [created["id"]]
