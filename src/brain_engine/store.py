from __future__ import annotations

import hashlib
import re
import sqlite3
import unicodedata
import uuid
from collections.abc import Iterable, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from brain_engine.config import DEFAULT_KINDS
from brain_engine.db import connection
from brain_engine.markdown import MarkdownDocument, parse_markdown, render_markdown


RESERVED_FILENAMES = {"index.md", "log.md"}
# Bump when the index tables change; an index with another version is dropped and rebuilt.
INDEX_SCHEMA = "2"


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^\w\s-]", "", value, flags=re.UNICODE)
    value = re.sub(r"[-\s]+", "-", value).strip("-")
    return value[:80] or "memory"


def normalize_content(value: str) -> str:
    value = unicodedata.normalize("NFKC", value)
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = "\n".join(line.rstrip() for line in value.strip().splitlines())
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value


def content_sha256(kind: str, content: str) -> str:
    normalized = f"{kind.strip().lower()}\n{normalize_content(content)}"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _fts_query(query: str) -> str:
    terms = re.findall(r"[\w]+", query, flags=re.UNICODE)
    if not terms:
        return '""'
    return " OR ".join(f'"{term.replace(chr(34), "")}"' for term in terms[:12])


@dataclass
class SearchResult:
    id: str
    title: str
    type: str
    path: str
    status: str
    verified_by_user: bool
    snippet: str


@dataclass
class MemoryRecord:
    id: str
    path: str
    metadata: dict[str, Any]
    content: str


class MemoryStore:
    def __init__(
        self,
        root: Path,
        actor: str,
        human_id: str,
        *,
        state_dir: Path | None = None,
        kinds: Mapping[str, str] = DEFAULT_KINDS,
        dated_kinds: Iterable[str] = (),
    ) -> None:
        self.root = root
        self.actor = actor
        self.human_id = human_id
        self.state_dir = state_dir if state_dir is not None else self.root.parent / ".brain"
        # Memory kind -> folder under root. Dated kinds record an event, so their filenames start with the date.
        self.kinds = dict(kinds)
        self.dated_kinds = frozenset(dated_kinds)
        self.db_path = self.state_dir / "index.sqlite3"
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        for directory in self.kinds.values():
            (self.root / directory).mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> AbstractContextManager[sqlite3.Connection]:
        return connection(self.db_path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            row = conn.execute("SELECT value FROM meta WHERE key='schema'").fetchone()
            if not row or row["value"] != INDEX_SCHEMA:
                # FTS schemas cannot be altered, and the index is derived from Markdown anyway.
                conn.execute("DROP TABLE IF EXISTS memories_fts")
                conn.execute("DROP TABLE IF EXISTS documents")
                conn.execute("DELETE FROM meta")
            # One row per Markdown file, with the stat it was indexed at. Unparseable files get a
            # row with no id, so they are not parsed again until they change.
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    docid INTEGER PRIMARY KEY,
                    path TEXT NOT NULL UNIQUE,
                    mtime_ns INTEGER NOT NULL,
                    size INTEGER NOT NULL,
                    id TEXT,
                    type TEXT,
                    status TEXT,
                    content_sha256 TEXT
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS documents_id ON documents(id)")
            conn.execute("CREATE INDEX IF NOT EXISTS documents_sha256 ON documents(content_sha256)")
            # Rows share their rowid with the file's documents.docid.
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                    id UNINDEXED,
                    title,
                    body,
                    tags,
                    type,
                    path UNINDEXED,
                    verified UNINDEXED,
                    status UNINDEXED,
                    tokenize='unicode61'
                )
                """
            )
            conn.execute(
                "INSERT INTO meta(key,value) VALUES('schema',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (INDEX_SCHEMA,),
            )

    def _documents(self) -> list[Path]:
        return sorted(p for p in self.root.rglob("*.md") if p.name not in RESERVED_FILENAMES)

    def ensure_index(self) -> bool:
        """Bring the index up to date with the Markdown files, parsing only the ones that changed.

        When nothing changed this costs a stat per file, so a `git pull` or a hand edit is picked
        up by the next call without reading the vault again.
        """
        return self._sync(rebuild=False)

    def reindex(self) -> int:
        """Rebuild the index from every file. Returns the number of searchable memories."""
        self._sync(rebuild=True)
        return self.count()

    def _sync(self, *, rebuild: bool) -> bool:
        on_disk: dict[str, tuple[int, int]] = {}
        for path in self._documents():
            try:
                stat = path.stat()
            except FileNotFoundError:
                continue
            on_disk[path.relative_to(self.root).as_posix()] = (stat.st_mtime_ns, stat.st_size)
        with self._connect() as conn:
            if rebuild:
                conn.execute("DELETE FROM memories_fts")
                conn.execute("DELETE FROM documents")
            indexed = {
                row["path"]: (row["mtime_ns"], row["size"])
                for row in conn.execute("SELECT path, mtime_ns, size FROM documents")
            }
            removed = indexed.keys() - on_disk.keys()
            changed = [rel for rel, stamp in on_disk.items() if indexed.get(rel) != stamp]
            for rel in removed:
                self._forget(conn, rel)
            for rel in changed:
                self._forget(conn, rel)
                self._index_file(conn, rel, on_disk[rel])
        return bool(removed or changed)

    @staticmethod
    def _forget(conn: sqlite3.Connection, rel: str) -> None:
        conn.execute("DELETE FROM memories_fts WHERE rowid IN (SELECT docid FROM documents WHERE path=?)", (rel,))
        conn.execute("DELETE FROM documents WHERE path=?", (rel,))

    def _index_file(self, conn: sqlite3.Connection, rel: str, stamp: tuple[int, int]) -> None:
        try:
            doc = parse_markdown(self.root / rel)
        except FileNotFoundError:
            return
        except ValueError:
            conn.execute("INSERT INTO documents(path, mtime_ns, size) VALUES (?,?,?)", (rel, *stamp))
            return
        meta = doc.metadata
        memory_id = str(meta.get("id") or rel)
        kind = str(meta.get("type") or "note")
        status = str(meta.get("status") or "draft")
        docid = conn.execute(
            "INSERT INTO documents(path, mtime_ns, size, id, type, status, content_sha256) VALUES (?,?,?,?,?,?,?)",
            (rel, *stamp, memory_id, kind, status, content_sha256(kind, doc.body)),
        ).lastrowid
        if meta.get("searchable", True) is False:
            return
        tags_value = meta.get("tags") or []
        tags = " ".join(map(str, tags_value if isinstance(tags_value, list) else [tags_value]))
        verified = "1" if bool(meta.get("verified_by_user", False)) else "0"
        conn.execute(
            "INSERT INTO memories_fts(rowid,id,title,body,tags,type,path,verified,status) VALUES (?,?,?,?,?,?,?,?,?)",
            (docid, memory_id, str(meta.get("title") or Path(rel).stem), doc.body, tags, kind, rel, verified, status),
        )

    def count(self) -> int:
        """Searchable memories in the FTS index, as of its last update."""
        with self._connect() as conn:
            return conn.execute("SELECT count(*) FROM memories_fts").fetchone()[0]

    def ids(self) -> set[str]:
        """The id of every memory in the Markdown folder, searchable or not. A file that doesn't
        parse has no id, so its memory is gone from here as it is from search. `unparsed_paths`
        lists those files."""
        self.ensure_index()
        with self._connect() as conn:
            return {row["id"] for row in conn.execute("SELECT id FROM documents WHERE id IS NOT NULL")}

    def unparsed_paths(self) -> list[str]:
        """The Markdown files that don't parse, as of the index's last update."""
        with self._connect() as conn:
            return [row["path"] for row in conn.execute("SELECT path FROM documents WHERE id IS NULL ORDER BY path")]

    def statuses(self, memory_ids: list[str]) -> dict[str, str]:
        """The status of each id the index knows, as of its last update."""
        if not memory_ids:
            return {}
        placeholders = ",".join("?" for _ in memory_ids)
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT id, status FROM documents WHERE id IN ({placeholders})", memory_ids
            ).fetchall()
        return {row["id"]: row["status"] for row in rows}

    def search(
        self,
        query: str,
        limit: int = 5,
        kind: str | None = None,
        include_superseded: bool = False,
    ) -> list[SearchResult]:
        self.ensure_index()
        limit = max(1, min(limit, 10))
        match_query = _fts_query(query)
        sql = (
            "SELECT id, title, type, path, verified, status, "
            "snippet(memories_fts, 2, '', '', ' … ', 28) AS snippet, "
            "bm25(memories_fts, 3.0, 1.0, 1.5, 1.0, 0.0, 0.0, 0.0, 0.0) AS score "
            "FROM memories_fts WHERE memories_fts MATCH ?"
        )
        params: list[Any] = [match_query]
        if kind:
            sql += " AND type = ?"
            params.append(kind)
        if not include_superseded:
            sql += " AND status != 'superseded'"
        sql += " ORDER BY score LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [
            SearchResult(
                id=row["id"],
                title=row["title"],
                type=row["type"],
                path=row["path"],
                status=row["status"],
                verified_by_user=row["verified"] == "1",
                snippet=(row["snippet"] or "")[:700],
            )
            for row in rows
        ]

    def _load(self, memory_id: str) -> tuple[Path, MarkdownDocument]:
        # The index says which file holds the id and reading that file confirms it. Only a miss
        # (a file added, moved or deleted since the last update) updates the index and looks again.
        for refresh in (False, True):
            if refresh:
                self.ensure_index()
            with self._connect() as conn:
                rows = conn.execute("SELECT path FROM documents WHERE id=? ORDER BY path", (memory_id,)).fetchall()
            for row in rows:
                path = self.root / row["path"]
                try:
                    doc = parse_markdown(path)
                except (FileNotFoundError, ValueError):
                    continue
                if str(doc.metadata.get("id") or row["path"]) == memory_id:
                    return path, doc
        raise KeyError(f"Unknown memory id: {memory_id}")

    def record(self, memory_id: str) -> MemoryRecord:
        path, doc = self._load(memory_id)
        return MemoryRecord(
            id=memory_id,
            path=path.relative_to(self.root).as_posix(),
            metadata=doc.metadata,
            content=doc.body,
        )

    def iter_records(self, searchable_only: bool = True) -> list[MemoryRecord]:
        records: list[MemoryRecord] = []
        for path in self._documents():
            try:
                doc = parse_markdown(path)
            except ValueError:
                continue
            if searchable_only and doc.metadata.get("searchable", True) is False:
                continue
            memory_id = str(doc.metadata.get("id") or path.relative_to(self.root).as_posix())
            records.append(
                MemoryRecord(
                    id=memory_id,
                    path=path.relative_to(self.root).as_posix(),
                    metadata=doc.metadata,
                    content=doc.body,
                )
            )
        return records

    def get(self, memory_id: str, offset: int = 0, max_chars: int = 8000) -> dict[str, Any]:
        record = self.record(memory_id)
        offset = max(0, offset)
        max_chars = max(500, min(max_chars, 20000))
        chunk = record.content[offset : offset + max_chars]
        return {
            "id": memory_id,
            "path": record.path,
            "metadata": record.metadata,
            "content": chunk,
            "offset": offset,
            "returned_chars": len(chunk),
            "total_chars": len(record.content),
            "truncated": offset + len(chunk) < len(record.content),
            "next_offset": offset + len(chunk) if offset + len(chunk) < len(record.content) else None,
        }

    def find_exact_duplicate(self, kind: str, content: str) -> MemoryRecord | None:
        # The index hashes each file's body when it changes, so hand edits count too.
        self.ensure_index()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, path FROM documents WHERE content_sha256=? ORDER BY path LIMIT 1",
                (content_sha256(kind, content),),
            ).fetchone()
        if row is None:
            return None
        doc = parse_markdown(self.root / row["path"])
        return MemoryRecord(id=row["id"], path=row["path"], metadata=doc.metadata, content=doc.body)

    def remember(
        self,
        *,
        title: str,
        content: str,
        kind: str = "note",
        tags: list[str] | None = None,
        source: str | None = None,
        verified_by_user: bool = False,
        searchable: bool = True,
        extra_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        kind = kind.strip().lower()
        if kind not in self.kinds:
            raise ValueError(f"Unsupported memory type: {kind}. Allowed: {', '.join(self.kinds)}")

        duplicate = self.find_exact_duplicate(kind, content)
        if duplicate is not None:
            return {
                "id": duplicate.id,
                "path": duplicate.path,
                "type": str(duplicate.metadata.get("type") or kind),
                "title": str(duplicate.metadata.get("title") or title),
                "verified_by_user": bool(duplicate.metadata.get("verified_by_user", False)),
                "searchable": duplicate.metadata.get("searchable", True) is not False,
                "duplicate": True,
            }

        timestamp = now_iso()
        memory_id = f"mem_{uuid.uuid4().hex[:12]}"
        folder = self.root / self.kinds[kind]
        folder.mkdir(parents=True, exist_ok=True)
        date_prefix = datetime.now(timezone.utc).strftime("%Y-%m-%d") if kind in self.dated_kinds else ""
        filename = f"{date_prefix + '-' if date_prefix else ''}{slugify(title)}-{memory_id[-6:]}.md"
        path = folder / filename
        metadata: dict[str, Any] = {
            "type": kind,
            "id": memory_id,
            "title": title.strip(),
            "tags": tags or [],
            "status": "stable" if verified_by_user else "draft",
            "created_at": timestamp,
            "generated": {"by": self.actor, "at": timestamp},
            "verified_by_user": verified_by_user,
            "searchable": searchable,
            "content_sha256": content_sha256(kind, content),
        }
        if source:
            metadata["sources"] = [{"resource": source}]
        if verified_by_user:
            metadata["verified"] = {"by": f"human:{self.human_id}", "at": timestamp}
        if extra_metadata:
            metadata.update(extra_metadata)
        path.write_text(render_markdown(metadata, content), encoding="utf-8")
        self.ensure_index()
        return {
            "id": memory_id,
            "path": path.relative_to(self.root).as_posix(),
            "type": kind,
            "title": title,
            "verified_by_user": verified_by_user,
            "searchable": searchable,
            "duplicate": False,
        }

    def edit(self, memory_id: str, content: str, reason: str | None = None) -> dict[str, Any]:
        path, doc = self._load(memory_id)
        timestamp = now_iso()
        kind = str(doc.metadata.get("type") or "note")
        doc.metadata["generated"] = {"by": self.actor, "at": timestamp}
        doc.metadata["verified_by_user"] = False
        doc.metadata.pop("verified", None)
        doc.metadata["status"] = "draft"
        doc.metadata["content_sha256"] = content_sha256(kind, content)
        if reason:
            doc.metadata["last_edit_reason"] = reason
        path.write_text(render_markdown(doc.metadata, content), encoding="utf-8")
        self.ensure_index()
        return {
            "id": memory_id,
            "path": path.relative_to(self.root).as_posix(),
            "verified_by_user": False,
            "status": "draft",
            "edited": True,
        }

    # Backwards-compatible alias used by the first version.
    update = edit

    def correct(
        self,
        memory_id: str,
        correction: str,
        reason: str,
        source: str | None = None,
    ) -> dict[str, Any]:
        old_path, old_doc = self._load(memory_id)
        result = self.remember(
            title=f"Correction: {old_doc.metadata.get('title', memory_id)}",
            content=correction,
            kind=str(old_doc.metadata.get("type") or "note"),
            tags=list(old_doc.metadata.get("tags") or []),
            source=source,
            verified_by_user=False,
            searchable=True,
            extra_metadata={"corrects": memory_id, "correction_reason": reason},
        )
        if result.get("duplicate"):
            return {**result, "corrected": False, "reason": "correction already exists"}

        old_doc.metadata["status"] = "superseded"
        old_doc.metadata["superseded_by"] = result["id"]
        old_doc.metadata["superseded_at"] = now_iso()
        old_path.write_text(render_markdown(old_doc.metadata, old_doc.body), encoding="utf-8")
        self.ensure_index()
        return {**result, "corrected": True, "corrects": memory_id}

    def verify(self, memory_id: str) -> dict[str, Any]:
        path, doc = self._load(memory_id)
        timestamp = now_iso()
        doc.metadata["verified_by_user"] = True
        doc.metadata["verified"] = {"by": f"human:{self.human_id}", "at": timestamp}
        if doc.metadata.get("status") != "superseded":
            doc.metadata["status"] = "stable"
        path.write_text(render_markdown(doc.metadata, doc.body), encoding="utf-8")
        self.ensure_index()
        return {
            "id": memory_id,
            "verified_by_user": True,
            "status": str(doc.metadata.get("status")),
        }

    def remember_session(
        self,
        *,
        title: str,
        summary: str,
        transcript: str | None = None,
        tags: list[str] | None = None,
        source: str | None = None,
    ) -> dict[str, Any]:
        summary_result = self.remember(
            title=title,
            content=summary,
            kind="session",
            tags=tags,
            source=source,
            verified_by_user=False,
            searchable=True,
        )
        transcript_result = None
        if transcript:
            transcript_result = self.remember(
                title=f"{title} - verbatim transcript",
                content=transcript,
                kind="transcript",
                tags=(tags or []) + ["verbatim"],
                source=source,
                verified_by_user=False,
                searchable=False,
            )
            summary_path, doc = self._load(summary_result["id"])
            doc.metadata["transcript_memory_id"] = transcript_result["id"]
            summary_path.write_text(render_markdown(doc.metadata, doc.body), encoding="utf-8")
            self.ensure_index()
        return {"summary": summary_result, "transcript": transcript_result}
