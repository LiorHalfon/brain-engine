from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class MarkdownDocument:
    metadata: dict[str, Any]
    body: str


def parse_markdown(path: Path) -> MarkdownDocument:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        raise ValueError(f"Missing YAML frontmatter: {path}")
    end = text.find("\n---\n", 4)
    if end == -1:
        raise ValueError(f"Unterminated YAML frontmatter: {path}")
    raw_meta = text[4:end]
    try:
        metadata = yaml.safe_load(raw_meta) or {}
    except yaml.YAMLError as exc:
        # A typo or conflict markers. Left as is, one such file stopped every call that reads the index.
        raise ValueError(f"Invalid YAML frontmatter: {path}") from exc
    if not isinstance(metadata, dict):
        raise ValueError(f"Frontmatter must be a mapping: {path}")
    body = text[end + 5 :].lstrip("\n")
    return MarkdownDocument(metadata=metadata, body=body)


def render_markdown(metadata: dict[str, Any], body: str) -> str:
    frontmatter = yaml.safe_dump(
        metadata,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    ).strip()
    return f"---\n{frontmatter}\n---\n\n{body.rstrip()}\n"
