from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _toml_value(value: Any) -> str:
    # JSON strings, numbers and lists of them are valid TOML too.
    return json.dumps(value)


def make_vault(path: Path, **fields: Any) -> Path:
    """Write a brain.toml into `path` with a test name and instructions plus `fields`.

    A dict value becomes a table, e.g. kinds={"decision": "decisions"}.
    """
    path.mkdir(parents=True, exist_ok=True)
    values = {"name": "test-vault", "instructions": "Test vault.", **fields}
    lines = [f"{key} = {_toml_value(value)}" for key, value in values.items() if not isinstance(value, dict)]
    for key, table in values.items():
        if isinstance(table, dict):
            lines += ["", f"[{key}]"] + [f"{_toml_value(k)} = {_toml_value(v)}" for k, v in table.items()]
    (path / "brain.toml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
