from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CONFIG_FILENAME = "brain.toml"
# The kinds the engine's own tools write: `remember` defaults to note, `remember_session` writes
# session and transcript. A vault that leaves one out gets this folder for it.
DEFAULT_KINDS = {"note": "notes", "session": "sessions", "transcript": "sessions/transcripts"}
# graphiti-core 0.31.2's own default, written down so that upgrading it can't switch models unnoticed.
DEFAULT_GRAPH_MODEL = "gpt-6-luna"
DEFAULT_PORT = 8766
FIELDS = {
    "name",
    "instructions",
    "port",
    "memory_dir",
    "state_dir",
    "kinds",
    "dated_kinds",
    "graph_model",
    "graph_small_model",
}


class ConfigError(ValueError):
    pass


def _env(environ: Mapping[str, str], name: str) -> str | None:
    # A blank value means "not set", so a `.env` line like `MEMORY_PORT=` falls back.
    value = environ.get(name, "").strip()
    return value or None


def find_vault(arg: str | None, environ: Mapping[str, str]) -> Path:
    """The vault folder: `--vault`, then MEMORY_VAULT, then the current directory."""
    chosen = (arg or "").strip() or _env(environ, "MEMORY_VAULT") or "."
    return Path(chosen).expanduser().resolve()


def _str(config: dict[str, Any], key: str, default: str | None = None) -> str:
    value = config.get(key, default)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{CONFIG_FILENAME}: `{key}` must be a non-empty string")
    return value


def _kinds(config: dict[str, Any]) -> dict[str, str]:
    table = config.get("kinds", {})
    if not isinstance(table, dict):
        raise ConfigError(f'{CONFIG_FILENAME}: `kinds` must be a table of kind = "folder"')
    for kind, folder in table.items():
        if not kind or kind != kind.strip().lower():
            raise ConfigError(f"{CONFIG_FILENAME}: kind `{kind}` must be lowercase with no spaces around it")
        if not isinstance(folder, str) or not folder.strip():
            raise ConfigError(f"{CONFIG_FILENAME}: the folder for kind `{kind}` must be a non-empty string")
    return {**table, **{kind: folder for kind, folder in DEFAULT_KINDS.items() if kind not in table}}


def _dated_kinds(config: dict[str, Any], kinds: dict[str, str]) -> frozenset[str]:
    value = config.get("dated_kinds", [])
    if not isinstance(value, list) or not all(isinstance(kind, str) for kind in value):
        raise ConfigError(f"{CONFIG_FILENAME}: `dated_kinds` must be a list of kinds")
    unknown = [kind for kind in value if kind not in kinds]
    if unknown:
        raise ConfigError(f"{CONFIG_FILENAME}: dated kind `{unknown[0]}` has no folder in `kinds`")
    return frozenset(value)


def _flag(environ: Mapping[str, str], name: str, default: bool) -> bool:
    value = _env(environ, name)
    if value is None:
        return default
    if value.lower() in {"1", "true", "yes", "on"}:
        return True
    if value.lower() in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"{name} must be true or false, not {value!r}")


def _port(raw: object, source: str) -> int:
    try:
        if isinstance(raw, (bool, float)):
            raise ValueError
        port = int(raw)  # type: ignore[call-overload]
        if not 0 < port < 65536:
            raise ValueError
    except (TypeError, ValueError):
        raise ConfigError(f"{source} must be a port number, not {raw!r}") from None
    return port


@dataclass(frozen=True)
class Settings:
    """One vault's settings: its brain.toml, overridden by the environment, over engine defaults."""

    vault: Path
    name: str
    instructions: str
    memory_dir: Path
    state_dir: Path
    kinds: dict[str, str]
    dated_kinds: frozenset[str]
    host: str
    port: int
    access: str
    actor: str
    human_id: str
    graph_enabled: bool
    graph_model: str
    graph_small_model: str

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/mcp"

    @classmethod
    def load(cls, vault: Path, environ: Mapping[str, str] | None = None) -> "Settings":
        environ = os.environ if environ is None else environ
        path = vault / CONFIG_FILENAME
        try:
            config = tomllib.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise ConfigError(
                f"No {CONFIG_FILENAME} in {vault}. Pass --vault or set MEMORY_VAULT to the vault folder."
            ) from None
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"{path}: {exc}") from None
        unknown = sorted(set(config) - FIELDS)
        if unknown:
            raise ConfigError(f"{path}: unknown field `{unknown[0]}`. Fields: {', '.join(sorted(FIELDS))}")

        name = _str(config, "name")
        kinds = _kinds(config)
        memory_root = _env(environ, "MEMORY_ROOT")
        access = (_env(environ, "MEMORY_ACCESS") or "read-write").lower()
        if access not in {"read-only", "read-write"}:
            raise ConfigError("MEMORY_ACCESS must be 'read-only' or 'read-write'")
        env_port = _env(environ, "MEMORY_PORT")
        return cls(
            vault=vault,
            name=name,
            instructions=_str(config, "instructions"),
            memory_dir=(
                Path(memory_root).expanduser().resolve()
                if memory_root
                else vault / _str(config, "memory_dir", "memory")
            ),
            state_dir=vault / _str(config, "state_dir", ".brain"),
            kinds=kinds,
            dated_kinds=_dated_kinds(config, kinds),
            host=_env(environ, "MEMORY_HOST") or "127.0.0.1",
            port=(
                _port(env_port, "MEMORY_PORT")
                if env_port
                else _port(config.get("port", DEFAULT_PORT), f"{CONFIG_FILENAME}: `port`")
            ),
            access=access,
            actor=_env(environ, "MEMORY_ACTOR") or f"{name}/local",
            human_id=_env(environ, "MEMORY_HUMAN_ID") or "owner",
            graph_enabled=_flag(environ, "MEMORY_GRAPH_ENABLED", True),
            graph_model=_env(environ, "MEMORY_GRAPH_MODEL") or _str(config, "graph_model", DEFAULT_GRAPH_MODEL),
            graph_small_model=(
                _env(environ, "MEMORY_GRAPH_SMALL_MODEL")
                or _str(config, "graph_small_model", DEFAULT_GRAPH_MODEL)
            ),
        )
