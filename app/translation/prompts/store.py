"""File-backed prompt store: built-in defaults overlaid with user-saved prompts.

Flat structure: every prompt is one subdir of ``root`` named by its id, holding the
prompt's components::

    <root>/<id>/meta.toml      # title, tags, notes
    <root>/<id>/system.md      # system prompt template ({{var}} placeholders)
    <root>/<id>/user.md        # user template (usually {{source_window}})

Built-ins (``templates.BUILTIN_PROMPTS``) are always present. A user dir with the same
id as a built-in overrides it, so the built-in default stays tunable. Ids are flat,
single-segment safe tokens (``translate_image_menu``); no nesting, no domain/stage.
"""
from __future__ import annotations

import re
import shutil
import tomllib
from functools import lru_cache
from pathlib import Path

from app.translation.prompts.templates import BUILTIN_PROMPTS
from app.translation.prompts.templates import DEFAULT_USER_TEMPLATE
from app.translation.prompts.templates import PromptEntry


_ID = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$")


class PromptStoreError(RuntimeError):
    """Base error for prompt-store operations."""


class PromptNotFoundError(PromptStoreError):
    pass


class PromptValidationError(PromptStoreError):
    pass


class PromptConflictError(PromptStoreError):
    pass


def normalize_prompt_id(prompt_id: str) -> str:
    raw = str(prompt_id or "").strip()
    if not _ID.match(raw):
        raise PromptValidationError(f"invalid prompt id: {prompt_id!r} (use letters, digits, '_' or '-')")
    return raw


class PromptStore:
    def __init__(self, root: Path) -> None:
        self._root = Path(root).resolve()

    def list(self) -> list[PromptEntry]:
        merged: dict[str, PromptEntry] = dict(BUILTIN_PROMPTS)
        if self._root.exists():
            for entry_dir in sorted(p for p in self._root.iterdir() if p.is_dir()):
                entry = self._load_dir(entry_dir)
                if entry is not None:
                    merged[entry.id] = entry
        return [merged[key] for key in sorted(merged)]

    def get(self, prompt_id: str) -> PromptEntry:
        normalized = normalize_prompt_id(prompt_id)
        entry = self._load_dir(self._dir_for(normalized))
        if entry is not None:
            return entry
        builtin = BUILTIN_PROMPTS.get(normalized)
        if builtin is not None:
            return builtin
        raise PromptNotFoundError(f"prompt not found: {normalized}")

    def create(self, entry: PromptEntry) -> PromptEntry:
        normalized = normalize_prompt_id(entry.id)
        if self._dir_for(normalized).exists() or normalized in BUILTIN_PROMPTS:
            raise PromptConflictError(f"prompt already exists: {normalized}")
        return self._write(normalized, entry)

    def update(self, prompt_id: str, entry: PromptEntry) -> PromptEntry:
        normalized = normalize_prompt_id(prompt_id)
        # Update is not an upsert: a typo'd id must 404, not silently create a new prompt.
        # A builtin id is updatable — the disk entry it writes is the deliberate override.
        if not self._dir_for(normalized).exists() and normalized not in BUILTIN_PROMPTS:
            raise PromptNotFoundError(f"prompt not found: {normalized}")
        return self._write(normalized, entry)

    def delete(self, prompt_id: str) -> None:
        normalized = normalize_prompt_id(prompt_id)
        entry_dir = self._dir_for(normalized)
        if not entry_dir.exists():
            if normalized in BUILTIN_PROMPTS:
                raise PromptValidationError(f"cannot delete built-in prompt: {normalized}")
            raise PromptNotFoundError(f"prompt not found: {normalized}")
        shutil.rmtree(entry_dir)

    def _write(self, normalized: str, entry: PromptEntry) -> PromptEntry:
        if not str(entry.system or "").strip():
            raise PromptValidationError("prompt system text must not be empty")
        record = PromptEntry(
            id=normalized,
            system=str(entry.system),
            user=str(entry.user or DEFAULT_USER_TEMPLATE),
            tags=list(entry.tags or []),
            builtin=False,
        )
        entry_dir = self._dir_for(normalized)
        entry_dir.mkdir(parents=True, exist_ok=True)
        # The API round-trips only ``tags``, but hand-written meta.toml files carry keys the code
        # does not model (title, notes). Carry those through the rewrite instead of wiping them.
        extra_meta: dict = {}
        meta_path = entry_dir / "meta.toml"
        if meta_path.exists():
            try:
                extra_meta = tomllib.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, tomllib.TOMLDecodeError):
                extra_meta = {}
        (entry_dir / "system.md").write_text(record.system, encoding="utf-8")
        (entry_dir / "user.md").write_text(record.user, encoding="utf-8")
        meta_path.write_text(_render_meta_toml(record, extra=extra_meta), encoding="utf-8")
        return record

    def _dir_for(self, normalized: str) -> Path:
        entry_dir = (self._root / normalized).resolve()
        entry_dir.relative_to(self._root)  # reject traversal
        return entry_dir

    def _load_dir(self, entry_dir: Path) -> PromptEntry | None:
        system_path = entry_dir / "system.md"
        if not system_path.exists():
            return None
        try:
            prompt_id = normalize_prompt_id(entry_dir.name)
        except PromptValidationError:
            return None
        meta: dict = {}
        meta_path = entry_dir / "meta.toml"
        if meta_path.exists():
            try:
                meta = tomllib.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, tomllib.TOMLDecodeError):
                meta = {}
        user_path = entry_dir / "user.md"
        return PromptEntry(
            id=prompt_id,
            system=system_path.read_text(encoding="utf-8"),
            user=user_path.read_text(encoding="utf-8") if user_path.exists() else DEFAULT_USER_TEMPLATE,
            tags=[str(tag) for tag in (meta.get("tags") or [])],
            builtin=False,
        )


def _toml_str(value: str) -> str:
    escaped = (
        str(value)
        .replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\t", "\\t")
    )
    return f'"{escaped}"'


def _render_meta_toml(entry: PromptEntry, extra: dict | None = None) -> str:
    lines = [
        f"{key} = {_toml_str(value)}"
        for key, value in (extra or {}).items()
        if key != "tags" and isinstance(value, str)
    ]
    tags = ", ".join(_toml_str(tag) for tag in entry.tags)
    lines.append(f"tags = [{tags}]")
    return "\n".join(lines) + "\n"


@lru_cache(maxsize=8)
def get_prompt_store(root: str) -> PromptStore:
    return PromptStore(Path(root))


def store_for(prompts_root: str) -> PromptStore:
    """Resolve ``settings.service.prompts_root`` (relative paths against the repo root)
    to a single shared, cached store instance."""
    root = Path(prompts_root or "data/prompts")
    if not root.is_absolute():
        root = Path(__file__).resolve().parents[3] / root
    return get_prompt_store(str(root.resolve()))
