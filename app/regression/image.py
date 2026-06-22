"""The exact pixels the pipeline renders on: a testset file after app.main's canonical ingest.

Both capture and replay go through this so the replayed render matches the snapshot, and the
``image_sha256`` identity is the canonical bytes (not the raw upload). Script-support only — the
core package (fixture/replay/snapshot/compare) does not import the web layer.
"""
from __future__ import annotations

from pathlib import Path

_MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}


def canonical_bytes(path: Path) -> bytes:
    from app.main import _canonical_image_bytes  # lazy: keep FastAPI out of the import graph

    mime = _MIME.get(path.suffix.lower(), "application/octet-stream")
    return _canonical_image_bytes(path.read_bytes(), mime)
