"""Shared HTTP-surface helpers: the one error dialect and the upload ceiling.

Lives outside ``app.main`` so extracted route modules can use them without importing the
composition root (which imports them back — a cycle otherwise).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi.responses import JSONResponse

# Upload ceiling: a phone photo tops out around 10-15 MB; anything past this is not a
# translation job and would otherwise be read into memory whole (plus a canonical re-encode).
MAX_UPLOAD_BYTES = 32 * 1024 * 1024


def error_response(
    status_code: int,
    *,
    code: str,
    message: str,
    retryable: bool | None = None,
    details: dict[str, Any] | None = None,
) -> JSONResponse:
    payload: dict[str, Any] = {"code": str(code), "message": str(message)}
    if retryable is not None:
        payload["retryable"] = bool(retryable)
    if details:
        payload["details"] = dict(details)
    return JSONResponse(status_code=int(status_code), content=payload)


def repo_root_dir() -> Path:
    return Path(__file__).resolve().parents[2]
