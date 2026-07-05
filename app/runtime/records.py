from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.util import iso_utc
from app.core.util import parse_utc_unix
from app.core.util import safe_token


@dataclass
class RequestRecord:
    request_id: str
    payload_hash: str
    request: dict[str, Any]
    task: str
    state: str
    submitted_at_utc: str
    submitted_mono: float
    started_at_utc: str | None = None
    started_mono: float | None = None
    finished_at_utc: str | None = None
    finished_mono: float | None = None
    stage: str | None = None
    timings: dict[str, float] | None = None
    response: dict[str, Any] | None = None
    error: dict[str, Any] | None = None


class RequestStore:
    def __init__(
        self,
        *,
        work_root: Path,
        records_max: int,
        records_ttl_s: dict[str, int],
    ) -> None:
        self.work_root = work_root
        self.records: dict[str, RequestRecord] = {}
        self._records_max = int(records_max)
        self._records_ttl_s = dict(records_ttl_s)

    def get(self, request_id: str) -> RequestRecord | None:
        return self.records.get(str(request_id))

    def set(self, rec: RequestRecord) -> None:
        self.records[str(rec.request_id)] = rec

    def values(self):
        return self.records.values()

    def mark_terminal(
        self,
        rec: RequestRecord,
        *,
        state: str,
        error: dict[str, Any] | None = None,
        response: dict[str, Any] | None = None,
        stage: str | None = None,
    ) -> None:
        rec.state = str(state)
        rec.finished_at_utc = iso_utc()
        rec.finished_mono = time.monotonic()
        rec.stage = str(stage or state)
        rec.response = dict(response or {}) if response is not None else None
        rec.error = dict(error or {}) if error is not None else None
        if rec.started_mono is not None and rec.finished_mono is not None:
            timings = dict(rec.timings or {})
            timings["pool_run_wall_s"] = round(max(0.0, rec.finished_mono - rec.started_mono), 6)
            rec.timings = timings

    def to_lifecycle(self, rec: RequestRecord, *, queue_position: int | None) -> dict[str, Any]:
        return {
            "request_id": rec.request_id,
            "state": rec.state,
            "task": rec.task,
            "queue_position": queue_position,
            "submitted_at_utc": rec.submitted_at_utc,
            "started_at_utc": rec.started_at_utc,
            "finished_at_utc": rec.finished_at_utc,
            "stage": rec.stage,
            "timings": dict(rec.timings or {}),
            "response": rec.response,
            "error": rec.error,
        }

    def prune(self) -> list[Path]:
        """Drop expired/overflowing terminal records; RETURNS their artifact dirs instead of
        deleting them. Deleting here would put ``rmtree`` (a TTL boundary can expire dozens of
        record dirs full of debug PNGs) inside every submit/poll, under the runtime lock and on
        the event loop — the caller hands the paths to ``remove_dirs`` off-loop, lock released."""
        now_unix = time.time()
        removable: list[str] = []
        for rid, rec in self.records.items():
            state = str(rec.state or "").strip().lower()
            if state not in {"completed", "failed", "cancelled"}:
                continue
            ttl_s = int(self._records_ttl_s.get(state, 0))
            ref_unix = parse_utc_unix(rec.finished_at_utc) or parse_utc_unix(rec.submitted_at_utc)
            if ttl_s > 0 and ref_unix is not None and (now_unix - ref_unix) >= ttl_s:
                removable.append(str(rid))
        overflow = max(0, len(self.records) - int(self._records_max))
        if overflow:
            terminal_rows: list[tuple[float, str]] = []
            for rid, rec in self.records.items():
                if str(rec.state) not in {"completed", "failed", "cancelled"}:
                    continue
                ref_unix = parse_utc_unix(rec.finished_at_utc) or parse_utc_unix(rec.submitted_at_utc) or now_unix
                terminal_rows.append((float(ref_unix), str(rid)))
            terminal_rows.sort(key=lambda row: row[0])
            removable.extend(rid for _ref, rid in terminal_rows[:overflow])
        stale_dirs: list[Path] = []
        for rid in set(removable):
            self.records.pop(rid, None)
            stale_dirs.extend(self._request_dirs(rid))
        return stale_dirs

    def _request_dirs(self, request_id: str) -> list[Path]:
        roots = [
            (self.work_root / safe_token(request_id)).resolve(),
            (self.work_root / "_uploads" / safe_token(request_id)).resolve(),
        ]
        out: list[Path] = []
        for root in roots:
            try:
                root.relative_to(self.work_root)
            except ValueError:
                continue
            out.append(root)
        return out


def remove_dirs(paths: list[Path]) -> None:
    for path in paths:
        shutil.rmtree(path, ignore_errors=True)
