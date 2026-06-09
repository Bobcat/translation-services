from __future__ import annotations

import asyncio
import json
import time
import uuid
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from app.core.config import AppSettings
from app.core.schemas import RequestPayload
from app.tasks.translate_image import run_translate_image_pipeline
from app.core.util import iso_utc
from app.core.util import safe_token

from .records import RequestRecord
from .records import RequestStore
from .scheduler import RequestScheduler


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _json_hash(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


class RequestRuntime:
    def __init__(self, *, settings: AppSettings) -> None:
        self._settings = settings
        work_root = Path(settings.service.work_root).expanduser()
        if not work_root.is_absolute():
            work_root = (_repo_root() / work_root).resolve()
        self.work_root = work_root.resolve()
        self.work_root.mkdir(parents=True, exist_ok=True)
        self._queue_limit = int(settings.scheduler.queue_limit)
        self._scheduler = RequestScheduler(runner_slots=settings.scheduler.runner_slots)
        self._record_store = RequestStore(
            work_root=self.work_root,
            records_max=settings.scheduler.records_max,
            records_ttl_s=settings.scheduler.records_ttl_s,
        )
        self._lock = asyncio.Lock()
        self._cond = asyncio.Condition(self._lock)
        self._tasks: list[asyncio.Task[None]] = []
        self._stopping = False
        self._events: list[dict[str, Any]] = []
        self._next_event_seq = 1

    async def start(self) -> None:
        async with self._lock:
            if self._tasks:
                return
            self._stopping = False
            for idx in range(self._settings.scheduler.runner_slots):
                task = asyncio.create_task(self._runner_loop(idx), name=f"request-runtime-runner-{idx}")
                self._tasks.append(task)

    async def stop(self) -> None:
        async with self._lock:
            self._stopping = True
            self._cond.notify_all()
            tasks = list(self._tasks)
            self._tasks.clear()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def submit(self, raw_payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        raw_payload = dict(raw_payload or {})
        try:
            payload = RequestPayload(**raw_payload)
        except ValidationError as exc:
            return 400, {
                "code": "REQUEST_INVALID",
                "message": "request_json is invalid",
                "retryable": False,
                "details": {"errors": exc.errors()},
            }

        image = dict(raw_payload.get("image") or {})
        image_path = str(image.get("local_path") or "").strip()
        image_mime_type = str(image.get("mime_type") or "").strip()
        if not image_path:
            return 400, {
                "code": "REQUEST_INPUT_REQUIRED",
                "message": "uploaded image_file is required",
                "retryable": False,
            }
        request_id = str(payload.request_id or f"req_{uuid.uuid4().hex}").strip()

        prepared = payload.model_dump()
        prepared["request_id"] = request_id
        prepared["image"] = image
        payload_hash = _json_hash(prepared)

        async with self._lock:
            self._record_store.prune()
            existing = self._record_store.get(request_id)
            if existing is not None:
                if existing.payload_hash != payload_hash:
                    return 409, {
                        "code": "REQUEST_ID_CONFLICT",
                        "message": "request_id already exists with different payload",
                        "retryable": False,
                        "details": {"request_id": request_id},
                    }
                return 200, self._to_lifecycle(existing)

            if self._scheduler.depth() >= self._queue_limit:
                return 429, {
                    "code": "REQUEST_QUEUE_FULL",
                    "message": "queue depth limit reached",
                    "retryable": True,
                    "details": {
                        "queue_depth": int(self._scheduler.depth()),
                        "queue_limit": int(self._queue_limit),
                    },
                }

            submitted_mono = time.monotonic()
            rec = RequestRecord(
                request_id=request_id,
                payload_hash=payload_hash,
                request=prepared,
                task=payload.task,
                state="queued",
                submitted_at_utc=iso_utc(),
                submitted_mono=round(float(submitted_mono), 6),
                stage="queued",
            )
            self._record_store.set(rec)
            self._scheduler.enqueue(rec)
            self._cond.notify_all()
            return 202, self._to_lifecycle(rec)

    async def get_request(self, request_id: str) -> tuple[int, dict[str, Any]]:
        rid = str(request_id or "").strip()
        async with self._lock:
            self._record_store.prune()
            rec = self._record_store.get(rid)
            if rec is None:
                return 404, {
                    "code": "REQUEST_NOT_FOUND",
                    "message": "request_id not found",
                    "retryable": False,
                    "details": {"request_id": rid},
                }
            return 200, self._to_lifecycle(rec)

    async def cancel(self, request_id: str) -> tuple[int, dict[str, Any]]:
        rid = str(request_id or "").strip()
        async with self._lock:
            rec = self._record_store.get(rid)
            if rec is None:
                return 404, {
                    "code": "REQUEST_NOT_FOUND",
                    "message": "request_id not found",
                    "retryable": False,
                    "details": {"request_id": rid},
                }
            if rec.state == "queued":
                self._scheduler.remove(rid)
                self._record_store.mark_terminal(rec, state="cancelled", stage="cancelled")
                self._emit_completion(rec, event="request_cancelled")
            elif rec.state == "running":
                rec.state = "cancel_requested"
                rec.stage = "cancel_requested"
            return 200, self._to_lifecycle(rec)

    async def completions(self, *, since_seq: int = 0, limit: int = 100) -> tuple[int, dict[str, Any]]:
        safe_since = max(0, int(since_seq))
        safe_limit = max(1, min(1000, int(limit)))
        async with self._lock:
            events = [event for event in self._events if int(event["seq"]) > safe_since]
            events = events[:safe_limit]
            next_seq = max([safe_since] + [int(event["seq"]) for event in events])
            return 200, {"events": events, "next_seq": next_seq}

    async def status(self) -> dict[str, Any]:
        async with self._lock:
            queue_depth = self._scheduler.depth()
            running = sum(1 for rec in self._record_store.values() if rec.state in {"running", "cancel_requested"})
            return {
                "runner_slots": int(self._settings.scheduler.runner_slots),
                "running": int(running),
                "queue_depth": int(queue_depth),
                "records": {"count": int(len(self._record_store.records))},
            }

    async def artifact_path(self, *, request_id: str, artifact_name: str) -> tuple[int, dict[str, Any]]:
        status_code, body = await self.get_request(request_id)
        if status_code != 200:
            return status_code, body
        response = dict(body.get("response") or {})
        artifacts = dict(response.get("artifacts") or {})
        artifact = dict(artifacts.get(str(artifact_name)) or {})
        path_value = str(artifact.get("path") or "").strip()
        if not path_value:
            return 404, {
                "code": "REQUEST_ARTIFACT_NOT_FOUND",
                "message": "artifact not found",
                "retryable": False,
                "details": {"request_id": request_id, "artifact": artifact_name},
            }
        artifact_path = Path(path_value).resolve()
        try:
            artifact_path.relative_to(self.work_root)
        except ValueError:
            return 400, {
                "code": "REQUEST_ARTIFACT_PATH_INVALID",
                "message": "artifact path is outside work_root",
                "retryable": False,
            }
        if not artifact_path.exists() or not artifact_path.is_file():
            return 404, {
                "code": "REQUEST_ARTIFACT_NOT_FOUND",
                "message": "artifact file is missing",
                "retryable": False,
                "details": {"request_id": request_id, "artifact": artifact_name},
            }
        return 200, {"path": str(artifact_path), "mime_type": str(artifact.get("mime_type") or "application/octet-stream")}

    async def _runner_loop(self, slot_idx: int) -> None:
        while True:
            rid = await self._dequeue_next_request_id()
            if rid is None:
                return
            await self._run_request(slot_idx=slot_idx, request_id=rid)

    async def _dequeue_next_request_id(self) -> str | None:
        async with self._lock:
            while True:
                if self._stopping:
                    return None
                rid = self._scheduler.dequeue_next(records=self._record_store.records)
                if rid is not None:
                    rec = self._record_store.get(rid)
                    if rec is None or rec.state != "queued":
                        continue
                    rec.state = "running"
                    rec.stage = "running"
                    rec.started_at_utc = iso_utc()
                    rec.started_mono = time.monotonic()
                    queue_wait_s = max(0.0, rec.started_mono - rec.submitted_mono)
                    rec.timings = {"pool_queue_wait_s": round(float(queue_wait_s), 6)}
                    return rid
                await self._cond.wait()

    async def _run_request(self, *, slot_idx: int, request_id: str) -> None:
        del slot_idx
        async with self._lock:
            rec = self._record_store.get(request_id)
            if rec is None:
                return
            request = dict(rec.request)
            rec.stage = self._stage_for_task(str(rec.task))

        try:
            response = await asyncio.to_thread(self._process_request, request_id, request)
        except Exception as exc:
            async with self._lock:
                rec = self._record_store.get(request_id)
                if rec is None:
                    return
                self._record_store.mark_terminal(
                    rec,
                    state="failed",
                    stage="failed",
                    error={
                        "code": "REQUEST_FAILED",
                        "message": str(exc).strip() or exc.__class__.__name__,
                    },
                )
                self._emit_completion(rec, event="request_failed")
                self._cond.notify_all()
            return

        async with self._lock:
            rec = self._record_store.get(request_id)
            if rec is None:
                return
            if rec.state == "cancel_requested":
                self._record_store.mark_terminal(rec, state="cancelled", stage="cancelled")
                self._emit_completion(rec, event="request_cancelled")
            else:
                self._record_store.mark_terminal(rec, state="completed", stage="completed", response=response)
                self._emit_completion(rec, event="request_completed")
            self._cond.notify_all()

    def _process_request(self, request_id: str, request: dict[str, Any]) -> dict[str, Any]:
        return self._process_translate_image_request(request_id, request)

    def _process_translate_image_request(self, request_id: str, request: dict[str, Any]) -> dict[str, Any]:
        image = dict(request.get("image") or {})
        input_path = Path(str(image.get("local_path") or "")).resolve()
        input_mime_type = str(image.get("mime_type") or "application/octet-stream")
        if not input_path.exists() or not input_path.is_file():
            raise RuntimeError("input image file is missing")

        result = run_translate_image_pipeline(
            settings=self._settings,
            input_path=input_path,
            input_mime_type=input_mime_type,
            request=request,
        )

        job_root = (self.work_root / safe_token(request_id)).resolve()
        job_root.mkdir(parents=True, exist_ok=True)
        output_path = (job_root / "output.png").resolve()
        output_path.write_bytes(result.image)
        debug_overlay_path = (job_root / "debug_overlay.png").resolve()
        if result.debug_image is not None:
            debug_overlay_path.write_bytes(result.debug_image)
        rectified_debug_path = (job_root / "rectified_debug.png").resolve()
        if result.rectified_debug_image is not None:
            rectified_debug_path.write_bytes(result.rectified_debug_image)
        projected_overlay_debug_path = (job_root / "projected_overlay_debug.png").resolve()
        if result.projected_overlay_debug_image is not None:
            projected_overlay_debug_path.write_bytes(result.projected_overlay_debug_image)
        grouping_overlay_debug_path = (job_root / "grouping_overlay_debug.png").resolve()
        if result.grouping_overlay_debug_image is not None:
            grouping_overlay_debug_path.write_bytes(result.grouping_overlay_debug_image)
        rendered_path = (job_root / "rendered.png").resolve()
        if result.rendered_image is not None:
            rendered_path.write_bytes(result.rendered_image)
        segments_path = (job_root / "segments.json").resolve()
        segments_path.write_text(
            json.dumps({"segments": result.segments}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        debug = result.debug or {}
        for name in ("request", "grouping", "translation"):
            if debug.get(name) is not None:
                (job_root / f"{name}.json").write_text(
                    json.dumps(debug[name], ensure_ascii=False, indent=2), encoding="utf-8"
                )
        calls = debug.get("llm_calls") or []
        if calls:
            calls_dir = (job_root / "llm_calls").resolve()
            calls_dir.mkdir(parents=True, exist_ok=True)
            for index, call in enumerate(calls, start=1):
                role = str(call.get("role") or "call")
                safe = "".join(c if (c.isalnum() or c in "._-") else "_" for c in role)[:60]
                (calls_dir / f"{index:02d}_{safe}.json").write_text(
                    json.dumps(call, ensure_ascii=False, indent=2), encoding="utf-8"
                )
        artifacts = {
            "input": {
                "path": str(input_path),
                "mime_type": input_mime_type,
            },
            "output": {
                "path": str(output_path),
                "mime_type": result.mime_type,
            },
            "segments": {
                "path": str(segments_path),
                "mime_type": "application/json",
            },
        }
        if result.debug_image is not None:
            artifacts["debug_overlay"] = {
                "path": str(debug_overlay_path),
                "mime_type": result.debug_mime_type,
            }
        if result.rectified_debug_image is not None:
            artifacts["rectified_debug"] = {
                "path": str(rectified_debug_path),
                "mime_type": result.rectified_debug_mime_type,
            }
        if result.projected_overlay_debug_image is not None:
            artifacts["projected_overlay_debug"] = {
                "path": str(projected_overlay_debug_path),
                "mime_type": result.projected_overlay_debug_mime_type,
            }
        if result.grouping_overlay_debug_image is not None:
            artifacts["grouping_overlay_debug"] = {
                "path": str(grouping_overlay_debug_path),
                "mime_type": result.grouping_overlay_debug_mime_type,
            }
        if result.rendered_image is not None:
            artifacts["rendered"] = {
                "path": str(rendered_path),
                "mime_type": result.rendered_mime_type,
            }
        response = {
            "task": request["task"],
            "artifacts": artifacts,
            "segments": result.segments,
            "metadata": dict(result.metadata),
            "metrics": dict(result.metrics),
        }
        if result.ocr is not None:
            response["ocr"] = dict(result.ocr)
        return response

    def _to_lifecycle(self, rec: RequestRecord) -> dict[str, Any]:
        return self._record_store.to_lifecycle(rec, queue_position=self._scheduler.queue_position(rec))

    def _emit_completion(self, rec: RequestRecord, *, event: str) -> None:
        item = {
            "seq": int(self._next_event_seq),
            "event": str(event),
            "request_id": rec.request_id,
            "state": rec.state,
            "task": rec.task,
            "submitted_at_utc": rec.submitted_at_utc,
            "finished_at_utc": rec.finished_at_utc,
            "response": rec.response,
            "error": rec.error,
        }
        self._next_event_seq += 1
        self._events.append(item)
        self._events = self._events[-1000:]

    def _stage_for_task(self, task: str) -> str:
        if task == "translate_image":
            return "ocr_inspect"
        return "running"
