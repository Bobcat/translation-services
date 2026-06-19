"""Composition root (HTTP entry point: ``app.main:app``).

Wires the service together and exposes the API:

- ``app.core``        settings, request/response schemas, small helpers
- ``app.runtime``        job execution: FIFO queue, runner loop, lifecycle store
- ``app.tasks``       feature pipelines (one module per task; the readable flows)
- ``app.ocr``         OCR component (paddle backend, merge, overlay, doc-unwarp)
- ``app.translation`` translation component (language-pair routing)

``create_app`` builds settings -> runtime and registers the ``/v1/*`` routes;
``translate_image`` requests are handed to ``app.tasks.translate_image``.
"""
from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from io import BytesIO
import json
from pathlib import Path
from typing import Any
import uuid

from fastapi import Body
from fastapi import FastAPI
from fastapi import File
from fastapi import Form
from fastapi import Query
from fastapi import UploadFile
from fastapi.responses import FileResponse
from fastapi.responses import JSONResponse

from pydantic import BaseModel
from pydantic import Field

from app.core.config import load_settings
from app.runtime.service import RequestRuntime
from app.core.schemas import CompletionsEnvelope
from app.core.schemas import RequestLifecycle
from app.core.schemas import RequestSubmitEnvelope
from app.core.util import safe_token
from app.translation.prompts import PromptConflictError
from app.translation.prompts import PromptEntry
from app.translation.prompts import PromptNotFoundError
from app.translation.prompts import PromptValidationError
from app.translation.prompts import store_for
from app.translation.prompts.templates import DEFAULT_USER_TEMPLATE


SUPPORTED_IMAGE_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}


class PromptBody(BaseModel):
    system: str
    user: str = DEFAULT_USER_TEMPLATE
    tags: list[str] = Field(default_factory=list)

    def to_entry(self, prompt_id: str) -> PromptEntry:
        return PromptEntry(
            id=prompt_id,
            system=self.system,
            user=self.user or DEFAULT_USER_TEMPLATE,
            tags=list(self.tags),
        )


class PromptCreateBody(PromptBody):
    id: str


def create_app(settings_path: str | Path | None = None) -> FastAPI:
    settings = load_settings(settings_path)
    runtime = RequestRuntime(settings=settings)
    prompt_store = store_for(settings.service.prompts_root)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        await runtime.start()
        try:
            yield
        finally:
            await runtime.stop()

    app = FastAPI(title="Translation Services API", lifespan=lifespan)

    @app.post("/v1/requests", response_model=RequestSubmitEnvelope)
    async def submit_request(
        request_json: str = Form(...),
        image_file: UploadFile = File(...),
    ) -> JSONResponse:
        try:
            parsed = json.loads(request_json)
        except Exception as exc:
            return _error(
                400,
                code="REQUEST_JSON_INVALID",
                message="request_json must be a JSON object",
                retryable=False,
            )
        if not isinstance(parsed, dict):
            return _error(
                400,
                code="REQUEST_JSON_INVALID",
                message="request_json must be a JSON object",
                retryable=False,
            )

        mime_type = str(image_file.content_type or "").strip().lower()
        if mime_type not in SUPPORTED_IMAGE_MIME_TYPES:
            return _error(
                400,
                code="REQUEST_MIME_TYPE_UNSUPPORTED",
                message="image_file must be image/jpeg, image/png, or image/webp",
                retryable=False,
                details={"mime_type": mime_type or "unknown"},
            )
        request_id = str(parsed.get("request_id") or "").strip()
        if not request_id:
            stem = safe_token(Path(str(image_file.filename or "image")).stem, fallback="image")
            request_id = f"req_{stem}_{uuid.uuid4().hex}"
            parsed["request_id"] = request_id

        image_bytes = await image_file.read()
        if not image_bytes:
            return _error(
                400,
                code="REQUEST_EMPTY_INPUT",
                message="uploaded image_file is empty",
                retryable=False,
            )
        try:
            canonical_image_bytes = _canonical_image_bytes(image_bytes, mime_type)
        except ValueError as exc:
            return _error(
                400,
                code="REQUEST_INVALID_INPUT",
                message=str(exc),
                retryable=False,
            )
        upload_root = (runtime.work_root / "_uploads").resolve()
        upload_dir = (upload_root / safe_token(request_id)).resolve()
        try:
            upload_dir.relative_to(upload_root)
        except ValueError:
            return _error(
                400,
                code="REQUEST_UPLOAD_PATH_INVALID",
                message="invalid request_id upload path",
                retryable=False,
            )
        upload_dir.mkdir(parents=True, exist_ok=True)
        input_path = (upload_dir / f"input{_suffix_for_mime(mime_type)}").resolve()
        input_path.write_bytes(canonical_image_bytes)

        parsed["image"] = {
            "local_path": str(input_path),
            "mime_type": mime_type,
            "filename": str(image_file.filename or ""),
            "size_bytes": int(len(canonical_image_bytes)),
        }
        status_code, body = await runtime.submit(parsed)
        return JSONResponse(status_code=int(status_code), content=body)

    @app.post("/v1/requests/{source_request_id}/retranslate", response_model=RequestSubmitEnvelope)
    async def retranslate_request(
        source_request_id: str,
        body: dict[str, Any] = Body(default_factory=dict),
    ) -> JSONResponse:
        status_code, payload = await runtime.submit_retranslate(
            source_request_id=source_request_id, body=dict(body or {})
        )
        return JSONResponse(status_code=int(status_code), content=payload)

    @app.get("/v1/prompts")
    async def list_prompts() -> JSONResponse:
        return JSONResponse(status_code=200, content={"prompts": [e.to_dict() for e in prompt_store.list()]})

    @app.post("/v1/prompts")
    async def create_prompt(body: PromptCreateBody) -> JSONResponse:
        try:
            entry = prompt_store.create(body.to_entry(body.id))
        except (PromptConflictError, PromptValidationError, PromptNotFoundError) as exc:
            return _prompt_error(exc)
        return JSONResponse(status_code=200, content=entry.to_dict())

    @app.get("/v1/prompts/{prompt_id:path}")
    async def get_prompt(prompt_id: str) -> JSONResponse:
        try:
            entry = prompt_store.get(prompt_id)
        except (PromptValidationError, PromptNotFoundError) as exc:
            return _prompt_error(exc)
        return JSONResponse(status_code=200, content=entry.to_dict())

    @app.put("/v1/prompts/{prompt_id:path}")
    async def update_prompt(prompt_id: str, body: PromptBody) -> JSONResponse:
        try:
            entry = prompt_store.update(prompt_id, body.to_entry(prompt_id))
        except (PromptValidationError, PromptNotFoundError) as exc:
            return _prompt_error(exc)
        return JSONResponse(status_code=200, content=entry.to_dict())

    @app.delete("/v1/prompts/{prompt_id:path}")
    async def delete_prompt(prompt_id: str) -> JSONResponse:
        try:
            prompt_store.delete(prompt_id)
        except (PromptValidationError, PromptNotFoundError) as exc:
            return _prompt_error(exc)
        return JSONResponse(status_code=200, content={"id": prompt_id, "deleted": True})

    @app.get("/v1/requests/{request_id}", response_model=RequestLifecycle)
    async def get_request(request_id: str) -> JSONResponse:
        status_code, body = await runtime.get_request(request_id)
        return JSONResponse(status_code=int(status_code), content=body)

    @app.post("/v1/requests/{request_id}/cancel", response_model=RequestLifecycle)
    async def cancel_request(request_id: str) -> JSONResponse:
        status_code, body = await runtime.cancel(request_id)
        return JSONResponse(status_code=int(status_code), content=body)

    @app.get("/v1/requests/{request_id}/artifacts/{artifact_name}")
    async def get_artifact(request_id: str, artifact_name: str):
        status_code, body = await runtime.artifact_path(request_id=request_id, artifact_name=artifact_name)
        if int(status_code) != 200:
            return JSONResponse(status_code=int(status_code), content=body)
        path = Path(str(body["path"]))
        return FileResponse(path=str(path), media_type=str(body["mime_type"]), filename=path.name)

    @app.get("/v1/completions", response_model=CompletionsEnvelope)
    async def get_completions(
        since_seq: int = Query(default=0),
        limit: int = Query(default=100),
    ) -> JSONResponse:
        status_code, body = await runtime.completions(since_seq=since_seq, limit=limit)
        return JSONResponse(status_code=int(status_code), content=body)

    @app.get("/v1/status")
    async def get_status() -> dict[str, Any]:
        return await runtime.status()

    return app


def _error(
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


def _prompt_error(exc: Exception) -> JSONResponse:
    if isinstance(exc, PromptNotFoundError):
        return _error(404, code="PROMPT_NOT_FOUND", message=str(exc), retryable=False)
    if isinstance(exc, PromptConflictError):
        return _error(409, code="PROMPT_CONFLICT", message=str(exc), retryable=False)
    return _error(400, code="PROMPT_INVALID", message=str(exc), retryable=False)


def _suffix_for_mime(mime_type: str) -> str:
    normalized = str(mime_type or "").strip().lower()
    if normalized == "image/png":
        return ".png"
    if normalized == "image/webp":
        return ".webp"
    if normalized in {"image/jpeg", "image/jpg"}:
        return ".jpg"
    return ".bin"


def _canonical_image_bytes(image_bytes: bytes, mime_type: str) -> bytes:
    from PIL import Image
    from PIL import ImageOps
    from PIL import UnidentifiedImageError

    try:
        with Image.open(BytesIO(image_bytes)) as original:
            image_format = _image_format_for_mime(mime_type)
            # Pass the upload through untouched when it is already in canonical form: the stored
            # format matches the target, there is no EXIF orientation to bake in, and the mode is
            # one OCR / the renderer read directly. Re-encoding an already-fine JPEG only stacks a
            # lossy compression generation that perturbs OCR (and inflates the file) for no gain —
            # the normalize path below runs only when something actually needs fixing (a rotated
            # phone photo, an odd mode, a wrong container).
            orientation = original.getexif().get(0x0112)  # 274; values 2..8 need a transpose
            ok_modes = {"RGB", "L"} if image_format == "JPEG" else {"RGB", "RGBA", "L", "LA", "P"}
            if (
                (original.format or "").upper() == image_format
                and orientation in (None, 0, 1)
                and original.mode in ok_modes
            ):
                return image_bytes
            image = ImageOps.exif_transpose(original)
            out = BytesIO()
            save_kwargs: dict[str, object] = {}
            if image_format == "JPEG":
                if image.mode not in {"RGB", "L"}:
                    image = image.convert("RGB")
                save_kwargs["quality"] = 95
            image.save(out, format=image_format, **save_kwargs)
            return out.getvalue()
    except UnidentifiedImageError as exc:
        raise ValueError("uploaded image_file could not be decoded") from exc
    except OSError as exc:
        raise ValueError("uploaded image_file could not be normalized") from exc


def _image_format_for_mime(mime_type: str) -> str:
    normalized = str(mime_type or "").strip().lower()
    if normalized == "image/png":
        return "PNG"
    if normalized == "image/webp":
        return "WEBP"
    if normalized in {"image/jpeg", "image/jpg"}:
        return "JPEG"
    raise ValueError("uploaded image_file has unsupported image format")


app = create_app()
