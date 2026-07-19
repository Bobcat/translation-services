"""``/v1/requests*`` + lifecycle queries — the job-submission surface.

Intake validates and canonicalises the upload up front (an image is EXIF-transposed and
re-encoded only when needed; a PDF is validated + censused but stored verbatim), writes it
under ``work_root/_uploads/<request_id>/`` and hands the parsed request to the runtime.
"""
from __future__ import annotations

import json
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any

import anyio

from fastapi import Body
from fastapi import FastAPI
from fastapi import File
from fastapi import Form
from fastapi import Query
from fastapi import UploadFile
from fastapi.responses import FileResponse
from fastapi.responses import JSONResponse

from app.core.config import AppSettings
from app.core.schemas import CompletionsEnvelope
from app.core.schemas import RequestLifecycle
from app.core.schemas import RequestSubmitEnvelope
from app.core.util import safe_token
from app.pdf.document import PdfValidationError
from app.pdf.document import profile_pdf
from app.routes.common import MAX_UPLOAD_BYTES
from app.routes.common import error_response as _error
from app.runtime.service import RequestRuntime

SUPPORTED_IMAGE_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}
SUPPORTED_DOCUMENT_MIME_TYPES = {"application/pdf"}


def register(app: FastAPI, *, settings: AppSettings, runtime: RequestRuntime) -> None:
    @app.post("/v1/requests", response_model=RequestSubmitEnvelope)
    async def submit_request(
        request_json: str = Form(...),
        image_file: UploadFile | None = File(default=None),
        document_file: UploadFile | None = File(default=None),
    ) -> JSONResponse:
        try:
            parsed = json.loads(request_json)
        except Exception:
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

        # A document task uploads its file under ``document_file``; the image tasks keep
        # ``image_file``. Branch on the declared task so each path validates its own input.
        if str(parsed.get("task") or "").strip() == "translate_pdf":
            return await _submit_pdf_request(parsed, document_file)
        if image_file is None:
            return _error(
                400,
                code="REQUEST_INPUT_REQUIRED",
                message="image_file is required",
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
        if len(image_bytes) > MAX_UPLOAD_BYTES:
            return _error(
                413,
                code="REQUEST_INPUT_TOO_LARGE",
                message=f"uploaded image_file exceeds {MAX_UPLOAD_BYTES} bytes",
                retryable=False,
            )
        try:
            # PIL decode + re-encode of a full upload takes 100s of ms — off the event loop, or
            # every concurrent user's polls stall behind it.
            canonical_image_bytes = await anyio.to_thread.run_sync(
                _canonical_image_bytes, image_bytes, mime_type
            )
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
        # A UNIQUE filename per submission: the upload is written before ``runtime.submit``
        # decides, so a conflicting/duplicate resubmission with the same request_id must not
        # overwrite the file an accepted record's ``image.local_path`` already points at.
        input_path = (upload_dir / f"input-{uuid.uuid4().hex[:8]}{_suffix_for_mime(mime_type)}").resolve()
        input_path.write_bytes(canonical_image_bytes)

        parsed["image"] = {
            "local_path": str(input_path),
            "mime_type": mime_type,
            "filename": str(image_file.filename or ""),
            "size_bytes": int(len(canonical_image_bytes)),
        }
        status_code, body = await runtime.submit(parsed)
        if int(status_code) != 202:  # rejected or deduped -> this submission's file is unowned
            await anyio.to_thread.run_sync(lambda: input_path.unlink(missing_ok=True))
        return JSONResponse(status_code=int(status_code), content=body)

    async def _submit_pdf_request(parsed: dict[str, Any], document_file: UploadFile | None) -> JSONResponse:
        """The translate_pdf intake: validate + census the PDF up front (parse errors,
        encryption and the page cap reject at submit time, not minutes into the run),
        store the bytes VERBATIM (a PDF is never re-encoded at intake), and inject
        ``parsed["document"]`` — the document counterpart of ``parsed["image"]``."""
        if document_file is None:
            return _error(
                400,
                code="REQUEST_INPUT_REQUIRED",
                message="document_file is required for task translate_pdf",
                retryable=False,
            )
        mime_type = str(document_file.content_type or "").strip().lower()
        if mime_type not in SUPPORTED_DOCUMENT_MIME_TYPES:
            return _error(
                400,
                code="REQUEST_MIME_TYPE_UNSUPPORTED",
                message="document_file must be application/pdf",
                retryable=False,
                details={"mime_type": mime_type or "unknown"},
            )
        request_id = str(parsed.get("request_id") or "").strip()
        if not request_id:
            stem = safe_token(Path(str(document_file.filename or "document")).stem, fallback="document")
            request_id = f"req_{stem}_{uuid.uuid4().hex}"
            parsed["request_id"] = request_id
        document_bytes = await document_file.read()
        if not document_bytes:
            return _error(
                400,
                code="REQUEST_EMPTY_INPUT",
                message="uploaded document_file is empty",
                retryable=False,
            )
        if len(document_bytes) > MAX_UPLOAD_BYTES:
            return _error(
                413,
                code="REQUEST_INPUT_TOO_LARGE",
                message=f"uploaded document_file exceeds {MAX_UPLOAD_BYTES} bytes",
                retryable=False,
            )
        try:
            profile = await anyio.to_thread.run_sync(
                lambda: profile_pdf(document_bytes, page_cap=settings.pdf.page_cap)
            )
        except PdfValidationError as exc:
            return _error(400, code=exc.code, message=str(exc), retryable=False)
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
        input_path = (upload_dir / f"input-{uuid.uuid4().hex[:8]}.pdf").resolve()
        input_path.write_bytes(document_bytes)

        parsed["document"] = {
            "local_path": str(input_path),
            "mime_type": mime_type,
            "filename": str(document_file.filename or ""),
            "size_bytes": int(len(document_bytes)),
            "page_count": int(profile.page_count),
        }
        status_code, body = await runtime.submit(parsed)
        if int(status_code) != 202:  # rejected or deduped -> this submission's file is unowned
            await anyio.to_thread.run_sync(lambda: input_path.unlink(missing_ok=True))
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

    @app.post("/v1/requests/{source_request_id}/rerender", response_model=RequestSubmitEnvelope)
    async def rerender_request(
        source_request_id: str,
        body: dict[str, Any] = Body(default_factory=dict),
    ) -> JSONResponse:
        status_code, payload = await runtime.submit_rerender(
            source_request_id=source_request_id, body=dict(body or {})
        )
        return JSONResponse(status_code=int(status_code), content=payload)

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
            icc_profile = original.info.get("icc_profile")
            if icc_profile:  # keep the colour profile through the re-encode (render re-embeds it)
                save_kwargs["icc_profile"] = icc_profile
            if image_format == "JPEG":
                if image.mode not in {"RGB", "L"}:
                    image = image.convert("RGB")
                save_kwargs["quality"] = 95
            image.save(out, format=image_format, **save_kwargs)
            return out.getvalue()
    except UnidentifiedImageError as exc:
        raise ValueError("uploaded image_file could not be decoded") from exc
    except Image.DecompressionBombError as exc:
        # Subclasses Exception directly (not OSError): without this arm a small file declaring
        # huge dimensions escapes as a 500 instead of the 400 invalid-input envelope.
        raise ValueError("uploaded image_file dimensions exceed the safety limit") from exc
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
