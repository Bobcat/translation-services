"""``/v1/pdf-regression/*`` — the document regression surface (design doc slice 2c).

The HTTP twin of ``scripts/pdf_regress.py``: capture a document fixture from a completed
``translate_pdf`` run, replay/diff it (optionally with benchmark-on-replay), accept a deliberate
change, and serve the reviewer artifacts. Replay/capture/accept are GPU-bound and run minutes,
not milliseconds — they execute in a worker thread and the caller (workbench proxy) uses a long
timeout, the same contract as ``/v1/benchmark/run``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import anyio

from fastapi import Body
from fastapi import FastAPI
from fastapi import Query
from fastapi.responses import FileResponse
from fastapi.responses import JSONResponse

from app.core.config import AppSettings
from app.core.util import safe_token
from app.regression.pdf import fixture as dfx
from app.regression.pdf.capture import CaptureError
from app.regression.pdf.capture import capture_document
from app.regression.pdf.capture import delete_variant
from app.regression.pdf.capture import testset_name_for
from app.regression.pdf.run import accept_document
from app.regression.pdf.run import run_document
from app.routes.common import error_response as _error
from app.runtime.service import RequestRuntime

# Document-level artifacts a reviewer may fetch; page artifacts are the fixed PNG trio.
_DOCUMENT_ARTIFACTS = {
    "source.pdf": "application/pdf",
    "accepted.pdf": "application/pdf",
    "actual.pdf": "application/pdf",
}
_PAGE_ARTIFACTS = {"snapshot.png", "actual.png", "snapshot_diff.png"}


def register(app: FastAPI, *, settings: AppSettings, runtime: RequestRuntime) -> None:
    root = dfx.PDF_REGRESSION_ROOT

    def _variant_path(name: str, lang: str, variant: str) -> Path | None:
        """Resolved ``<root>/<name>/<lang>/<vN>``, or None when the path escapes the root."""
        candidate = (root / name / lang / variant).resolve()
        try:
            candidate.relative_to(root.resolve())
        except ValueError:
            return None
        return candidate

    @app.get("/v1/pdf-regression/fixtures")
    async def pdf_regression_list() -> JSONResponse:
        documents = await anyio.to_thread.run_sync(dfx.list_documents)
        return JSONResponse(status_code=200, content={"documents": documents})

    @app.get("/v1/pdf-regression/status")
    async def pdf_regression_status(request_id: str = Query(...)) -> JSONResponse:
        """What capturing this completed run WOULD produce — the PDF twin of the image
        ``/regression/status``. Resolves the run's source PDF to a testset document by content
        hash (the fixture name) and reports the fixtures that already exist for it, so the capture
        panel can show a live badge and disable/enable the button before anything runs."""
        status_code, input_art = await runtime.artifact_path(request_id=request_id, artifact_name="input")
        if int(status_code) != 200:
            return JSONResponse(status_code=int(status_code), content=input_art)
        if str(input_art.get("mime_type")) != "application/pdf":
            return _error(409, code="REGRESSION_NOT_A_PDF_RUN", message="request is not a translate_pdf run", retryable=False)
        source_pdf = Path(str(input_art["path"]))

        def _status() -> dict[str, Any]:
            name = testset_name_for(source_pdf)
            langs: dict[str, list[str]] = {}
            if name:
                for doc in dfx.list_documents():
                    if doc["name"] == name:
                        langs.setdefault(doc["target_lang"], []).append(doc["variant"])
            return {"request_id": request_id, "name": name, "in_testset": name is not None, "langs": langs}

        return JSONResponse(status_code=200, content=await anyio.to_thread.run_sync(_status))

    @app.post("/v1/pdf-regression/capture")
    async def pdf_regression_capture(body: dict[str, Any] = Body(default_factory=dict)) -> JSONResponse:
        request_id = str(body.get("request_id") or "").strip()
        if not request_id:
            return _error(400, code="REGRESSION_BAD_REQUEST", message="request_id is required", retryable=False)
        status_code, input_art = await runtime.artifact_path(request_id=request_id, artifact_name="input")
        if int(status_code) != 200:
            return JSONResponse(status_code=int(status_code), content=input_art)
        if str(input_art.get("mime_type")) != "application/pdf":
            return _error(409, code="REGRESSION_NOT_A_PDF_RUN", message="request is not a translate_pdf run", retryable=False)
        source_pdf = Path(str(input_art["path"]))
        job_root = (runtime.work_root / safe_token(request_id)).resolve()
        name = str(body.get("name") or "").strip() or await anyio.to_thread.run_sync(
            lambda: testset_name_for(source_pdf)
        )
        if not name:
            return _error(
                400, code="REGRESSION_NAME_REQUIRED",
                message="source does not match a testset/pdf document; pass a name", retryable=False,
            )
        try:
            out = await anyio.to_thread.run_sync(
                lambda: capture_document(
                    settings,
                    job_root=job_root,
                    source_pdf=source_pdf,
                    name=name,
                    variant=str(body.get("variant") or "").strip() or None,
                    freeze_score=bool(body.get("freeze_score", True)),
                )
            )
        except CaptureError as exc:
            return _error(409, code="REGRESSION_CAPTURE_REFUSED", message=str(exc), retryable=False)
        return JSONResponse(status_code=200, content=out)

    @app.post("/v1/pdf-regression/run")
    async def pdf_regression_run(body: dict[str, Any] = Body(default_factory=dict)) -> JSONResponse:
        name = str(body.get("name") or "").strip()
        lang = str(body.get("lang") or "").strip()
        variant = str(body.get("variant") or "").strip()
        if not (name and lang and variant):
            return _error(400, code="REGRESSION_BAD_REQUEST", message="name, lang and variant are required", retryable=False)
        variant_path = _variant_path(name, lang, variant)
        if variant_path is None:
            return _error(400, code="REGRESSION_PATH_INVALID", message="invalid path", retryable=False)
        if not (variant_path / "document.json").exists():
            return _error(404, code="REGRESSION_FIXTURE_NOT_FOUND", message="document fixture not found", retryable=False)
        result = await anyio.to_thread.run_sync(
            lambda: run_document(settings, variant_path=variant_path, score=bool(body.get("score")))
        )
        return JSONResponse(status_code=200, content={"name": name, "lang": lang, "variant": variant, **result})

    @app.post("/v1/pdf-regression/accept")
    async def pdf_regression_accept(body: dict[str, Any] = Body(default_factory=dict)) -> JSONResponse:
        name = str(body.get("name") or "").strip()
        lang = str(body.get("lang") or "").strip()
        variant = str(body.get("variant") or "").strip()
        if not (name and lang and variant):
            return _error(400, code="REGRESSION_BAD_REQUEST", message="name, lang and variant are required", retryable=False)
        variant_path = _variant_path(name, lang, variant)
        if variant_path is None:
            return _error(400, code="REGRESSION_PATH_INVALID", message="invalid path", retryable=False)
        if not (variant_path / "document.json").exists():
            return _error(404, code="REGRESSION_FIXTURE_NOT_FOUND", message="document fixture not found", retryable=False)
        result = await anyio.to_thread.run_sync(
            lambda: accept_document(
                settings, variant_path=variant_path, freeze_score=bool(body.get("freeze_score", True))
            )
        )
        status_code = 200 if result.get("ok") else 409
        return JSONResponse(status_code=status_code, content={"name": name, "lang": lang, "variant": variant, **result})

    @app.get("/v1/pdf-regression/fixtures/{name}/{lang}/{variant}/artifact/{artifact}")
    async def pdf_regression_document_artifact(name: str, lang: str, variant: str, artifact: str):
        variant_path = _variant_path(name, lang, variant)
        if variant_path is None:
            return _error(400, code="REGRESSION_PATH_INVALID", message="invalid path", retryable=False)
        if artifact == "accepted_scores":
            scores = await anyio.to_thread.run_sync(lambda: dfx.load_accepted_scores(variant_path))
            if scores is None:
                return _error(404, code="REGRESSION_ARTIFACT_NOT_FOUND", message="no accepted scores frozen", retryable=False)
            return JSONResponse(status_code=200, content=scores)
        media_type = _DOCUMENT_ARTIFACTS.get(artifact)
        if media_type is None:
            return _error(404, code="REGRESSION_ARTIFACT_UNKNOWN", message="unknown artifact", retryable=False)
        path = variant_path / artifact
        if not path.exists():
            return _error(404, code="REGRESSION_ARTIFACT_NOT_FOUND", message="artifact not found", retryable=False)
        return FileResponse(path=str(path), media_type=media_type, filename=path.name)

    @app.get("/v1/pdf-regression/fixtures/{name}/{lang}/{variant}/pages/{page}/{artifact}")
    async def pdf_regression_page_artifact(name: str, lang: str, variant: str, page: int, artifact: str):
        variant_path = _variant_path(name, lang, variant)
        if variant_path is None:
            return _error(400, code="REGRESSION_PATH_INVALID", message="invalid path", retryable=False)
        if artifact not in _PAGE_ARTIFACTS:
            return _error(404, code="REGRESSION_ARTIFACT_UNKNOWN", message="unknown artifact", retryable=False)
        path = dfx.page_dir(variant_path, max(1, int(page))) / artifact
        if not path.exists():
            return _error(404, code="REGRESSION_ARTIFACT_NOT_FOUND", message="artifact not found", retryable=False)
        return FileResponse(path=str(path), media_type="image/png", filename=path.name)

    @app.delete("/v1/pdf-regression/fixtures/{name}/{lang}/{variant}")
    async def pdf_regression_delete(name: str, lang: str, variant: str) -> JSONResponse:
        variant_path = _variant_path(name, lang, variant)
        if variant_path is None:
            return _error(400, code="REGRESSION_PATH_INVALID", message="invalid path", retryable=False)
        ok = await anyio.to_thread.run_sync(lambda: delete_variant(variant_path))
        return JSONResponse(status_code=200 if ok else 404, content={"deleted": ok})
