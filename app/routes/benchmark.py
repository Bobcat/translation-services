"""``/v1/benchmark/*`` — the pdf document-pair benchmark surface (design doc slice 2c).

Moved verbatim from ``app.main`` (the approved seam once it crossed 800 lines); behaviour
unchanged. The measurement shares the process's cached layout/OCR engines with the pipeline
(predict locks serialize access), so a benchmark run competes for the same GPU but needs no
extra VRAM.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import uuid

import anyio

from fastapi import FastAPI
from fastapi import File
from fastapi import Form
from fastapi import Query
from fastapi import UploadFile
from fastapi.responses import JSONResponse
from fastapi.responses import Response

from app.benchmark.measurement import measure_pair
from app.benchmark.overlay import overlay_png
from app.benchmark.scoring import score_measurement
from app.benchmark.store import default_data_root
from app.benchmark.store import find_run
from app.benchmark.store import runs_index
from app.benchmark.store import save_run
from app.core.config import AppSettings
from app.pdf.document import PdfValidationError
from app.pdf.document import profile_pdf
from app.routes.common import MAX_UPLOAD_BYTES
from app.routes.common import error_response as _error
from app.routes.common import repo_root_dir
from app.runtime.service import RequestRuntime


def register(app: FastAPI, *, settings: AppSettings, runtime: RequestRuntime) -> None:
    benchmark_root = default_data_root()
    testset_pdf_root = (repo_root_dir() / "testset" / "pdf").resolve()

    @app.get("/v1/benchmark/results")
    async def benchmark_results() -> JSONResponse:
        runs = await anyio.to_thread.run_sync(lambda: runs_index(benchmark_root))
        return JSONResponse(status_code=200, content={"runs": runs})

    @app.get("/v1/benchmark/testset")
    async def benchmark_testset() -> JSONResponse:
        names = sorted(p.name for p in testset_pdf_root.glob("*.pdf")) if testset_pdf_root.is_dir() else []
        return JSONResponse(status_code=200, content={"documents": names})

    @app.post("/v1/benchmark/run")
    async def benchmark_run(
        request_json: str = Form(...),
        source_file: UploadFile | None = File(default=None),
        translated_file: UploadFile | None = File(default=None),
    ) -> JSONResponse:
        """One multipart surface, two source modes: ``{"request_id": …}`` scores a
        completed translate_pdf run (input vs rendered artifact); otherwise a
        ``translated_file`` upload is scored against either ``{"testset_doc": …}``
        or an uploaded ``source_file``. ``system`` labels the column; ``doc_id``
        defaults to the source filename stem so testset re-runs land on one row."""
        try:
            body = json.loads(request_json)
            assert isinstance(body, dict)
        except Exception:
            return _error(400, code="REQUEST_JSON_INVALID", message="request_json must be a JSON object", retryable=False)

        temp_files: list[Path] = []
        try:
            request_id = str(body.get("request_id") or "").strip()
            if request_id:
                status_code, input_art = await runtime.artifact_path(request_id=request_id, artifact_name="input")
                if int(status_code) != 200:
                    return JSONResponse(status_code=int(status_code), content=input_art)
                status_code, rendered_art = await runtime.artifact_path(request_id=request_id, artifact_name="rendered")
                if int(status_code) != 200:
                    return JSONResponse(status_code=int(status_code), content=rendered_art)
                if str(rendered_art.get("mime_type")) != "application/pdf":
                    return _error(409, code="BENCHMARK_NOT_A_PDF_RUN", message="request has no PDF rendered artifact", retryable=False)
                source_path = Path(str(input_art["path"]))
                translated_path = Path(str(rendered_art["path"]))
                system = str(body.get("system") or "ours").strip() or "ours"
            else:
                if translated_file is None:
                    return _error(400, code="REQUEST_INPUT_REQUIRED", message="translated_file (or request_id) is required", retryable=False)
                system = str(body.get("system") or "").strip()
                if not system:
                    return _error(400, code="BENCHMARK_SYSTEM_REQUIRED", message="system label is required for an import", retryable=False)
                testset_doc = str(body.get("testset_doc") or "").strip()
                if testset_doc:
                    source_path = (testset_pdf_root / testset_doc).resolve()
                    try:
                        source_path.relative_to(testset_pdf_root)
                    except ValueError:
                        return _error(400, code="BENCHMARK_BAD_TESTSET_DOC", message="invalid testset document name", retryable=False)
                    if not source_path.exists():
                        return _error(404, code="BENCHMARK_TESTSET_DOC_NOT_FOUND", message="testset document not found", retryable=False)
                try:
                    if not testset_doc:
                        if source_file is None:
                            return _error(400, code="REQUEST_INPUT_REQUIRED", message="testset_doc or source_file is required", retryable=False)
                        source_path = await _spool_upload(source_file, temp_files)
                    translated_path = await _spool_upload(translated_file, temp_files)
                except ValueError as exc:
                    return _error(413, code="REQUEST_INPUT_TOO_LARGE", message=str(exc), retryable=False)

            doc_id = str(body.get("doc_id") or "").strip() or source_path.stem
            for path, side in ((source_path, "source"), (translated_path, "translated")):
                try:
                    await anyio.to_thread.run_sync(lambda p=path: profile_pdf(p, page_cap=settings.pdf.page_cap))
                except PdfValidationError as exc:
                    return _error(400, code=exc.code, message=f"{side}: {exc}", retryable=False)

            def _measure_and_store():
                measurement = measure_pair(settings=settings, source_pdf=source_path, translated_pdf=translated_path)
                scores = score_measurement(measurement)
                run = save_run(
                    data_root=benchmark_root, doc_id=doc_id, system=system,
                    source_pdf=source_path, translated_pdf=translated_path,
                    measurement=measurement, scores=scores,
                )
                return run, scores

            run, scores = await anyio.to_thread.run_sync(_measure_and_store)
            return JSONResponse(status_code=200, content={
                "doc_id": run.doc_id, "system": run.system, "run_id": run.run_id,
                "scoring_version": scores.get("scoring_version"),
                "axes": scores.get("axes"), "indicators": scores.get("indicators"),
                "flags": scores.get("flags"),
            })
        finally:
            for path in temp_files:
                await anyio.to_thread.run_sync(lambda p=path: p.unlink(missing_ok=True))

    async def _spool_upload(upload: UploadFile, temp_files: list[Path]) -> Path:
        data = await upload.read()
        if len(data) > MAX_UPLOAD_BYTES:
            raise ValueError(f"uploaded file exceeds {MAX_UPLOAD_BYTES} bytes")
        target = (runtime.work_root / "_uploads" / f"benchmark-{uuid.uuid4().hex}.pdf").resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        temp_files.append(target)
        return target

    @app.get("/v1/benchmark/runs/{doc_id}/{system}")
    async def benchmark_run_detail(doc_id: str, system: str, run_id: str = Query(default="")) -> JSONResponse:
        run = await anyio.to_thread.run_sync(
            lambda: find_run(benchmark_root, doc_id=doc_id, system=system, run_id=run_id.strip() or None)
        )
        if run is None:
            return _error(404, code="BENCHMARK_RUN_NOT_FOUND", message="no stored run for this document/system", retryable=False)
        scores = await anyio.to_thread.run_sync(run.load_scores)
        return JSONResponse(status_code=200, content={
            "doc_id": run.doc_id, "system": run.system, "run_id": run.run_id, "scores": scores,
        })

    @app.get("/v1/benchmark/runs/{doc_id}/{system}/{run_id}/overlay/{side}/{page}")
    async def benchmark_overlay(doc_id: str, system: str, run_id: str, side: str, page: int):
        run = await anyio.to_thread.run_sync(
            lambda: find_run(benchmark_root, doc_id=doc_id, system=system, run_id=run_id)
        )
        if run is None:
            return _error(404, code="BENCHMARK_RUN_NOT_FOUND", message="no stored run for this document/system", retryable=False)
        try:
            payload = await anyio.to_thread.run_sync(
                lambda: overlay_png(run, side=side, page_index=max(0, int(page) - 1))
            )
        except ValueError as exc:
            return _error(400, code="BENCHMARK_OVERLAY_INVALID", message=str(exc), retryable=False)
        return Response(content=payload, media_type="image/png")
