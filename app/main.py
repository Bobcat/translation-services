"""Composition root (HTTP entry point: ``app.main:app``).

Wires the service together: settings -> runtime -> the ``/v1/*`` route modules.

- ``app.core``        settings, request/response schemas, small helpers
- ``app.runtime``     job execution: FIFO queue, runner loop, lifecycle store
- ``app.tasks``       feature pipelines (one module per task; the readable flows)
- ``app.ocr``         OCR component (paddle backend, merge, segment, overlay)
- ``app.layout``      document-layout detector + the evidence read from it
- ``app.grouping``    VLM hint + aligner -> translation units
- ``app.translation`` language-pair routing, llm-pool translate, prompt library
- ``app.replacement`` erase + re-render
- ``app.pdf``         PDF intake, page raster, text-layer cells, assembly
- ``app.benchmark``   document-pair benchmark (measurement + scoring + store)
- ``app.regression``  the replay harnesses (image fixtures, pdf document fixtures)
- ``app.routes``      the HTTP surface, one module per concern, registered below
"""
from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.core.config import load_settings
from app.routes import benchmark as benchmark_routes
from app.routes import image_regression as image_regression_routes
from app.routes import pdf_regression as pdf_regression_routes
from app.routes import prompts as prompt_routes
from app.routes import requests as request_routes
from app.routes.common import error_response as _error
from app.runtime.service import RequestRuntime


def create_app(settings_path: str | Path | None = None) -> FastAPI:
    settings = load_settings(settings_path)
    runtime = RequestRuntime(settings=settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        await runtime.start()
        try:
            yield
        finally:
            await runtime.stop()

    app = FastAPI(title="Translation Services API", lifespan=lifespan)

    @app.exception_handler(RequestValidationError)
    async def _framework_validation_error(_request, exc: RequestValidationError) -> JSONResponse:
        # One error dialect for the whole API: without this, form/query validation (a missing
        # image_file, ``since_seq=abc``) speaks FastAPI's 422 ``{"detail": [...]}`` while every
        # service-side rejection speaks ``{code, message, retryable}`` — clients need two parsers.
        details = "; ".join(
            f"{'.'.join(str(loc) for loc in err.get('loc') or ())}: {err.get('msg')}"
            for err in exc.errors()
        )
        return _error(
            400,
            code="REQUEST_INVALID",
            message=details or "request validation failed",
            retryable=False,
        )

    request_routes.register(app, settings=settings, runtime=runtime)
    prompt_routes.register(app, settings=settings, runtime=runtime)
    image_regression_routes.register(app, settings=settings, runtime=runtime)
    pdf_regression_routes.register(app, settings=settings, runtime=runtime)
    benchmark_routes.register(app, settings=settings, runtime=runtime)
    return app


app = create_app()
