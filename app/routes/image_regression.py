"""``/v1/regression/*`` — the image regression surface: testset admin, fixture capture,
replay and re-baseline. The document (PDF) twin lives in ``app.routes.pdf_regression``."""
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
from app.regression.image import capture as regression_capture
from app.regression.image import run as regression_run
from app.routes.common import error_response as _error
from app.runtime.service import RequestRuntime


def register(app: FastAPI, *, settings: AppSettings, runtime: RequestRuntime) -> None:
    @app.get("/v1/regression/status")
    async def regression_status(name: str = Query(...)) -> JSONResponse:
        return JSONResponse(status_code=200, content=regression_capture.status(name))

    @app.post("/v1/regression/testset")
    async def regression_add_testset(body: dict[str, Any] = Body(default_factory=dict)) -> JSONResponse:
        request_id = str(body.get("request_id") or "").strip()
        name = str(body.get("name") or "").strip()
        # Optional destination subdir (the workbench picker); '' = flat testset root.
        subdir = str(body.get("subdir") or "").strip().strip("/")
        if not request_id or not name:
            return _error(400, code="REGRESSION_BAD_REQUEST", message="request_id and name are required", retryable=False)
        # ``name``/``subdir`` land in a filesystem path — same resolve/relative_to guard as every
        # sibling regression endpoint, so "../" or an absolute name cannot write outside the testset.
        rel = f"{subdir}/{name}" if subdir else name
        testset_root = regression_capture.TESTSET_ROOT.resolve()
        dest = (regression_capture.TESTSET_ROOT / f"{rel}.x").resolve()
        try:
            dest.relative_to(testset_root)
        except ValueError:
            return _error(400, code="REGRESSION_BAD_NAME", message="name must stay inside the testset", retryable=False)
        # Unique-stem invariant: a stem may live at exactly one place in the testset tree, else the
        # fixture mirror is ambiguous. The workbench also disables Add when in_testset, this is the guard.
        existing = regression_capture.testset_image(name)
        if existing is not None:
            return _error(409, code="REGRESSION_DUPLICATE_STEM",
                          message=f"stem {name!r} is already in the testset at {existing}", retryable=False)
        status_code, lifecycle = await runtime.get_request(request_id)
        if int(status_code) != 200:
            return JSONResponse(status_code=int(status_code), content=lifecycle)
        response = dict(lifecycle.get("response") or {})
        input_path = Path(str((dict((response.get("artifacts") or {}).get("input") or {})).get("path") or ""))
        if not input_path.exists():
            return _error(404, code="REGRESSION_INPUT_MISSING", message="input artifact not available", retryable=False)
        dest = regression_capture.TESTSET_ROOT / f"{rel}{input_path.suffix or '.png'}"

        def _copy_input() -> None:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(input_path.read_bytes())

        await anyio.to_thread.run_sync(_copy_input)
        return JSONResponse(status_code=200, content={"path": str(dest), **regression_capture.status(name)})

    @app.post("/v1/regression/fixtures")
    async def regression_add_fixture(body: dict[str, Any] = Body(default_factory=dict)) -> JSONResponse:
        request_id = str(body.get("request_id") or "").strip()
        name = str(body.get("name") or "").strip()
        variant = str(body.get("variant") or "").strip() or None
        allow_duplicate = bool(body.get("allow_duplicate"))
        if not request_id or not name:
            return _error(400, code="REGRESSION_BAD_REQUEST", message="request_id and name are required", retryable=False)
        status_code, lifecycle = await runtime.get_request(request_id)
        if int(status_code) != 200:
            return JSONResponse(status_code=int(status_code), content=lifecycle)
        if (lifecycle.get("state") or "") != "completed":
            return _error(409, code="REGRESSION_NOT_COMPLETED", message=f"request state is {lifecycle.get('state')}", retryable=False)
        status_code, rendered_art = await runtime.artifact_path(request_id=request_id, artifact_name="rendered")
        if int(status_code) != 200:
            return JSONResponse(status_code=int(status_code), content=rendered_art)
        status_code, input_art = await runtime.artifact_path(request_id=request_id, artifact_name="input")
        if int(status_code) != 200:
            return JSONResponse(status_code=int(status_code), content=input_art)
        rendered = Path(str(rendered_art["path"])).read_bytes()
        source_path = Path(str(input_art["path"]))
        response = dict(lifecycle.get("response") or {})
        # A re-translate response carries only the new translations + render, not the OCR cells /
        # grouping hint / model the replay needs. Walk up source_request_id to the run that did the
        # grouping and graft those frozen inputs on, so the fixture stays replayable (grouping ->
        # align -> render) with the re-translate's own translations, target language and flags.
        source_request_id = str((response.get("metadata") or {}).get("source_request_id") or "").strip()
        if source_request_id and not (response.get("ocr") or {}).get("cells"):
            grouping_response = None
            seen: set[str] = set()
            cursor_id = source_request_id
            while cursor_id and cursor_id not in seen:
                seen.add(cursor_id)
                sc, src_lifecycle = await runtime.get_request(cursor_id)
                if int(sc) != 200:
                    break
                cursor = dict(src_lifecycle.get("response") or {})
                if (cursor.get("ocr") or {}).get("cells"):
                    grouping_response = cursor
                    break
                cursor_id = str((cursor.get("metadata") or {}).get("source_request_id") or "").strip()
            if grouping_response is None:
                return _error(
                    409, code="REGRESSION_GROUPING_SOURCE_MISSING",
                    message="re-translate source run with the OCR grouping is unavailable; capture from the original run",
                    retryable=False,
                )
            response = regression_capture.graft_grouping_inputs(response, grouping_response)
        try:
            out = await anyio.to_thread.run_sync(
                lambda: regression_capture.capture(
                    settings.ocr, response=response, rendered_png=rendered,
                    source_bytes=source_path.read_bytes(), source_suffix=source_path.suffix or ".png",
                    name=name, variant=variant, allow_duplicate=allow_duplicate,
                )
            )
        except ValueError as exc:  # the source stem is not unique in the testset (mirror is ambiguous)
            return _error(409, code="REGRESSION_AMBIGUOUS_STEM", message=str(exc), retryable=False)
        return JSONResponse(status_code=200, content={**out, **regression_capture.status(name)})

    @app.get("/v1/regression/fixtures")
    async def regression_list() -> JSONResponse:
        return JSONResponse(status_code=200, content={"images": regression_capture.list_fixtures()})

    @app.get("/v1/regression/subdirs")
    async def regression_subdirs() -> JSONResponse:
        return JSONResponse(status_code=200, content={"subdirs": regression_capture.list_subdirs()})

    @app.get("/v1/regression/fixtures/{name}/{lang}/{variant}/{artifact}")
    async def regression_variant_artifact(name: str, lang: str, variant: str, artifact: str):
        root = regression_capture.REGRESSION_ROOT.resolve()
        variant_path = (regression_capture.resolve_fixture_root(name) / lang / variant).resolve()
        try:
            variant_path.relative_to(root)
        except ValueError:
            return _error(400, code="REGRESSION_PATH_INVALID", message="invalid path", retryable=False)
        if artifact in {"snapshot.png", "actual.png", "snapshot_diff.png"}:
            path, media_type = variant_path / artifact, "image/png"
        elif artifact == "source":  # the fixture's own captured image (variable extension)
            sources = sorted(variant_path.glob("source.*"))
            path, media_type = (sources[0] if sources else None), None
        else:
            return _error(404, code="REGRESSION_ARTIFACT_UNKNOWN", message="unknown artifact", retryable=False)
        if path is None or not path.exists():
            return _error(404, code="REGRESSION_ARTIFACT_NOT_FOUND", message="artifact not found", retryable=False)
        return FileResponse(path=str(path), media_type=media_type, filename=path.name)

    @app.post("/v1/regression/run")
    async def regression_run_variant(body: dict[str, Any] = Body(default_factory=dict)) -> JSONResponse:
        name = str(body.get("name") or "").strip()
        lang = str(body.get("lang") or "").strip()
        variant = str(body.get("variant") or "").strip()
        if not (name and lang and variant):
            return _error(400, code="REGRESSION_BAD_REQUEST", message="name, lang and variant are required", retryable=False)
        root = regression_capture.REGRESSION_ROOT.resolve()
        variant_path = (regression_capture.resolve_fixture_root(name) / lang / variant).resolve()
        try:
            variant_path.relative_to(root)
        except ValueError:
            return _error(400, code="REGRESSION_PATH_INVALID", message="invalid path", retryable=False)
        if not (variant_path / "fixture.json").exists():
            return _error(404, code="REGRESSION_FIXTURE_NOT_FOUND", message="fixture not found", retryable=False)
        result = await anyio.to_thread.run_sync(
            lambda: regression_run.run_variant(settings.ocr, variant_path=variant_path)
        )
        return JSONResponse(status_code=200, content={"name": name, "lang": lang, "variant": variant, **result})

    @app.post("/v1/regression/resnapshot")
    async def regression_resnapshot(body: dict[str, Any] = Body(default_factory=dict)) -> JSONResponse:
        name = str(body.get("name") or "").strip()
        lang = str(body.get("lang") or "").strip()
        variant = str(body.get("variant") or "").strip()
        if not (name and lang and variant):
            return _error(400, code="REGRESSION_BAD_REQUEST", message="name, lang and variant are required", retryable=False)
        root = regression_capture.REGRESSION_ROOT.resolve()
        variant_path = (regression_capture.resolve_fixture_root(name) / lang / variant).resolve()
        try:
            variant_path.relative_to(root)
        except ValueError:
            return _error(400, code="REGRESSION_PATH_INVALID", message="invalid path", retryable=False)
        if not (variant_path / "fixture.json").exists():
            return _error(404, code="REGRESSION_FIXTURE_NOT_FOUND", message="fixture not found", retryable=False)
        result = await anyio.to_thread.run_sync(
            lambda: regression_run.resnapshot(settings.ocr, variant_path=variant_path)
        )
        return JSONResponse(status_code=200, content={"name": name, "lang": lang, "variant": variant, **result})

    @app.delete("/v1/regression/fixtures/{name}")
    async def regression_delete_name(name: str) -> JSONResponse:
        ok = regression_capture.delete_path(name)
        return JSONResponse(status_code=200 if ok else 404, content={"deleted": ok})

    @app.delete("/v1/regression/fixtures/{name}/{lang}")
    async def regression_delete_lang(name: str, lang: str) -> JSONResponse:
        ok = regression_capture.delete_path(name, lang)
        return JSONResponse(status_code=200 if ok else 404, content={"deleted": ok})

    @app.delete("/v1/regression/fixtures/{name}/{lang}/{variant}")
    async def regression_delete_variant(name: str, lang: str, variant: str) -> JSONResponse:
        ok = regression_capture.delete_path(name, lang, variant)
        return JSONResponse(status_code=200 if ok else 404, content={"deleted": ok})
