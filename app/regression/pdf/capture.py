"""Build a document fixture from a completed ``translate_pdf`` run's persisted artifacts.

Freezes exactly what ran — no VLM / translator / OCR re-run. Everything comes from the job dir
(``pages/page-NNN/{grouping,translation,request,llm_calls}.json`` + ``rendered.png`` +
``rendered.pdf``) plus the uploaded source PDF. Two derivations are not persisted by the run and
are re-derived here, then VERIFIED before anything is written:

- ``ignored_cell_ids`` per page comes from a capture-time align replay (pure code on frozen
  inputs — deterministic, so it equals what the run produced);
- the whole per-page deterministic chain is replayed and diffed against the run's own output
  (align structure exactly, render via re-OCR). A page whose replay cannot reproduce the live run
  would freeze a born-failing fixture — the capture refuses instead, listing the diffs.

The run must have per-page ``request.json`` (the resolved request flags, written by the pipeline
since this harness landed). Runs that predate it cannot be captured faithfully: re-run the
document.
"""
from __future__ import annotations

import json
import shutil
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from app.core.config import AppSettings
from app.pdf.document import profile_pdf
from app.pdf.raster import PageRasterizer
from app.pdf.textlayer import PageTextExtractor
from app.regression.pages import fixture as fx
from app.regression.pages.fixture import _next_variant
from app.regression.pages.fixture import build_fixture
from app.regression.pages.compare import diff_units
from app.regression.pages.compare import reocr_mismatches
from app.regression.pdf import checks
from app.regression.pdf import fixture as dfx
from app.regression.pages.replay import replay_fixture
from app.regression.pages.snapshot import reocr_rows


class CaptureError(ValueError):
    """A run that cannot be frozen faithfully; the message says why."""


TESTSET_PDF_ROOT = Path("testset/pdf")


def _iter_source_pdfs(testset_root: Path) -> "Iterator[Path]":
    """Every source PDF under ``testset_root`` at any depth, pruning ``_``-prefixed dirs
    (``_regression`` holds fixtures, not sources). Mirrors the image harness's source walk."""
    if not testset_root.is_dir():
        return
    stack = [testset_root]
    while stack:
        for child in stack.pop().iterdir():
            if child.is_dir():
                if not child.name.startswith("_"):
                    stack.append(child)
            elif child.suffix.lower() == ".pdf":
                yield child


def find_testset_pdfs(name: str, *, testset_root: Path = TESTSET_PDF_ROOT) -> list[Path]:
    """All source PDFs whose stem is ``name``. More than one violates the unique-stem invariant."""
    return sorted(p for p in _iter_source_pdfs(testset_root) if p.stem == name)


def testset_pdf(name: str, *, testset_root: Path = TESTSET_PDF_ROOT) -> Path | None:
    """The source PDF for ``name`` anywhere in the testset tree, or ``None``. Stems are unique
    across the tree, so the first match is it."""
    matches = find_testset_pdfs(name, testset_root=testset_root)
    return matches[0] if matches else None


def source_reldir(name: str, *, testset_root: Path = TESTSET_PDF_ROOT) -> str:
    """The dir of ``name``'s source PDF relative to the testset root (``''`` = flat root), so a
    fixture can mirror it under ``_regression``. Raises ``ValueError`` when the stem is not unique;
    returns ``''`` when the source is not in the testset yet."""
    matches = find_testset_pdfs(name, testset_root=testset_root)
    if len(matches) > 1:
        joined = ", ".join(str(p.relative_to(testset_root)) for p in matches)
        raise ValueError(f"stem {name!r} is not unique in the testset ({joined})")
    if not matches:
        return ""
    rel = str(matches[0].parent.relative_to(testset_root))
    return "" if rel == "." else rel


def list_subdirs(*, testset_root: Path = TESTSET_PDF_ROOT) -> list[str]:
    """Existing non-underscore subdirectories under the testset root (relative paths, any depth),
    for the Add-to-testset destination picker. Empty dirs are included so a fresh PDF can be filed."""
    out: list[str] = []
    if not testset_root.is_dir():
        return out
    stack = [testset_root]
    while stack:
        for child in stack.pop().iterdir():
            if child.is_dir() and not child.name.startswith("_"):
                out.append(str(child.relative_to(testset_root)))
                stack.append(child)
    return sorted(out)


def testset_name_for(source_pdf: Path, *, testset_root: Path = TESTSET_PDF_ROOT) -> str | None:
    """The testset stem whose bytes match the uploaded source — the natural fixture name.
    Shared by the CLI and the capture endpoint so both default identically."""
    digest = fx.sha256(source_pdf.read_bytes())
    for candidate in _iter_source_pdfs(testset_root):
        if fx.sha256(candidate.read_bytes()) == digest:
            return candidate.stem
    return None


@dataclass(frozen=True)
class _PageCapture:
    page: int
    fixture: fx.Fixture
    snapshot: fx.Snapshot
    snapshot_png: bytes
    llm_calls: list[dict[str, Any]]
    timings: dict[str, float]


def capture_document(
    settings: AppSettings,
    *,
    job_root: Path,
    source_pdf: Path,
    name: str,
    variant: str | None = None,
    root: Path = dfx.PDF_REGRESSION_ROOT,
    freeze_score: bool = True,
) -> dict[str, Any]:
    document = _load_json(job_root / "document.json", "document.json")
    rendered_pdf = job_root / "rendered.pdf"
    if not rendered_pdf.exists():
        raise CaptureError(f"{rendered_pdf} is missing (incomplete run)")
    source_bytes = source_pdf.read_bytes()
    dpi = int(document.get("analysis_dpi") or 0)
    run_pages = list(document.get("pages") or [])
    if not dpi or not run_pages:
        raise CaptureError("document.json carries no analysis_dpi/pages (incomplete run)")

    census = [
        {
            "page": int(entry.get("page") or 0),
            **{key: entry.get(key) for key in dfx.CENSUS_PROFILE_KEYS},
            "cell_source": str(entry.get("cell_source") or "ocr"),
        }
        for entry in run_pages
    ]

    # Drift guard: the frozen source must still profile exactly as it did at run time — if the
    # census code or the file changed since, the fixture would freeze a lie.
    profile = profile_pdf(source_pdf, page_cap=len(census))
    census_diffs = checks.census_diffs(census, [page.to_dict() for page in profile.pages])
    if census_diffs:
        raise CaptureError(
            "source no longer profiles as the run's census: " + "; ".join(census_diffs)
        )

    pages: list[_PageCapture] = []
    failures: list[str] = []
    target_lang = ""
    with (
        PageRasterizer(source_pdf, dpi=dpi) as rasterizer,
        PageTextExtractor(source_pdf, dpi=dpi) as extractor,
        TemporaryDirectory(prefix="pdf-capture-") as tmp,
    ):
        for entry in census:
            page_no = int(entry["page"])
            page_dir = job_root / "pages" / f"page-{page_no:03d}"
            capture = _capture_page(
                settings,
                page_dir=page_dir,
                page_no=page_no,
                page_index=page_no - 1,
                cell_source=str(entry["cell_source"]),
                rasterizer=rasterizer,
                extractor=extractor,
                tmp=Path(tmp),
            )
            if isinstance(capture, str):
                failures.append(capture)
                continue
            if target_lang and capture.fixture.target_lang != target_lang:
                failures.append(
                    f"page {page_no}: target_lang {capture.fixture.target_lang!r} != {target_lang!r}"
                )
            target_lang = target_lang or capture.fixture.target_lang
            pages.append(capture)
    if failures:
        raise CaptureError("capture verification failed:\n  " + "\n  ".join(failures))

    accepted_bytes = rendered_pdf.read_bytes()
    assembled_diffs = checks.assembled_pdf_diffs(accepted_bytes, census)
    if assembled_diffs:
        raise CaptureError("run's assembled pdf is inconsistent: " + "; ".join(assembled_diffs))

    lang = target_lang or "unknown"
    # Nest the fixture to mirror the source PDF's subdir in the testset (docpack/07_… ->
    # _regression/docpack/07_…), exactly like the image harness. Falls back to the flat root when
    # the source is not in the testset (a name was typed for a PDF that was never added).
    try:
        reldir = source_reldir(name)
    except ValueError:
        reldir = ""
    name_dir = (root / reldir / name) if reldir else (root / name)
    lang_dir = name_dir / lang
    resolved_variant = variant or _next_variant(lang_dir)
    variant_path = lang_dir / resolved_variant

    document_fixture = dfx.DocumentFixture(
        source_sha256=fx.sha256(source_bytes),
        analysis_dpi=dpi,
        target_lang=lang,
        census=census,
    )
    dfx.save_document(variant_path, document_fixture)
    dfx.source_pdf_path(variant_path).write_bytes(source_bytes)
    dfx.accepted_pdf_path(variant_path).write_bytes(accepted_bytes)
    for capture in pages:
        page_path = dfx.page_dir(variant_path, capture.page)
        fx.save(page_path, capture.fixture, capture.snapshot)
        (page_path / "snapshot.png").write_bytes(capture.snapshot_png)
        # Forensics-only sidecar, same rationale as the image capture: the exact prompts of the
        # run that produced this fixture, nowhere else once the work dir is TTL-swept.
        (page_path / "llm_calls.json").write_text(
            json.dumps(capture.llm_calls, ensure_ascii=False, indent=1), encoding="utf-8"
        )

    out: dict[str, Any] = {
        "path": str(variant_path),
        "name": name,
        "target_lang": lang,
        "variant": resolved_variant,
        "pages": len(pages),
        "units": sum(
            len(c.fixture.hint_translations) + len(c.fixture.leftover_translations) for c in pages
        ),
        "timings": {
            key: round(sum(c.timings.get(key, 0.0) for c in pages), 1)
            for key in ("group_ms", "render_ms")
        },
    }
    if freeze_score:
        out["accepted_scores"] = freeze_accepted_score(
            settings, variant_path=variant_path, target_lang=lang
        )
    return out


def freeze_accepted_score(
    settings: AppSettings, *, variant_path: Path, target_lang: str
) -> dict[str, Any]:
    """Benchmark ``accepted.pdf`` against ``source.pdf`` and freeze the result as the accepted
    score — the baseline every later replay's score is diffed against. GPU-bound (layout + OCR
    over both renders); imports lazily so replay-only paths never load the measurement stack."""
    from app.benchmark.measurement import measure_pair
    from app.benchmark.scoring import score_measurement
    from app.regression.pages.snapshot import ocr_language_for_target

    measurement = measure_pair(
        settings=settings,
        source_pdf=dfx.source_pdf_path(variant_path),
        translated_pdf=dfx.accepted_pdf_path(variant_path),
        ocr_language=ocr_language_for_target(target_lang),
    )
    scores = score_measurement(measurement)
    dfx.accepted_measurement_path(variant_path).write_text(
        json.dumps(measurement, ensure_ascii=False), encoding="utf-8"
    )
    dfx.save_accepted_scores(variant_path, scores)
    return scores


def _capture_page(
    settings: AppSettings,
    *,
    page_dir: Path,
    page_no: int,
    page_index: int,
    cell_source: str,
    rasterizer: PageRasterizer,
    extractor: PageTextExtractor,
    tmp: Path,
) -> _PageCapture | str:
    """One page: freeze from artifacts, derive ignored cells, verify the replay. Returns the
    capture, or a failure string (the caller aggregates so ALL failing pages get reported)."""
    try:
        grouping = _load_json(page_dir / "grouping.json", f"page {page_no}: grouping.json")
        translation = _load_json(page_dir / "translation.json", f"page {page_no}: translation.json")
        request_meta = _load_json(page_dir / "request.json", f"page {page_no}: request.json")
    except CaptureError as exc:
        if "request.json" in str(exc):
            return (
                f"page {page_no}: request.json is missing — the run predates per-page request "
                "persistence; re-run the document and capture from the fresh run"
            )
        return str(exc)
    llm_calls = _load_optional_json(page_dir / "llm_calls.json") or []
    rendered_png = page_dir / "rendered.png"
    input_png = page_dir / "input.png"
    if not rendered_png.exists() or not input_png.exists():
        return f"page {page_no}: rendered.png/input.png missing (incomplete run)"

    raster_bytes = rasterizer.render_png(page_index)
    if fx.sha256(raster_bytes) != fx.sha256(input_png.read_bytes()):
        return (
            f"page {page_no}: rasterizing source.pdf at the run dpi no longer reproduces the "
            "run's input.png (raster engine drift since the run?)"
        )
    if cell_source == "pdf_text_layer":
        extracted = extractor.cells_for_page(page_index)
        cell_diffs = checks.extraction_diffs(list(grouping.get("cells") or []), extracted.cells)
        if cell_diffs:
            return f"page {page_no}: " + "; ".join(cell_diffs)

    translation_units = _merge_translations(grouping, translation, page_no)
    if isinstance(translation_units, str):
        return translation_units
    response_like = {
        "ocr": {"cells": grouping.get("cells") or [], "translation_units": translation_units},
        "metadata": {
            **request_meta,
            "grouping_model": _grouping_model(llm_calls, request_meta),
            "layout_regions": grouping.get("layout_regions") or [],
        },
        "llm_calls": llm_calls,
    }
    fixture = build_fixture(response_like, source_bytes=raster_bytes)

    raster_path = tmp / f"page-{page_no:03d}.png"
    raster_path.write_bytes(raster_bytes)
    actual_units, actual_ignored, replay_png, timings = replay_fixture(raster_path, fixture)
    snapshot = fx.Snapshot(
        expected_units=[fx.expected_unit_of(unit) for unit in translation_units],
        ignored_cells=actual_ignored,
        reocr=reocr_rows(settings.ocr, rendered_png.read_bytes(), fixture.target_lang),
    )
    diffs = diff_units(snapshot.expected_units, actual_units, snapshot.ignored_cells, actual_ignored)
    replay_reocr_diffs, _boxes = reocr_mismatches(
        snapshot.reocr, reocr_rows(settings.ocr, replay_png, fixture.target_lang)
    )
    diffs += replay_reocr_diffs
    if diffs:
        return f"page {page_no}: replay does not reproduce the run: " + "; ".join(diffs)
    return _PageCapture(
        page=page_no,
        fixture=fixture,
        snapshot=snapshot,
        snapshot_png=rendered_png.read_bytes(),
        llm_calls=llm_calls,
        timings=timings,
    )


def _merge_translations(
    grouping: dict[str, Any], translation: list[dict[str, Any]], page_no: int
) -> list[dict[str, Any]] | str:
    """The run's post-preserve-filter unit list with its translations merged back on — the same
    shape ``response.ocr.translation_units`` has on the image flow. ``translation.json`` holds one
    entry per unit that went to translation (including skipped ones), so its id set IS the
    post-filter set; grouping.json holds the full pre-filter align output in order."""
    by_id: dict[int, dict[str, Any]] = {}
    for entry in translation:
        by_id[int(entry.get("unit_id") or 0)] = entry
    units: list[dict[str, Any]] = []
    for unit in grouping.get("units") or []:
        entry = by_id.pop(int(unit.get("id") or 0), None)
        if entry is None:
            continue  # filtered out before translation (preserve_heuristic_text)
        unit = dict(unit)
        unit["translated_text"] = str(entry.get("translated_text") or "")
        unit["field_translations"] = entry.get("field_translations")
        units.append(unit)
    if by_id:
        return (
            f"page {page_no}: translation.json has unit ids missing from grouping.json: "
            f"{sorted(by_id)} (corrupt run artifacts)"
        )
    return units


def _grouping_model(llm_calls: list[dict[str, Any]], request_meta: dict[str, Any]) -> str:
    """The RESOLVED grouping model: what was actually sent to the pool (the request may have left
    it empty for the configured default). A page without a grouping call has no hint either — the
    request value (possibly empty) is then the only, and sufficient, record."""
    for call in llm_calls:
        if "grouping" in str(call.get("role") or "").lower():
            model = str((call.get("payload") or {}).get("model") or "")
            if model:
                return model
    return str(request_meta.get("grouping_model") or "")


def _load_json(path: Path, label: str) -> Any:
    if not path.exists():
        raise CaptureError(f"{label} is missing under {path.parent}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_optional_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def delete_variant(variant_path: Path, *, root: Path = dfx.PDF_REGRESSION_ROOT) -> bool:
    """Remove one captured variant dir; refuses to escape the pdf regression root."""
    target = variant_path.resolve()
    resolved_root = root.resolve()
    try:
        target.relative_to(resolved_root)
    except ValueError:
        return False
    if target == resolved_root or not target.exists():
        return False
    shutil.rmtree(target)
    return True
