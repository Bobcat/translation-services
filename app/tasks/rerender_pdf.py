"""Re-entry pipeline: re-render a prior DOCUMENT run's pages with new render flags.

The document twin of :mod:`app.tasks.rerender_image`, and it delegates to it: every page
of the source run kept its own ``grouping.json`` / ``translation.json`` / ``input.png``,
so a page re-renders exactly like a standalone image re-render, and the pages are
reassembled onto the original page sizes. No VLM, OCR or translator call anywhere — so a
render-flag A/B on a document compares exactly the render, with zero run-variance.

A page the source run left untranslated (an image-only page has no cached units) carries
its rendered PNG through verbatim rather than failing the document.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any
from typing import Callable

from app.core.config import AppSettings
from app.pdf.assemble import PageImage
from app.pdf.assemble import assemble_pdf
from app.tasks.rerender_image import run_rerender_image_pipeline
from app.tasks.translate_pdf import TranslatePdfResult


def run_rerender_pdf_pipeline(
    *,
    settings: AppSettings,
    source_pages_root: Path,
    source_document: dict[str, Any],
    request: dict[str, Any],
    pages_root: Path,
    checkpoint: Callable[[], None] = lambda: None,
    progress: Callable[[int, int], None] = lambda done, total: None,
) -> TranslatePdfResult:
    started_at = time.perf_counter()
    source_pages = list(source_document.get("pages") or [])
    pages_total = len(source_pages)
    if not pages_total:
        raise RuntimeError("source document run has no pages to re-render")
    progress(0, pages_total)

    page_summaries: list[dict[str, Any]] = []
    assembled_pages: list[PageImage] = []
    replacement_wall_ms = 0.0

    for order, page in enumerate(source_pages):
        checkpoint()
        page_no = int(page.get("page") or order + 1)
        source_dir = source_pages_root / f"page-{page_no:03d}"
        page_dir = pages_root / f"page-{page_no:03d}"
        page_dir.mkdir(parents=True, exist_ok=True)

        source_png = source_dir / "input.png"
        if not source_png.exists():
            raise RuntimeError(f"source page {page_no} input is missing (it may have been pruned)")
        page_png = page_dir / "input.png"
        shutil.copyfile(source_png, page_png)

        grouping_path = source_dir / "grouping.json"
        translation_path = source_dir / "translation.json"
        rendered_path = page_dir / "rendered.png"
        if grouping_path.exists() and translation_path.exists():
            result = run_rerender_image_pipeline(
                settings=settings,
                input_path=page_png,
                source_grouping=json.loads(grouping_path.read_text(encoding="utf-8")),
                source_translation=json.loads(translation_path.read_text(encoding="utf-8")),
                request=request,
                checkpoint=checkpoint,
            )
            rendered_path.write_bytes(result.rendered_image or page_png.read_bytes())
            debug = result.debug or {}
            for name in ("grouping", "translation", "request"):
                if debug.get(name) is not None:
                    (page_dir / f"{name}.json").write_text(
                        json.dumps(debug[name], ensure_ascii=False, indent=2), encoding="utf-8"
                    )
            replacement_wall_ms += float(result.metrics.get("replacement_wall_ms") or 0.0)
        else:
            # Nothing was cached for this page (an image-only page): its previous render is
            # the page, and no render flag can change it.
            previous = source_dir / "rendered.png"
            rendered_path.write_bytes(
                previous.read_bytes() if previous.exists() else page_png.read_bytes()
            )

        summary = {key: value for key, value in page.items() if key != "artifacts"}
        summary["artifacts"] = {"input": str(page_png), "rendered": str(rendered_path)}
        page_summaries.append(summary)
        assembled_pages.append(
            PageImage(
                png_path=rendered_path,
                width_pt=float(page.get("width_pt") or 0.0),
                height_pt=float(page.get("height_pt") or 0.0),
            )
        )
        progress(page_no, pages_total)

    checkpoint()
    assemble_started_at = time.perf_counter()
    rendered_pdf = assemble_pdf(assembled_pages)
    assemble_wall_ms = max(0.0, (time.perf_counter() - assemble_started_at) * 1000.0)

    document = {
        "page_count": pages_total,
        "pages_total": pages_total,
        "pages_done": pages_total,
        "analysis_dpi": int(source_document.get("analysis_dpi") or 0),
        "pages": page_summaries,
    }
    metadata = {
        "source_lang_code": str(request.get("source_lang_code") or ""),
        "target_lang_code": str(request.get("target_lang_code") or ""),
        "page_count": pages_total,
        "analysis_dpi": int(source_document.get("analysis_dpi") or 0),
        "translation_source": "cached_rerender",
        "source_request_id": str(request.get("source_request_id") or ""),
        "render_size_mode": str(request.get("render_size_mode") or ""),
        "erase_fill_mode": str(request.get("erase_fill_mode") or ""),
        "width_fit_mode": str(request.get("width_fit_mode") or ""),
        "size_metric_mode": str(request.get("size_metric_mode") or ""),
        "size_cohort_mode": str(request.get("size_cohort_mode") or ""),
        "note": "Re-render of a prior document run's cached page translations with new render "
        "flags (no LLM call).",
    }
    metrics: dict[str, float | int | None] = {
        "page_count": pages_total,
        "replacement_wall_ms_total": round(replacement_wall_ms, 3),
        "assemble_wall_ms": round(assemble_wall_ms, 3),
        "translate_pdf_total_wall_ms": round(
            max(0.0, (time.perf_counter() - started_at) * 1000.0), 3
        ),
    }
    return TranslatePdfResult(
        rendered_pdf=rendered_pdf,
        mime_type="application/pdf",
        document=document,
        metadata=metadata,
        metrics=metrics,
    )
