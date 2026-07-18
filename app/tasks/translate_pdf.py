"""Task pipeline: PDF in -> translated PDF out (phase 0: raster pages).

Every page takes the raster route: render at the analysis dpi -> the full
``translate_image`` pipeline (VLM hint, OCR, align, translate, render) -> the
rendered pages are reassembled into a PDF on the original page sizes (pt).
The per-page census (born-digital | scanned | hybrid) is recorded in the
document summary but does not change routing yet — text-layer cells for
born-digital pages are phase 1 (docs/pdf-translation-design.md §6, §8).

Pages run sequentially inside this one job: the GPU stages are serial anyway,
``checkpoint`` still cancels between stages, and ``progress`` reports each
finished page to the lifecycle record. Per-page artifacts are written to
``pages_root/page-NNN/`` as they complete (input.png, rendered.png,
grouping.json, translation.json, request.json, llm_calls.json), so a document job leaves
inspectable state per page and a later per-document retranslate can re-enter
from the cached grouping exactly like the image flow.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any
from typing import Callable

from app.core.config import AppSettings
from app.pdf.assemble import PageImage
from app.pdf.assemble import assemble_pdf
from app.pdf.document import profile_pdf
from app.pdf.raster import PageRasterizer
from app.pdf.textlayer import PageTextExtractor
from app.tasks.translate_image import run_translate_image_pipeline


# Per-page pipeline metrics summed into the document metrics (same key names,
# ``_total`` suffixed), so the stage cost breakdown survives aggregation.
_SUMMED_PAGE_METRICS = (
    "ocr_wall_ms",
    "grouping_wall_ms",
    "layout_wall_ms",
    "align_wall_ms",
    "translation_wall_ms",
    "replacement_wall_ms",
    "translation_unit_count",
    "translated_unit_count",
    "ocr_segment_count",
    "llm_pool_request_count",
)


def _bool_flag(request: dict[str, Any], key: str, *, default: bool) -> bool:
    value = request.get(key)
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)


@dataclass(frozen=True)
class TranslatePdfResult:
    rendered_pdf: bytes
    mime_type: str
    document: dict[str, Any]
    metadata: dict[str, Any]
    metrics: dict[str, float | int | None]


def run_translate_pdf_pipeline(
    *,
    settings: AppSettings,
    input_path: Path,
    request: dict[str, Any],
    pages_root: Path,
    checkpoint: Callable[[], None] = lambda: None,
    progress: Callable[[int, int], None] = lambda done, total: None,
) -> TranslatePdfResult:
    started_at = time.perf_counter()
    profile = profile_pdf(input_path, page_cap=settings.pdf.page_cap)
    pages_total = profile.page_count
    progress(0, pages_total)

    page_summaries: list[dict[str, Any]] = []
    assembled_pages: list[PageImage] = []
    metric_totals: dict[str, float] = {key: 0.0 for key in _SUMMED_PAGE_METRICS}

    use_text_layer = _bool_flag(request, "use_pdf_text_layer", default=True)
    with (
        PageRasterizer(input_path, dpi=settings.pdf.analysis_dpi) as rasterizer,
        PageTextExtractor(input_path, dpi=settings.pdf.analysis_dpi) as extractor,
    ):
        for page in profile.pages:
            checkpoint()
            page_dir = pages_root / f"page-{page.index + 1:03d}"
            page_dir.mkdir(parents=True, exist_ok=True)
            page_png = page_dir / "input.png"
            page_png.write_bytes(rasterizer.render_png(page.index))

            # Born-digital pages take their cells from the PDF text layer
            # (exact text + style, OCR skipped); scanned and hybrid pages keep
            # the OCR route — a hybrid's hidden layer is not trustworthy.
            page_cells = None
            cell_stats: dict[str, Any] = {"cell_source": "ocr"}
            if use_text_layer and page.page_class == "born-digital":
                extracted = extractor.cells_for_page(page.index)
                page_cells = extracted.cells
                cell_stats = {
                    "cell_source": "pdf_text_layer",
                    "cell_count": len(extracted.cells),
                    "dropped_protected_lines": extracted.dropped_protected_lines,
                    "dropped_rotated_lines": extracted.dropped_rotated_lines,
                    "stripped_marker_spans": extracted.stripped_marker_spans,
                }

            page_result = run_translate_image_pipeline(
                settings=settings,
                input_path=page_png,
                input_mime_type="image/png",
                request=request,
                checkpoint=checkpoint,
                cells=page_cells,
            )

            # An image-only page with nothing translated renders as its input.
            rendered_path = page_dir / "rendered.png"
            rendered_path.write_bytes(page_result.rendered_image or page_png.read_bytes())
            debug = page_result.debug or {}
            # "request" carries the RESOLVED request flags per page — what a document-fixture
            # capture freezes (app/regression/pdf/capture.py); defaults live in translate_image
            # only, so persisting the resolved values is the one honest record.
            for name in ("grouping", "translation", "request"):
                if debug.get(name) is not None:
                    (page_dir / f"{name}.json").write_text(
                        json.dumps(debug[name], ensure_ascii=False, indent=2), encoding="utf-8"
                    )
            calls = debug.get("llm_calls") or []
            if calls:
                (page_dir / "llm_calls.json").write_text(
                    json.dumps(calls, ensure_ascii=False, indent=2), encoding="utf-8"
                )

            page_metrics = {
                key: page_result.metrics.get(key)
                for key in _SUMMED_PAGE_METRICS + ("translate_image_total_wall_ms",)
            }
            for key in _SUMMED_PAGE_METRICS:
                value = page_result.metrics.get(key)
                if isinstance(value, (int, float)):
                    metric_totals[key] += float(value)
            page_summaries.append(
                {
                    "page": page.index + 1,
                    **page.to_dict(),
                    **cell_stats,
                    "metrics": page_metrics,
                    "artifacts": {
                        "input": str(page_png),
                        "rendered": str(rendered_path),
                    },
                }
            )
            assembled_pages.append(
                PageImage(png_path=rendered_path, width_pt=page.width_pt, height_pt=page.height_pt)
            )
            progress(page.index + 1, pages_total)

    checkpoint()
    assemble_started_at = time.perf_counter()
    rendered_pdf = assemble_pdf(assembled_pages)
    assemble_wall_ms = max(0.0, (time.perf_counter() - assemble_started_at) * 1000.0)

    document = {
        "page_count": pages_total,
        "pages_total": pages_total,
        "pages_done": pages_total,
        "analysis_dpi": int(settings.pdf.analysis_dpi),
        "pages": page_summaries,
    }
    metadata = {
        "source_lang_code": str(request.get("source_lang_code") or ""),
        "target_lang_code": str(request.get("target_lang_code") or ""),
        "page_count": pages_total,
        "analysis_dpi": int(settings.pdf.analysis_dpi),
        "page_classes": [page.page_class for page in profile.pages],
        "note": "Phase-0 raster route: every page rendered at the analysis dpi through the "
        "translate_image pipeline; output PDF carries the rendered raster pages.",
    }
    metrics: dict[str, float | int | None] = {
        f"{key}_total": round(value, 3) for key, value in metric_totals.items()
    }
    metrics.update(
        {
            "page_count": pages_total,
            "assemble_wall_ms": round(assemble_wall_ms, 3),
            "translate_pdf_total_wall_ms": round(
                max(0.0, (time.perf_counter() - started_at) * 1000.0), 3
            ),
        }
    )
    return TranslatePdfResult(
        rendered_pdf=rendered_pdf,
        mime_type="application/pdf",
        document=document,
        metadata=metadata,
        metrics=metrics,
    )
