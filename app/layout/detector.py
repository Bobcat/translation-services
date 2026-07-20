"""The PP-DocLayout detection engine: page image in, labelled regions out.

``detect_layout_regions`` runs the model over one page render (~30-80ms warm on GPU; the
pipeline starts it in parallel with the multi-second grouping-VLM call, so its wall-clock
cost is hidden). Detection failure returns ``[]`` — layout is auxiliary evidence everywhere,
never a gate on a job.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

_MODEL_NAME = "PP-DocLayout_plus-L"

# Engine cache + locks, the app.ocr.paddleocr pattern: the predictor is not assumed thread-safe,
# so prediction serialises on one lock; building the engine (~3.3s + ~770 MiB VRAM, lazy on first
# use) must not block a warm predict, hence a separate build lock with double-checked caching.
_ENGINE: Any = None
_BUILD_LOCK = threading.Lock()
_PREDICT_LOCK = threading.Lock()

# Labels that are document FLOW text: evidence for the gate and members of column clustering.
TEXT_LABELS = {
    "text", "content", "paragraph_title", "doc_title", "abstract", "aside_text",
    "list", "header", "footer", "figure_title", "table_title", "reference",
}
# Labels that form a BODY COLUMN — the subset of TEXT_LABELS whose region marks a strip of
# running text with its own left and right margin. Deliberately narrower: a doc/figure/table
# title, a header and a footer are flow text but SPAN or stand apart, so they must never set a
# column's margin (measured: those are exactly the elements that make a page read as one
# column; see app.replacement.layout.bands).
COLUMN_LABELS = {
    "text", "content", "paragraph_title", "abstract", "aside_text", "list", "reference",
}
# Labels whose inner text keeps its original pixels (see app.layout.evidence).
PRESERVE_LABELS = {"image", "chart"}
# Labels that are document structure but not flow columns; their cells count as document
# evidence for the gate yet keep the global position estimate (a table has its own geometry).
STRUCTURE_LABELS = {"table"}


def detect_layout_regions(input_path: Path, *, threshold: float | None = None) -> list[dict[str, Any]]:
    """``[{"label", "score", "coordinate": [x0, y0, x1, y1]}]`` — ``[]`` on ANY failure (missing
    model, no GPU, decode error): layout evidence is optional, the job must not fail on it.

    ``threshold`` overrides the model's default detection threshold (0.5) and exists for the
    benchmark measurement layer ONLY (see the detector appendix in
    docs/pdf-benchmark-regression-design.md). The pipeline always calls without it: the kwarg is
    NOT a post-hoc score filter — it feeds the model's internal postprocess, and lowering it was
    measured to DROP full-page image regions on photo/scan pages (14 of 265 testset pages), which
    would flip the preserve/gate behaviour in ``app.layout.evidence``. Same for ``layout_nms``,
    measured to suppress those regions too; neither may silently change what align sees."""
    try:
        engine = _get_engine()
        kwargs = {} if threshold is None else {"threshold": threshold}
        with _PREDICT_LOCK:
            output = list(engine.predict(str(input_path), batch_size=1, **kwargs))
        boxes = (output[0].get("boxes") or []) if output else []
        return [
            {
                "label": str(box.get("label") or ""),
                "score": float(box.get("score") or 0.0),
                "coordinate": [float(v) for v in (box.get("coordinate") or [])[:4]],
            }
            for box in boxes
            if len(box.get("coordinate") or []) >= 4
        ]
    except Exception:  # noqa: BLE001 - auxiliary evidence, never job-fatal
        return []


def _get_engine() -> Any:
    global _ENGINE
    if _ENGINE is None:
        with _BUILD_LOCK:
            if _ENGINE is None:
                from paddlex import create_model

                _ENGINE = create_model(_MODEL_NAME)
    return _ENGINE
