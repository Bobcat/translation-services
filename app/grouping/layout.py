"""Stage #5b layout evidence: document-structure regions from a PaddleX detection model.

``detect_layout_regions`` runs PP-DocLayout over the input image (~30-80ms warm on GPU; the
pipeline starts it in parallel with the multi-second grouping-VLM call, so its wall-clock cost
is hidden). The regions feed ``app.grouping.align`` with two things reading-order text matching
cannot provide:

  - COLUMNS: on a multi-column page (main text + a metadata sidebar; a two-column article) the
    cells of separate logical flows interleave in y, which breaks align's "lower on the page ->
    further down the hint list" position estimate. Region x-clustering recovers the flows.
  - PRESERVE: text inside a detected ``image``/``chart`` region (a screenshot's UI labels, a
    plot's axis/legend text) is not document flow text: translating it garbles a screenshot the
    reader must compare against the real UI, and re-painting inside a plot smears bars and
    gridlines. Those cells keep their original pixels (align routes them to ``ignored_cell_ids``).

Layout is auxiliary evidence, never a gate on the job: detection failure returns ``[]`` and the
pipeline behaves exactly as before. The ``document_gate`` below decides whether the evidence is
trusted AT ALL — it judges the layout model's own confidence (enough high-score text regions,
enough of the cells covered), NOT align's confidence. On scene photos and receipts the model
itself reports few/low-score regions, the gate stays closed, and behaviour is bit-for-bit the
pre-layout pipeline (measured closed on all 51 regression fixtures at introduction time).
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
# Labels whose inner text keeps its original pixels (see module docstring).
PRESERVE_LABELS = {"image", "chart"}
# Labels that are document structure but not flow columns; their cells count as document
# evidence for the gate yet keep the global position estimate (a table has its own geometry).
STRUCTURE_LABELS = {"table"}

_GATE_MIN_TEXT_REGIONS = 3   # this many confident text regions, or the page is no document
_GATE_REGION_SCORE = 0.7     # region confidence for the gate and for preserve
_ASSIGN_REGION_SCORE = 0.6   # region confidence for cell->region assignment
_GATE_MIN_FREE_COVER = 0.5   # of the cells outside preserve/table regions: fraction in text regions
_COLUMN_FUSE_OVERLAP = 0.5   # x-overlap needed to join a column, vs the narrower of the two
_COLUMN_FUSE_STRONG = 0.7    # the stricter bar when a region touches SEVERAL columns at once


def detect_layout_regions(input_path: Path) -> list[dict[str, Any]]:
    """``[{"label", "score", "coordinate": [x0, y0, x1, y1]}]`` — ``[]`` on ANY failure (missing
    model, no GPU, decode error): layout evidence is optional, the job must not fail on it."""
    try:
        engine = _get_engine()
        with _PREDICT_LOCK:
            output = list(engine.predict(str(input_path), batch_size=1))
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


def document_gate(regions: list[dict[str, Any]], cells: list[dict[str, Any]]) -> bool:
    """Whether the layout evidence is trustworthy for this page: enough confident text regions
    AND most of the FREE cells (outside preserve/table regions — a screenshot-heavy quick guide
    is still a document) sit inside them."""
    confident = [r for r in regions
                 if r["label"] in TEXT_LABELS and r["score"] >= _GATE_REGION_SCORE]
    if len(confident) < _GATE_MIN_TEXT_REGIONS:
        return False
    free = covered = 0
    for cell in cells:
        if (_containing_region(cell, regions, PRESERVE_LABELS | STRUCTURE_LABELS,
                               _GATE_REGION_SCORE) is not None):
            continue
        free += 1
        if _containing_region(cell, regions, TEXT_LABELS, _ASSIGN_REGION_SCORE) is not None:
            covered += 1
    return free > 0 and covered / free >= _GATE_MIN_FREE_COVER


def preserved_cell_indices(
    regions: list[dict[str, Any]], cells: list[dict[str, Any]]
) -> set[int]:
    """Indices of cells whose centre sits inside a confident preserve region."""
    return {
        index for index, cell in enumerate(cells)
        if _containing_region(cell, regions, PRESERVE_LABELS, _GATE_REGION_SCORE) is not None
    }


def cell_columns(
    regions: list[dict[str, Any]], cells: list[dict[str, Any]]
) -> list[int | None] | None:
    """Per cell, the id of the layout COLUMN it belongs to (``None`` = no column: outside every
    text region, or inside a spanner). Returns ``None`` when fewer than two columns form — a
    single-column page keeps the global position estimate untouched."""
    text_indices = [i for i, r in enumerate(regions)
                    if r["label"] in TEXT_LABELS and r["score"] >= _ASSIGN_REGION_SCORE]
    column_of_region = _cluster_columns(regions, text_indices)
    if len(set(column_of_region.values())) < 2:
        return None
    out: list[int | None] = []
    for cell in cells:
        region = _containing_region(cell, regions, TEXT_LABELS, _ASSIGN_REGION_SCORE)
        out.append(column_of_region.get(region) if region is not None else None)
    return out


def _cluster_columns(
    regions: list[dict[str, Any]], text_indices: list[int]
) -> dict[int, int]:
    """Cluster text regions into columns by x-interval overlap, narrowest region first. Joining
    needs SUBSTANTIAL overlap (>50% of the narrower of region/column): regions of one column
    share its margins, so a true member overlaps by ~its own width, while a wide centred
    header line (an author row on a two-column paper) merely brushes each column — without
    the threshold such a line glues the young columns together before the spanner refusal
    below can protect them. A region touching SEVERAL columns must clear a stricter bar per
    touch (70%): genuine fragments of one column nest almost fully into the pieces they
    reunite, a brushing header does not. The refusal handles the rest: a region that would
    FUSE two established columns (each already >=2 regions) — a full-width title or float —
    joins neither; without it one page-wide element melts every column into one."""
    ordered = sorted(
        text_indices,
        key=lambda i: regions[i]["coordinate"][2] - regions[i]["coordinate"][0],
    )
    columns: list[dict[str, Any]] = []  # {"x0", "x1", "members"}
    for i in ordered:
        x0, _, x1, _ = regions[i]["coordinate"]

        def _overlap_ratio(column: dict[str, Any]) -> float:
            span = min(x1 - x0, column["x1"] - column["x0"])
            if span <= 0:
                return 0.0
            return (min(x1, column["x1"]) - max(x0, column["x0"])) / span

        hits = [c for c in columns if _overlap_ratio(c) > _COLUMN_FUSE_OVERLAP]
        if len(hits) > 1:
            hits = [c for c in hits if _overlap_ratio(c) > _COLUMN_FUSE_STRONG]
        if not hits:
            columns.append({"x0": x0, "x1": x1, "members": [i]})
            continue
        if len(hits) > 1 and sum(len(c["members"]) >= 2 for c in hits) >= 2:
            continue  # spanner: bridges established columns, belongs to none
        merged = {
            "x0": min([x0] + [c["x0"] for c in hits]),
            "x1": max([x1] + [c["x1"] for c in hits]),
            "members": [m for c in hits for m in c["members"]] + [i],
        }
        columns = [c for c in columns if c not in hits] + [merged]
    assignment: dict[int, int] = {}
    for column_id, column in enumerate(columns):
        for member in column["members"]:
            assignment[member] = column_id
    return assignment


def _containing_region(
    cell: dict[str, Any],
    regions: list[dict[str, Any]],
    labels: set[str],
    min_score: float,
) -> int | None:
    """Index of the smallest qualifying region containing the cell's centre (smallest wins, so
    a nested sidebar box beats a page-wide box), else ``None``."""
    bbox = cell.get("bbox") or {}
    cx = float(bbox.get("left") or 0.0) + float(bbox.get("width") or 0.0) / 2.0
    cy = float(bbox.get("top") or 0.0) + float(bbox.get("height") or 0.0) / 2.0
    best: int | None = None
    best_area: float | None = None
    for index, region in enumerate(regions):
        if region["label"] not in labels or region["score"] < min_score:
            continue
        x0, y0, x1, y1 = region["coordinate"]
        if x0 <= cx <= x1 and y0 <= cy <= y1:
            area = (x1 - x0) * (y1 - y0)
            if best_area is None or area < best_area:
                best, best_area = index, area
    return best
