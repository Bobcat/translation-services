"""Layout-evidence semantics (app.grouping.layout): the document gate, column clustering
with spanner refusal, and preserve-region assignment. Detection itself (the PaddleX model)
is not exercised here — these are the pure functions align consumes."""
from __future__ import annotations

from app.grouping.layout import cell_columns
from app.grouping.layout import document_gate
from app.grouping.layout import preserved_cell_indices


def _region(label: str, score: float, x0: float, y0: float, x1: float, y1: float) -> dict:
    return {"label": label, "score": score, "coordinate": [x0, y0, x1, y1]}


def _cell(left: float, top: float, width: float = 100, height: float = 20) -> dict:
    return {"text": "x", "bbox": {"left": left, "top": top, "width": width, "height": height}}


def test_document_gate_opens_on_documents_and_stays_closed_on_scenes() -> None:
    doc_regions = [
        _region("text", 0.95, 300, 100, 700, 300),
        _region("text", 0.92, 300, 320, 700, 500),
        _region("aside_text", 0.88, 40, 100, 250, 500),
    ]
    doc_cells = [_cell(320, 120), _cell(320, 340), _cell(60, 150)]
    assert document_gate(doc_regions, doc_cells)

    # a scene photo: the model itself reports one low-score region -> closed
    scene_regions = [_region("text", 0.53, 100, 100, 600, 900)]
    assert not document_gate(scene_regions, doc_cells)

    # enough regions but the cells live elsewhere (loose scene text) -> closed
    assert not document_gate(doc_regions, [_cell(900, 900), _cell(950, 950), _cell(320, 120)])


def test_document_gate_ignores_screenshot_cells_in_the_coverage_rule() -> None:
    # A quick-guide page: plenty of confident text regions, but most CELLS sit inside
    # screenshots. Those are preserve territory, not evidence against document-ness — the
    # coverage rule judges only the free cells.
    regions = [
        _region("text", 0.95, 40, 100, 400, 200),
        _region("text", 0.92, 40, 220, 400, 320),
        _region("paragraph_title", 0.90, 40, 40, 400, 80),
        _region("image", 0.90, 450, 40, 1000, 900),
    ]
    cells = [_cell(60, 120), _cell(60, 240), _cell(60, 50)] + [
        _cell(500 + i * 40, 100 + i * 60, width=60) for i in range(9)  # inside the screenshot
    ]
    assert document_gate(regions, cells)


def test_cell_columns_two_columns_and_spanner_refusal() -> None:
    # two established columns plus a full-width title that overlaps both: the title must
    # not fuse them (spanner), and cells inside it get no column
    regions = [
        _region("text", 0.9, 40, 200, 480, 400),
        _region("text", 0.9, 40, 420, 480, 600),
        _region("text", 0.9, 520, 200, 960, 400),
        _region("text", 0.9, 520, 420, 960, 600),
        _region("doc_title", 0.9, 60, 40, 940, 120),  # spans both columns
    ]
    cells = [_cell(60, 220), _cell(540, 220), _cell(100, 60)]
    columns = cell_columns(regions, cells)
    assert columns is not None
    left, right, title = columns
    assert left is not None and right is not None
    assert left != right
    assert title is None  # inside the spanner: no column

    # a single column (all regions overlap in x) -> None: keep the global estimate
    single = [
        _region("text", 0.9, 40, 100, 900, 300),
        _region("text", 0.9, 60, 320, 880, 500),
    ]
    assert cell_columns(single, cells) is None


def test_cell_columns_wide_header_line_does_not_glue_young_columns() -> None:
    # An academic first page: a wide centred author line brushes BOTH columns while each
    # column still has one region — the spanner refusal cannot protect them yet (it needs
    # established columns), so the fuse threshold must: brushing overlap (<50% of the
    # narrower side) does not join. The columns then establish and the full-width title is
    # refused as a spanner as usual.
    regions = [
        _region("text", 0.9, 200, 700, 880, 1200),     # left column, upper
        _region("text", 0.9, 205, 1250, 878, 1800),    # left column, lower
        _region("text", 0.9, 920, 700, 1600, 1200),    # right column, upper
        _region("text", 0.9, 923, 1250, 1602, 1800),   # right column, lower
        _region("text", 0.7, 430, 340, 1370, 390),     # author line: brushes both
        _region("doc_title", 0.9, 300, 100, 1500, 200),  # full-width title
    ]
    cells = [
        _cell(300, 800),    # left column
        _cell(1000, 800),   # right column
        _cell(600, 350),    # author line
    ]
    columns = cell_columns(regions, cells)
    assert columns is not None  # the two real columns survived
    left, right, author = columns
    assert left is not None and right is not None and left != right


def test_preserved_cells_inside_confident_image_or_chart_regions() -> None:
    regions = [
        _region("chart", 0.90, 100, 100, 500, 400),
        _region("image", 0.60, 600, 100, 900, 400),  # below preserve confidence
    ]
    cells = [_cell(200, 200), _cell(700, 200), _cell(950, 200)]
    assert preserved_cell_indices(regions, cells) == {0}
