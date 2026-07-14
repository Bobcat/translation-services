"""Run one fixture: replay -> re-OCR -> diff against its snapshot.

The fixture is self-contained — it renders on its own ``source.<ext>`` (the exact canonical bytes
the capture ran on), so a run never depends on the ``testset/`` file. Shared by
``scripts/regress.py`` and the ``/v1/regression/{run,resnapshot}`` endpoints. On a failure the
current render is dropped next to the snapshot as ``actual.png`` (removed on a pass).
"""
from __future__ import annotations

import time
from typing import Any

from app.core.config import OcrSettings
from app.regression import fixture as fx
from app.regression.compare import diff_units
from app.regression.compare import reocr_mismatches
from app.regression.replay import replay_fixture
from app.regression.snapshot import reocr_rows


def _resolve_source(variant_path) -> tuple[object, str | None]:
    """``(source_path, error)`` — the fixture's own source image, with an integrity check."""
    fixture, _ = fx.load(variant_path)
    source = fx.source_path(variant_path)
    if source is None:
        return None, "fixture has no source image"
    if fx.sha256(source.read_bytes()) != fixture.image_sha256:
        return None, "source image sha mismatch (corrupted fixture)"
    return source, None


def run_variant(ocr_settings: OcrSettings, *, variant_path) -> dict[str, Any]:
    """``{passed, diffs, has_actual}`` for one ``<name>/<lang>/<variant>``."""
    fixture, snapshot = fx.load(variant_path)
    source, error = _resolve_source(variant_path)
    if error:
        return {"passed": False, "diffs": [error], "has_actual": False}

    actual_units, actual_ignored, rendered, timings = replay_fixture(source, fixture)
    reocr_started = time.perf_counter()
    actual_rows = reocr_rows(ocr_settings, rendered, fixture.target_lang)
    timings = {**timings, "reocr_ms": (time.perf_counter() - reocr_started) * 1000.0}

    reocr_diffs, boxes = reocr_mismatches(snapshot.reocr, actual_rows)
    diffs = (
        diff_units(snapshot.expected_units, actual_units, snapshot.ignored_cells, actual_ignored)
        + reocr_diffs
    )
    actual_png = variant_path / "actual.png"
    snapshot_diff_png = variant_path / "snapshot_diff.png"
    if diffs:
        actual_png.write_bytes(rendered)
        _write_snapshot_diff(variant_path, boxes)
    else:
        for stale in (actual_png, snapshot_diff_png):
            if stale.exists():
                stale.unlink()
    timings = {key: round(value, 1) for key, value in timings.items()}
    return {"passed": not diffs, "diffs": diffs, "has_actual": bool(diffs), "timings": timings}


def resnapshot(ocr_settings: OcrSettings, *, variant_path) -> dict[str, Any]:
    """Re-baseline a variant: replay it and overwrite snapshot.json + snapshot.png with the current
    output (the fixture inputs and source stay). Accepts a deliberate render/align change."""
    fixture, _ = fx.load(variant_path)
    source, error = _resolve_source(variant_path)
    if error:
        return {"ok": False, "error": error}

    actual_units, actual_ignored, rendered, _timings = replay_fixture(source, fixture)
    snapshot = fx.Snapshot(
        expected_units=actual_units,
        ignored_cells=actual_ignored,
        reocr=reocr_rows(ocr_settings, rendered, fixture.target_lang),
    )
    fx.save_snapshot(variant_path, snapshot)
    (variant_path / "snapshot.png").write_bytes(rendered)
    for stale in (variant_path / "actual.png", variant_path / "snapshot_diff.png"):
        if stale.exists():
            stale.unlink()
    return {"ok": True}


# Box colours for the reviewer's marked-up snapshot: where a segment disappeared or moved
# (red), and where an extra segment sits in the actual (orange).
_DIFF_BOX_COLORS = {"missing": (220, 30, 30), "moved": (220, 30, 30), "extra": (235, 140, 20)}
_DIFF_BOX_PAD = 6
_DIFF_BOX_WIDTH = 3


def _write_snapshot_diff(variant_path, boxes: list[dict[str, Any]]) -> None:
    """``snapshot_diff.png``: the snapshot with a box around every mismatched re-OCR segment,
    so a reviewer flipping snapshot/actual sees WHERE to look instead of searching. Only the
    snapshot copy is marked — the actual stays clean for judging (and re-baselining). Align-only
    failures yield no boxes; the unmarked copy is still written so the viewer never 404s."""
    snapshot_png = variant_path / "snapshot.png"
    if not snapshot_png.exists():
        return
    from PIL import Image
    from PIL import ImageDraw

    image = Image.open(snapshot_png).convert("RGB")
    draw = ImageDraw.Draw(image)
    for box in boxes:
        left = int(box["left"]) - _DIFF_BOX_PAD
        top = int(box["top"]) - _DIFF_BOX_PAD
        right = int(box["left"]) + int(box["width"]) + _DIFF_BOX_PAD
        bottom = int(box["top"]) + int(box["height"]) + _DIFF_BOX_PAD
        draw.rectangle((left, top, right, bottom),
                       outline=_DIFF_BOX_COLORS.get(box.get("kind"), (220, 30, 30)),
                       width=_DIFF_BOX_WIDTH)
    image.save(variant_path / "snapshot_diff.png")
