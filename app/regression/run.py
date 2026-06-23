"""Run one fixture: replay -> re-OCR -> diff against its snapshot.

The fixture is self-contained — it renders on its own ``source.<ext>`` (the exact canonical bytes
the capture ran on), so a run never depends on the ``testset/`` file. Shared by
``scripts/regress.py`` and the ``/v1/regression/{run,resnapshot}`` endpoints. On a failure the
current render is dropped next to the snapshot as ``actual.png`` (removed on a pass).
"""
from __future__ import annotations

from typing import Any

from app.core.config import OcrSettings
from app.regression import fixture as fx
from app.regression.compare import diff_reocr
from app.regression.compare import diff_units
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

    actual_units, actual_ignored, rendered = replay_fixture(source, fixture)
    actual_rows = reocr_rows(ocr_settings, rendered, fixture.target_lang)

    diffs = (
        diff_units(snapshot.expected_units, actual_units, snapshot.ignored_cells, actual_ignored)
        + diff_reocr(snapshot.reocr, actual_rows)
    )
    actual_png = variant_path / "actual.png"
    if diffs:
        actual_png.write_bytes(rendered)
    elif actual_png.exists():
        actual_png.unlink()
    return {"passed": not diffs, "diffs": diffs, "has_actual": bool(diffs)}


def resnapshot(ocr_settings: OcrSettings, *, variant_path) -> dict[str, Any]:
    """Re-baseline a variant: replay it and overwrite snapshot.json + snapshot.png with the current
    output (the fixture inputs and source stay). Accepts a deliberate render/align change."""
    fixture, _ = fx.load(variant_path)
    source, error = _resolve_source(variant_path)
    if error:
        return {"ok": False, "error": error}

    actual_units, actual_ignored, rendered = replay_fixture(source, fixture)
    snapshot = fx.Snapshot(
        expected_units=actual_units,
        ignored_cells=actual_ignored,
        reocr=reocr_rows(ocr_settings, rendered, fixture.target_lang),
    )
    fx.save_snapshot(variant_path, snapshot)
    (variant_path / "snapshot.png").write_bytes(rendered)
    actual_png = variant_path / "actual.png"
    if actual_png.exists():
        actual_png.unlink()
    return {"ok": True}
