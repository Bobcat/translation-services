"""Run one fixture: replay -> re-OCR -> diff against its snapshot.

Shared by ``scripts/regress.py`` and the ``POST /v1/regression/run`` endpoint, so the CLI and the
admin view agree exactly. On a failure the current render is dropped next to the snapshot as
``actual.png`` (removed on a pass) for a visual snapshot-vs-actual comparison.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from app.core.config import OcrSettings
from app.regression import fixture as fx
from app.regression.capture import TESTSET_ROOT
from app.regression.capture import testset_image
from app.regression.compare import diff_reocr
from app.regression.compare import diff_units
from app.regression.image import canonical_bytes
from app.regression.replay import replay_fixture
from app.regression.snapshot import reocr_rows


def run_variant(
    ocr_settings: OcrSettings,
    *,
    variant_path: Path,
    name: str,
    testset_root: Path = TESTSET_ROOT,
) -> dict[str, Any]:
    """``{passed, diffs, has_actual}`` for one ``<name>/<lang>/<variant>``."""
    fixture, snapshot = fx.load(variant_path)
    image_path = testset_image(name, testset_root=testset_root)
    if image_path is None:
        return {"passed": False, "diffs": [f"testset image '{name}' not found"], "has_actual": False}
    canonical = canonical_bytes(image_path)
    if fx.sha256(canonical) != fixture.image_sha256:
        return {"passed": False, "diffs": ["image sha256 mismatch — fixture is stale"], "has_actual": False}

    with tempfile.NamedTemporaryFile(suffix=image_path.suffix) as handle:
        handle.write(canonical)
        handle.flush()
        actual_units, actual_ignored, rendered = replay_fixture(Path(handle.name), fixture)
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
