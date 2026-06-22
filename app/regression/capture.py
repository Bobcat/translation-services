"""Build a fixture + snapshot from a completed request, and report regression status for an image.

Shared by ``scripts/capture_fixture.py`` (fetches the request over HTTP) and the
``/v1/regression/*`` endpoints (read the request record in-process). Freezes exactly the result
that ran — no re-run of the VLM / translator.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from app.core.config import OcrSettings
from app.regression import fixture as fx
from app.regression.image import canonical_bytes
from app.regression.snapshot import build_snapshot

TESTSET_ROOT = Path("testset")
REGRESSION_ROOT = Path("testset/_regression")
_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp")


def testset_image(name: str, *, testset_root: Path = TESTSET_ROOT) -> Path | None:
    for suffix in _IMAGE_SUFFIXES:
        candidate = testset_root / f"{name}{suffix}"
        if candidate.exists():
            return candidate
    return None


def _raw_hint(llm_calls: list[dict[str, Any]]) -> str:
    for call in llm_calls:
        if "grouping" in str(call.get("role") or "").lower():
            return str((call.get("response") or {}).get("output_text") or "")
    return ""


def build_fixture(response: dict[str, Any], *, image_path: Path) -> fx.Fixture:
    ocr = response.get("ocr") or {}
    metadata = response.get("metadata") or {}
    units = ocr.get("translation_units") or []
    translations = {
        fx.anchor_key(unit): {
            "translated_text": unit.get("translated_text") or "",
            "field_translations": unit.get("field_translations"),
        }
        for unit in units
    }
    return fx.Fixture(
        image_sha256=fx.sha256(canonical_bytes(image_path)),
        cells=ocr.get("cells") or [],
        raw_hint=_raw_hint(response.get("llm_calls") or []),
        translations=translations,
        request_flags={
            "preserve_heuristic_text": bool(metadata.get("preserve_heuristic_text", True)),
            "preserve_unchanged_text": bool(metadata.get("preserve_unchanged_text", False)),
            "use_geometry_columns": bool(metadata.get("use_geometry_columns", True)),
        },
        grouping_model=str(metadata.get("grouping_model") or ""),
        target_lang=str(metadata.get("target_lang_code") or ""),
    )


def capture(
    ocr_settings: OcrSettings,
    *,
    response: dict[str, Any],
    rendered_png: bytes,
    image_path: Path,
    name: str,
    variant: str,
    regression_root: Path = REGRESSION_ROOT,
) -> dict[str, Any]:
    """Build and persist fixture.json + snapshot.json for one ``<name>/<variant>``."""
    fixture = build_fixture(response, image_path=image_path)
    ocr = response.get("ocr") or {}
    snapshot = build_snapshot(
        ocr_settings,
        units=ocr.get("translation_units") or [],
        ignored_cells=ocr.get("ignored_cell_ids") or [],
        rendered_png=rendered_png,
        target_lang=fixture.target_lang,
    )
    variant_path = fx.variant_dir(regression_root, name, variant)
    fx.save(variant_path, fixture, snapshot)
    return {"path": str(variant_path), "units": len(fixture.translations), "reocr_rows": len(snapshot.reocr)}


def status(
    name: str,
    *,
    regression_root: Path = REGRESSION_ROOT,
    testset_root: Path = TESTSET_ROOT,
) -> dict[str, Any]:
    """Whether the image is in the testset and how many fixtures it has — drives the workbench badges."""
    name_dir = regression_root / name
    variants = sorted(p.name for p in name_dir.iterdir() if p.is_dir()) if name_dir.exists() else []
    return {
        "name": name,
        "in_testset": testset_image(name, testset_root=testset_root) is not None,
        "fixture_count": len(variants),
        "variants": variants,
    }
