"""Build a fixture + snapshot from a completed request, and report regression status for an image.

Shared by ``scripts/capture_fixture.py`` (fetches the request over HTTP) and the
``/v1/regression/*`` endpoints (read the request record in-process). Freezes exactly the result
that ran — no re-run of the VLM / translator.
"""
from __future__ import annotations

import json
import re
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


def _next_variant(lang_dir: Path) -> str:
    """The next free ``vN`` under a ``<name>/<lang>`` dir (max existing + 1, so a deleted variant
    never collides)."""
    nums = []
    if lang_dir.exists():
        for child in lang_dir.iterdir():
            match = re.fullmatch(r"v(\d+)", child.name)
            if child.is_dir() and match:
                nums.append(int(match.group(1)))
    return f"v{(max(nums) + 1) if nums else 1}"


def _fixture_key(fixture: fx.Fixture) -> str:
    """A content hash of the fixture's frozen inputs. The snapshot is a deterministic function of
    these, so equal keys mean a true duplicate — no need to hash the (re-OCR-derived) snapshot."""
    return fx.sha256(json.dumps(fixture.to_dict(), sort_keys=True, ensure_ascii=False).encode("utf-8"))


def _find_duplicate(lang_dir: Path, key: str) -> str | None:
    """The variant under ``lang_dir`` whose fixture matches ``key``, else None. Cheap (small JSON
    per existing variant) and runs BEFORE the re-OCR, so a duplicate capture skips it entirely."""
    if not lang_dir.exists():
        return None
    for child in sorted(lang_dir.iterdir()):
        fixture_file = child / "fixture.json"
        if not (child.is_dir() and fixture_file.exists()):
            continue
        try:
            existing = fx.Fixture.from_dict(json.loads(fixture_file.read_text()))
        except Exception:  # noqa: BLE001 - a malformed fixture is just skipped
            continue
        if _fixture_key(existing) == key:
            return child.name
    return None


def capture(
    ocr_settings: OcrSettings,
    *,
    response: dict[str, Any],
    rendered_png: bytes,
    image_path: Path,
    name: str,
    variant: str | None = None,
    regression_root: Path = REGRESSION_ROOT,
) -> dict[str, Any]:
    """Build and persist a fixture+snapshot under ``<name>/<target_lang>/<variant>``. The target
    language is its own path level (it shapes the render, not the align); the variant counter runs
    per language. ``variant`` is auto-assigned (next free ``vN``) when not given."""
    fixture = build_fixture(response, image_path=image_path)
    lang = fixture.target_lang or "unknown"
    lang_dir = regression_root / name / lang
    # Auto-capture skips a true duplicate (same frozen inputs) — and the check runs before the
    # re-OCR, so it costs nothing. An explicit ``variant`` forces a (re-)write.
    if variant is None:
        duplicate = _find_duplicate(lang_dir, _fixture_key(fixture))
        if duplicate is not None:
            return {
                "path": str(lang_dir / duplicate),
                "name": name,
                "target_lang": lang,
                "variant": duplicate,
                "duplicate": True,
            }
    resolved_variant = variant or _next_variant(lang_dir)
    ocr = response.get("ocr") or {}
    snapshot = build_snapshot(
        ocr_settings,
        units=ocr.get("translation_units") or [],
        ignored_cells=ocr.get("ignored_cell_ids") or [],
        rendered_png=rendered_png,
        target_lang=fixture.target_lang,
    )
    variant_path = lang_dir / resolved_variant
    fx.save(variant_path, fixture, snapshot)
    # The approved render itself, for human inspection (not used in the diff — that stays re-OCR
    # based). Lets an admin view show what a snapshot looks like before deleting / re-approving.
    (variant_path / "snapshot.png").write_bytes(rendered_png)
    return {
        "path": str(variant_path),
        "name": name,
        "target_lang": lang,
        "variant": resolved_variant,
        "duplicate": False,
        "units": len(fixture.translations),
        "reocr_rows": len(snapshot.reocr),
    }


def status(
    name: str,
    *,
    regression_root: Path = REGRESSION_ROOT,
    testset_root: Path = TESTSET_ROOT,
) -> dict[str, Any]:
    """Whether the image is in the testset and its fixtures per target language — the workbench badges."""
    name_dir = regression_root / name
    langs: dict[str, list[str]] = {}
    if name_dir.exists():
        for lang_dir in sorted(p for p in name_dir.iterdir() if p.is_dir()):
            variants = sorted(
                p.name for p in lang_dir.iterdir()
                if p.is_dir() and (p / "fixture.json").exists()
            )
            if variants:
                langs[lang_dir.name] = variants
    return {
        "name": name,
        "in_testset": testset_image(name, testset_root=testset_root) is not None,
        "fixture_count": sum(len(v) for v in langs.values()),
        "langs": langs,
    }
