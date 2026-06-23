"""Build a fixture + snapshot from a completed request, and report regression status for an image.

Shared by ``scripts/capture_fixture.py`` (fetches the request over HTTP) and the
``/v1/regression/*`` endpoints (read the request record in-process). Freezes exactly the result
that ran — no re-run of the VLM / translator.
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

from app.core.config import OcrSettings
from app.regression import fixture as fx
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


def build_fixture(response: dict[str, Any], *, source_bytes: bytes) -> fx.Fixture:
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
        image_sha256=fx.sha256(source_bytes),
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


def graft_grouping_inputs(response: dict[str, Any], grouping_response: dict[str, Any]) -> dict[str, Any]:
    """Return ``response`` with the frozen grouping inputs the replay needs — OCR ``cells``,
    ``ignored_cell_ids``, the grouping VLM call (→ ``raw_hint``) and ``grouping_model`` — copied from
    ``grouping_response``. A re-translate response carries only the new translations + render; those
    grouping inputs live on the run that actually did the OCR/grouping. Translations, target language
    and request flags stay from ``response`` (the re-translate)."""
    g_ocr = grouping_response.get("ocr") or {}
    merged_ocr = dict(response.get("ocr") or {})
    merged_ocr["cells"] = g_ocr.get("cells") or []
    merged_ocr["ignored_cell_ids"] = g_ocr.get("ignored_cell_ids") or []
    grouping_calls = [
        call for call in (grouping_response.get("llm_calls") or [])
        if "grouping" in str(call.get("role") or "").lower()
    ]
    merged_meta = dict(response.get("metadata") or {})
    merged_meta["grouping_model"] = (
        str((grouping_response.get("metadata") or {}).get("grouping_model") or "")
        or str(merged_meta.get("grouping_model") or "")
    )
    out = dict(response)
    out["ocr"] = merged_ocr
    out["llm_calls"] = list(response.get("llm_calls") or []) + grouping_calls
    out["metadata"] = merged_meta
    return out


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
    source_bytes: bytes,
    source_suffix: str,
    name: str,
    variant: str | None = None,
    regression_root: Path = REGRESSION_ROOT,
) -> dict[str, Any]:
    """Build and persist a self-contained fixture under ``<name>/<target_lang>/<variant>``: the
    frozen inputs, the snapshot, the approved render, AND the exact ``source.<ext>`` the render ran
    on (the canonical upload). Replay renders on that source, so the fixture never depends on the
    ``testset/`` file matching what was captured. ``variant`` is auto-assigned when not given."""
    fixture = build_fixture(response, source_bytes=source_bytes)
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
    # The approved render, for human inspection (not used in the diff — that stays re-OCR based).
    (variant_path / "snapshot.png").write_bytes(rendered_png)
    # The exact image the render ran on, so replay is faithful regardless of testset/.
    suffix = source_suffix if source_suffix.startswith(".") else f".{source_suffix or 'png'}"
    (variant_path / f"source{suffix}").write_bytes(source_bytes)
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


def list_fixtures(
    *,
    regression_root: Path = REGRESSION_ROOT,
    testset_root: Path = TESTSET_ROOT,
) -> list[dict[str, Any]]:
    """The full inventory the admin view lists: one entry per image, its fixtures grouped per
    language with light metadata (no replay, no OCR)."""
    out: list[dict[str, Any]] = []
    if not regression_root.exists():
        return out
    for name_dir in sorted(p for p in regression_root.iterdir() if p.is_dir()):
        langs: dict[str, list[dict[str, Any]]] = {}
        for lang_dir in sorted(p for p in name_dir.iterdir() if p.is_dir()):
            variants: list[dict[str, Any]] = []
            for variant_dir in sorted(p for p in lang_dir.iterdir() if p.is_dir()):
                try:
                    fixture_data = json.loads((variant_dir / "fixture.json").read_text())
                    snapshot_data = json.loads((variant_dir / "snapshot.json").read_text())
                except (OSError, ValueError):
                    continue
                variants.append({
                    "variant": variant_dir.name,
                    "target_lang": fixture_data.get("target_lang") or lang_dir.name,
                    "units": len(fixture_data.get("translations") or {}),
                    "reocr_rows": len(snapshot_data.get("reocr") or []),
                    "has_snapshot_png": (variant_dir / "snapshot.png").exists(),
                })
            if variants:
                langs[lang_dir.name] = variants
        if langs:
            out.append({
                "name": name_dir.name,
                "in_testset": testset_image(name_dir.name, testset_root=testset_root) is not None,
                "langs": langs,
            })
    return out


def delete_path(
    name: str,
    lang: str | None = None,
    variant: str | None = None,
    *,
    regression_root: Path = REGRESSION_ROOT,
) -> bool:
    """Cascade-delete a name / lang / variant dir. Refuses to escape ``regression_root`` or to
    delete the root itself."""
    segments = [name, *([lang] if lang else []), *([variant] if variant else [])]
    target = regression_root.joinpath(*segments).resolve()
    root = regression_root.resolve()
    try:
        target.relative_to(root)
    except ValueError:
        return False
    if target == root or not target.exists():
        return False
    shutil.rmtree(target)
    return True
