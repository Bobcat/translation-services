#!/usr/bin/env python3
"""Capture a fixture + snapshot from a completed request on the live service.

The CLI precursor of the future POST /v1/regression/fixtures endpoint: it freezes exactly the
result that just ran (no re-run). Everything is on the completed request — cells, the raw VLM
hint (in the call log), the per-unit translations, the rendered artifact.

    python scripts/capture_fixture.py --request-id <id> --name nike-ad --variant v1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.core.config import load_settings
from app.regression import fixture as fx
from app.regression.image import canonical_bytes
from app.regression.snapshot import build_snapshot

_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp")


def _raw_hint(llm_calls: list[dict]) -> str:
    for call in llm_calls:
        if "grouping" in str(call.get("role") or "").lower():
            return str((call.get("response") or {}).get("output_text") or "")
    return ""


def _testset_image(testset: Path, name: str) -> Path:
    for suffix in _IMAGE_SUFFIXES:
        candidate = testset / f"{name}{suffix}"
        if candidate.exists():
            return candidate
    raise SystemExit(f"testset image '{name}' not found under {testset}/")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-id", required=True)
    parser.add_argument("--name", required=True, help="testset stem to store the fixture under")
    parser.add_argument("--variant", default="v1")
    parser.add_argument("--base-url", default="http://127.0.0.1:8030")
    parser.add_argument("--testset", default="testset")
    parser.add_argument("--root", default="testset/_regression")
    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    result = httpx.get(f"{base}/v1/requests/{args.request_id}", timeout=30).json()
    if (result.get("state") or "") != "completed":
        raise SystemExit(f"request not completed (state={result.get('state')})")
    response = result.get("response") or {}
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

    image_path = _testset_image(Path(args.testset), args.name)
    fixture = fx.Fixture(
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

    rendered = httpx.get(f"{base}/v1/requests/{args.request_id}/artifacts/rendered", timeout=30).content
    snapshot = build_snapshot(
        load_settings().ocr,
        units=units,
        ignored_cells=ocr.get("ignored_cell_ids") or [],
        rendered_png=rendered,
        target_lang=fixture.target_lang,
    )

    variant_path = fx.variant_dir(Path(args.root), args.name, args.variant)
    fx.save(variant_path, fixture, snapshot)
    print(f"wrote {variant_path}/  (units={len(fixture.translations)}, reocr_rows={len(snapshot.reocr)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
