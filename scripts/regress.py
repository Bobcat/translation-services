#!/usr/bin/env python3
"""Regression run: replay every fixture under ``testset/_regression/`` and diff against its snapshot.

For each ``<name>/<variant>/`` it canonicalises the testset image, verifies the fixture hash,
replays (parse -> align -> render), re-OCRs the render, and compares align exactly + render
behaviourally. Exits non-zero if any variant fails.

    python scripts/regress.py
    python scripts/regress.py --root testset/_regression
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.core.config import load_settings
from app.regression import fixture as fx
from app.regression.compare import diff_reocr
from app.regression.compare import diff_units
from app.regression.image import canonical_bytes
from app.regression.replay import replay_fixture
from app.regression.snapshot import reocr_rows

_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp")


def _testset_image(testset: Path, name: str) -> Path | None:
    for suffix in _IMAGE_SUFFIXES:
        candidate = testset / f"{name}{suffix}"
        if candidate.exists():
            return candidate
    return None


def _variant_dirs(root: Path) -> list[tuple[str, str, Path]]:
    """``(name, label, path)`` for every ``<name>/<lang>/<variant>`` holding a fixture+snapshot."""
    out: list[tuple[str, str, Path]] = []
    for name_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for lang_dir in sorted(p for p in name_dir.iterdir() if p.is_dir()):
            for variant_dir in sorted(p for p in lang_dir.iterdir() if p.is_dir()):
                if (variant_dir / "fixture.json").exists() and (variant_dir / "snapshot.json").exists():
                    out.append((name_dir.name, f"{lang_dir.name}/{variant_dir.name}", variant_dir))
    return out


def _run_one(ocr_settings, testset: Path, name: str, variant_path: Path) -> list[str]:
    fixture, snapshot = fx.load(variant_path)
    image_path = _testset_image(testset, name)
    if image_path is None:
        return [f"testset image '{name}' not found"]
    canonical = canonical_bytes(image_path)
    if fx.sha256(canonical) != fixture.image_sha256:
        return ["image sha256 mismatch — fixture is stale for this image"]

    with tempfile.NamedTemporaryFile(suffix=image_path.suffix) as handle:
        handle.write(canonical)
        handle.flush()
        actual_units, actual_ignored, rendered = replay_fixture(Path(handle.name), fixture)
    actual_rows = reocr_rows(ocr_settings, rendered, fixture.target_lang)

    diffs = (
        diff_units(snapshot.expected_units, actual_units, snapshot.ignored_cells, actual_ignored)
        + diff_reocr(snapshot.reocr, actual_rows)
    )
    # On a failure, drop the current render next to the snapshot so it can be eyeballed against
    # snapshot.png; remove a stale one on a pass.
    actual_png = variant_path / "actual.png"
    if diffs:
        actual_png.write_bytes(rendered)
    elif actual_png.exists():
        actual_png.unlink()
    return diffs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="testset/_regression")
    parser.add_argument("--testset", default="testset")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"no fixtures: {root} does not exist")
        return 0
    ocr_settings = load_settings().ocr
    variants = _variant_dirs(root)
    if not variants:
        print(f"no fixtures under {root}")
        return 0

    failures = 0
    for name, variant, variant_path in variants:
        try:
            diffs = _run_one(ocr_settings, Path(args.testset), name, variant_path)
        except Exception as exc:  # noqa: BLE001 - report and continue
            diffs = [f"replay error: {exc}"]
        label = f"{name}/{variant}"
        if diffs:
            failures += 1
            print(f"FAIL {label}")
            for line in diffs:
                print(f"     - {line}")
        else:
            print(f"PASS {label}")

    print(f"\n{len(variants) - failures}/{len(variants)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
