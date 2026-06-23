#!/usr/bin/env python3
"""Regression run: replay every fixture under ``testset/_regression/`` and diff against its snapshot.

Each ``<name>/<lang>/<variant>`` is replayed (parse -> align -> render), the render re-OCR'd, and
compared (align exactly, render via re-OCR). Exits non-zero if any variant fails. Delegates to
``app.regression.run`` so the CLI and the /v1/regression/run endpoint behave identically.

    python scripts/regress.py
    python scripts/regress.py --root testset/_regression
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.core.config import load_settings
from app.regression.run import run_variant


def _variant_dirs(root: Path) -> list[tuple[str, str, Path]]:
    """``(name, label, path)`` for every ``<name>/<lang>/<variant>`` holding a fixture+snapshot."""
    out: list[tuple[str, str, Path]] = []
    for name_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for lang_dir in sorted(p for p in name_dir.iterdir() if p.is_dir()):
            for variant_dir in sorted(p for p in lang_dir.iterdir() if p.is_dir()):
                if (variant_dir / "fixture.json").exists() and (variant_dir / "snapshot.json").exists():
                    out.append((name_dir.name, f"{lang_dir.name}/{variant_dir.name}", variant_dir))
    return out


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
    for name, label, variant_path in variants:
        try:
            diffs = run_variant(ocr_settings, variant_path=variant_path, name=name, testset_root=Path(args.testset))["diffs"]
        except Exception as exc:  # noqa: BLE001 - report and continue
            diffs = [f"replay error: {exc}"]
        if diffs:
            failures += 1
            print(f"FAIL {name}/{label}")
            for line in diffs:
                print(f"     - {line}")
        else:
            print(f"PASS {name}/{label}")

    print(f"\n{len(variants) - failures}/{len(variants)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
