#!/usr/bin/env python3
"""Run grouping over the testset via the live service and collect overlays.

Submits every image in ``testset/`` to a running translation-services instance,
polls to completion, and saves the grouping overlay + units JSON per image into
``testset/_inspection/`` so the results can be eyeballed side by side. Prints a
summary table (unit count, ignored, kinds, timing).

This is the inspection tool for judging grouping quality without scoring — it
does not tune anything. Requires the service running (the user runs it on dc1).

    python scripts/inspect_grouping.py
    python scripts/inspect_grouping.py --base-url http://127.0.0.1:8030 --target nl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from urllib import error, request
import uuid


_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
_MIME_BY_SUFFIX = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}
_TERMINAL = {"completed", "failed", "cancelled"}
# OCR needs a source language; grouping itself is language-agnostic. Rough per-image hints.
_SOURCE_BY_STEM = {
    "afstand-houden": "nl",
    "menukaart": "nl",
    "kassabon": "nl",
    "bol-philips": "nl",
    "la-bonne-vache": "fr",
    "nike-ad": "en",
    "danger-1": "en",
    "reynisfjara": "en",
}


def main() -> int:
    args = _parse_args()
    testset = Path(args.testset)
    out_dir = testset / "_inspection"
    out_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(
        path for path in testset.iterdir()
        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
    )
    if not images:
        print(f"no images found in {testset}/")
        return 1

    rows: list[dict[str, object]] = []
    for image_path in images:
        stem = image_path.stem
        source = _SOURCE_BY_STEM.get(stem, args.source)
        print(f"-> {image_path.name} (source={source})", flush=True)
        row = _run_one(
            base_url=args.base_url.rstrip("/"),
            image_path=image_path,
            source=source,
            target=args.target,
            out_dir=out_dir,
        )
        rows.append(row)

    _print_summary(rows, out_dir=out_dir)
    return 0


def _run_one(*, base_url: str, image_path: Path, source: str, target: str, out_dir: Path) -> dict[str, object]:
    stem = image_path.stem
    request_json = json.dumps(
        {
            "task": "translate_image",
            "source_lang_code": source,
            "target_lang_code": target,
            "ocr_route": "scene",
            "request_id": f"inspect_{stem}_{uuid.uuid4().hex[:8]}",
        }
    )
    try:
        submit = _post_multipart(base_url, image_path, request_json)
    except Exception as exc:  # noqa: BLE001 - inspection tool, report and continue
        return {"image": stem, "state": "submit_error", "error": str(exc)}

    request_id = str(submit.get("request_id") or "")
    state = str(submit.get("state") or "")
    result = submit
    deadline = time.time() + 180.0
    while state not in _TERMINAL and time.time() < deadline:
        time.sleep(0.8)
        try:
            result = _get_json(f"{base_url}/v1/requests/{request_id}")
        except Exception as exc:  # noqa: BLE001
            return {"image": stem, "state": "poll_error", "error": str(exc)}
        state = str(result.get("state") or "")

    if state != "completed":
        return {
            "image": stem,
            "state": state or "timeout",
            "error": json.dumps(result.get("error") or {}, ensure_ascii=False)[:300],
        }

    response = result.get("response") or {}
    ocr = response.get("ocr") or {}
    units = ocr.get("translation_units") or []
    metrics = response.get("metrics") or {}

    (out_dir / f"{stem}.units.json").write_text(
        json.dumps({"units": units, "ignored_cell_ids": ocr.get("ignored_cell_ids") or []}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    overlay_saved = _save_artifact(base_url, request_id, "grouping_overlay_debug", out_dir / f"{stem}.grouping.png")

    kinds = [str(unit.get("kind") or "?") for unit in units]
    return {
        "image": stem,
        "state": "completed",
        "cells": len(ocr.get("cells") or []),
        "units": len(units),
        "flow": kinds.count("flow"),
        "field": kinds.count("field"),
        "ignored": len(ocr.get("ignored_cell_ids") or []),
        "grouping_ms": round(float(metrics.get("grouping_wall_ms") or 0.0)),
        "overlay": "ok" if overlay_saved else "missing",
    }


def _post_multipart(base_url: str, image_path: Path, request_json: str) -> dict:
    boundary = f"inspect-{uuid.uuid4().hex}"
    mime = _MIME_BY_SUFFIX.get(image_path.suffix.lower(), "application/octet-stream")
    body = b"".join(
        [
            f"--{boundary}\r\n".encode(),
            b'Content-Disposition: form-data; name="request_json"\r\n',
            b"Content-Type: application/json; charset=utf-8\r\n\r\n",
            request_json.encode("utf-8"),
            b"\r\n",
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="image_file"; filename="{image_path.name}"\r\n'.encode(),
            f"Content-Type: {mime}\r\n\r\n".encode(),
            image_path.read_bytes(),
            b"\r\n",
            f"--{boundary}--\r\n".encode(),
        ]
    )
    req = request.Request(
        url=f"{base_url}/v1/requests",
        method="POST",
        headers={"Accept": "application/json", "Content-Type": f"multipart/form-data; boundary={boundary}"},
        data=body,
    )
    with request.urlopen(req, timeout=120.0) as response:
        return json.loads(response.read().decode("utf-8"))


def _get_json(url: str) -> dict:
    req = request.Request(url=url, method="GET", headers={"Accept": "application/json"})
    with request.urlopen(req, timeout=10.0) as response:
        return json.loads(response.read().decode("utf-8"))


def _save_artifact(base_url: str, request_id: str, name: str, dest: Path) -> bool:
    url = f"{base_url}/v1/requests/{request_id}/artifacts/{name}"
    try:
        with request.urlopen(request.Request(url=url, method="GET"), timeout=15.0) as response:
            dest.write_bytes(response.read())
        return True
    except error.HTTPError:
        return False


def _print_summary(rows: list[dict[str, object]], *, out_dir: Path) -> None:
    print(f"\n=== grouping inspection ({out_dir}) ===")
    header = f"{'image':<18} {'state':<10} {'cells':>5} {'units':>5} {'flow':>4} {'field':>5} {'ignored':>7} {'ms':>6}  overlay"
    print(header)
    print("-" * len(header))
    for row in rows:
        if row.get("state") == "completed":
            print(
                f"{str(row['image']):<18} {str(row['state']):<10} {row['cells']:>5} {row['units']:>5} "
                f"{row['flow']:>4} {row['field']:>5} {row['ignored']:>7} {row['grouping_ms']:>6}  {row['overlay']}"
            )
        else:
            print(f"{str(row['image']):<18} {str(row['state']):<10}  {str(row.get('error') or '')[:80]}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run grouping over the testset via the live service.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8030")
    parser.add_argument("--testset", default="testset")
    parser.add_argument("--source", default="en", help="fallback OCR source language for unknown images")
    parser.add_argument("--target", default="nl")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
