#!/usr/bin/env python3
"""Capture a fixture + snapshot from a completed request on the live service.

The CLI precursor of POST /v1/regression/fixtures: it freezes exactly the result that just ran
(no re-run). Fetches the request response + rendered artifact over HTTP, then delegates to
``app.regression.capture``.

    python scripts/capture_fixture.py --request-id <id> --name nike-ad --variant v1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.core.config import load_settings
from app.regression import capture as cap


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-id", required=True)
    parser.add_argument("--name", required=True, help="testset stem to store the fixture under")
    parser.add_argument("--variant", default=None, help="default: next free vN for this image+target lang")
    parser.add_argument("--base-url", default="http://127.0.0.1:8030")
    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    result = httpx.get(f"{base}/v1/requests/{args.request_id}", timeout=30).json()
    if (result.get("state") or "") != "completed":
        raise SystemExit(f"request not completed (state={result.get('state')})")
    image_path = cap.testset_image(args.name)
    if image_path is None:
        raise SystemExit(f"testset image '{args.name}' not found under {cap.TESTSET_ROOT}/")
    rendered = httpx.get(f"{base}/v1/requests/{args.request_id}/artifacts/rendered", timeout=30).content

    out = cap.capture(
        load_settings().ocr,
        response=result.get("response") or {},
        rendered_png=rendered,
        image_path=image_path,
        name=args.name,
        variant=args.variant,
    )
    print(f"wrote {out['path']}/  (units={out['units']}, reocr_rows={out['reocr_rows']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
