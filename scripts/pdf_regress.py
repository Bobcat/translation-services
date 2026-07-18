#!/usr/bin/env python3
"""Document regression CLI for ``translate_pdf`` (docs/pdf-benchmark-regression-design.md, slice 2b).

    # freeze a document fixture from a completed run (id or job dir under work_root)
    python scripts/pdf_regress.py capture --request req_...abc [--name 01_...] [--no-score]

    # replay every document fixture (or one) and diff against its snapshots
    python scripts/pdf_regress.py run [--name 01_...] [--score]

    # re-baseline a deliberately changed document (refuses on frozen-input diffs)
    python scripts/pdf_regress.py accept --name 01_... [--lang nl] [--variant v1] [--no-score]

    python scripts/pdf_regress.py list

``--score`` / the accepted-score freeze run the GPU measurement stack (layout + OCR over both
renders); plain replays stay on the deterministic chain only.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.core.config import load_settings
from app.regression.pdf import fixture as dfx
from app.regression.pdf.capture import CaptureError
from app.regression.pdf.capture import capture_document
from app.regression.pdf.capture import testset_name_for
from app.regression.pdf.run import accept_document
from app.regression.pdf.run import run_document


def _axes_line(scores: dict) -> str:
    axes = scores.get("axes") or {}
    indicators = scores.get("indicators") or {}
    return (
        f"layout {axes.get('layout', 0):6.2f}  retain {axes.get('retention', 0):6.2f}  "
        f"typo {axes.get('typography', 0):6.2f}  unchanged {indicators.get('unchanged_share', 0):5.1f}%"
    )


def _resolve_job_root(work_root: Path, request: str) -> Path:
    as_path = Path(request)
    if as_path.is_dir():
        return as_path.resolve()
    return (work_root / request).resolve()


def _find_source_pdf(work_root: Path, job_root: Path) -> Path | None:
    uploads = work_root / "_uploads" / job_root.name
    pdfs = sorted(uploads.glob("*.pdf")) if uploads.is_dir() else []
    return pdfs[0] if len(pdfs) == 1 else None


def cmd_capture(args: argparse.Namespace) -> int:
    settings = load_settings()
    work_root = Path(settings.service.work_root)
    job_root = _resolve_job_root(work_root, args.request)
    if not (job_root / "document.json").exists():
        print(f"no completed translate_pdf run at {job_root}", file=sys.stderr)
        return 1
    source_pdf = Path(args.source).resolve() if args.source else _find_source_pdf(work_root, job_root)
    if source_pdf is None or not source_pdf.exists():
        print("source pdf not found under _uploads; pass --source", file=sys.stderr)
        return 1
    name = args.name or testset_name_for(source_pdf)
    if not name:
        print("source does not match a testset/pdf document; pass --name", file=sys.stderr)
        return 1
    try:
        out = capture_document(
            settings,
            job_root=job_root,
            source_pdf=source_pdf,
            name=name,
            variant=args.variant,
            freeze_score=not args.no_score,
        )
    except CaptureError as exc:
        print(f"capture refused: {exc}", file=sys.stderr)
        return 1
    print(f"captured {out['name']}/{out['target_lang']}/{out['variant']}: "
          f"{out['pages']} page(s), {out['units']} unit(s)")
    if out.get("accepted_scores"):
        print(f"accepted score: {_axes_line(out['accepted_scores'])}")
    print(f"stored: {out['path']}")
    return 0


def _select_variants(args: argparse.Namespace) -> list[tuple[str, str, str, Path]]:
    return [
        (name, lang, variant, path)
        for name, lang, variant, path in dfx.variant_dirs()
        if (not args.name or name == args.name)
        and (not getattr(args, "lang", None) or lang == args.lang)
        and (not getattr(args, "variant", None) or variant == args.variant)
    ]


def cmd_run(args: argparse.Namespace) -> int:
    settings = load_settings()
    variants = _select_variants(args)
    if not variants:
        print("no document fixtures matched")
        return 0
    failures = 0
    for name, lang, variant, path in variants:
        try:
            result = run_document(settings, variant_path=path, score=args.score)
        except Exception as exc:  # noqa: BLE001 - report and continue
            result = {"passed": False, "frozen_input_diffs": [], "diffs": [f"replay error: {exc}"]}
        label = f"{name}/{lang}/{variant}"
        if result["passed"]:
            print(f"PASS {label}")
        else:
            failures += 1
            print(f"FAIL {label}")
            for diff in result.get("frozen_input_diffs") or []:
                print(f"     ! {diff}   [frozen input — re-capture, not accept]")
            for diff in result.get("diffs") or []:
                print(f"     - {diff}")
        score = result.get("score")
        if score and score.get("replay"):
            print(f"     score: {_axes_line(score['replay'])}")
    print(f"\n{len(variants) - failures}/{len(variants)} passed")
    return 1 if failures else 0


def cmd_accept(args: argparse.Namespace) -> int:
    settings = load_settings()
    variants = _select_variants(args)
    if len(variants) != 1:
        matched = ", ".join(f"{n}/{l}/{v}" for n, l, v, _ in variants) or "none"
        print(f"accept needs exactly one fixture; matched: {matched}", file=sys.stderr)
        return 1
    name, lang, variant, path = variants[0]
    result = accept_document(settings, variant_path=path, freeze_score=not args.no_score)
    if not result.get("ok"):
        print(f"accept refused: {result.get('error')}", file=sys.stderr)
        for diff in (result.get("frozen_input_diffs") or []) + (result.get("diffs") or []):
            print(f"  - {diff}", file=sys.stderr)
        return 1
    print(f"accepted {name}/{lang}/{variant} ({result['pages']} page(s))")
    if result.get("accepted_scores"):
        print(f"accepted score: {_axes_line(result['accepted_scores'])}")
    return 0


def cmd_list(_args: argparse.Namespace) -> int:
    documents = dfx.list_documents()
    if not documents:
        print("no document fixtures")
        return 0
    for doc in documents:
        scored = "scored" if doc["has_accepted_scores"] else "no accepted score"
        print(f"{doc['name']:52s} {doc['target_lang']:5s} {doc['variant']:4s} "
              f"{doc['pages']:2d} page(s) @{doc['analysis_dpi']}dpi  [{scored}]")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("capture", help="freeze a document fixture from a completed run")
    p.add_argument("--request", required=True, help="request id under work_root, or a job dir path")
    p.add_argument("--source", help="source pdf (default: the run's _uploads copy)")
    p.add_argument("--name", help="fixture name (default: matching testset/pdf stem)")
    p.add_argument("--variant", help="variant dir (default: next free vN)")
    p.add_argument("--no-score", action="store_true", help="skip the accepted-score freeze")
    p.set_defaults(func=cmd_capture)

    p = sub.add_parser("run", help="replay document fixtures and diff against snapshots")
    p.add_argument("--name")
    p.add_argument("--lang")
    p.add_argument("--variant")
    p.add_argument("--score", action="store_true", help="benchmark-on-replay vs the accepted score")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("accept", help="re-baseline one document fixture")
    p.add_argument("--name", required=True)
    p.add_argument("--lang")
    p.add_argument("--variant")
    p.add_argument("--no-score", action="store_true", help="skip re-freezing the accepted score")
    p.set_defaults(func=cmd_accept)

    p = sub.add_parser("list", help="print the document-fixture inventory")
    p.set_defaults(func=cmd_list)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
