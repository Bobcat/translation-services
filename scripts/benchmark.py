#!/usr/bin/env python3
"""Document-pair benchmark CLI (docs/pdf-benchmark-regression-design.md, slice 2a).

    # score one pair and store the run
    python scripts/benchmark.py measure --source a.pdf --translated a_nl.pdf \
        --doc-id pdf-01 --system ours

    # identity baseline: every testset pdf scored against itself
    python scripts/benchmark.py identity --testset testset/pdf

    # recompute scores.json for every stored run with the current scoring code
    python scripts/benchmark.py rescore

    # re-run the measurement layer over every stored pdf pair (GPU; only after a
    # measurement-layer change such as a model upgrade or detection-threshold change)
    python scripts/benchmark.py remeasure

    # print the leaderboard table from stored runs
    python scripts/benchmark.py report
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.benchmark.measurement import measure_pair
from app.benchmark.scoring import SCORING_VERSION
from app.benchmark.scoring import score_measurement
from app.benchmark.store import default_data_root
from app.benchmark.store import list_runs
from app.benchmark.store import save_run
from app.benchmark.store import write_scores
from app.core.config import load_settings


def _axes_line(scores: dict) -> str:
    axes = scores.get("axes") or {}
    flags = scores.get("flags") or {}
    flag_marks = "".join(
        "" if flags.get(key, True) else f" !{key.removesuffix('_equal')}"
        for key in ("page_count_equal", "image_regions_equal", "table_regions_equal")
    )
    indicators = scores.get("indicators") or {}
    ratio = indicators.get("volume_ratio")
    return (
        f"layout {axes.get('layout', 0):6.2f}  "
        f"anchors {axes.get('anchors', 0):6.2f}  "
        f"typo {axes.get('typography', 0):6.2f}  "
        f"unchanged {indicators.get('unchanged_share', 0):5.1f}%  "
        f"vol {'x%.2f' % ratio if ratio is not None else '  n/a'}{flag_marks}"
    )


def cmd_measure(args: argparse.Namespace) -> int:
    settings = load_settings()
    source = Path(args.source).resolve()
    translated = Path(args.translated).resolve()
    measurement = measure_pair(
        settings=settings, source_pdf=source, translated_pdf=translated,
        ocr_language=args.ocr_language,
    )
    scores = score_measurement(measurement)
    run = save_run(
        data_root=Path(args.data_root), doc_id=args.doc_id, system=args.system,
        source_pdf=source, translated_pdf=translated,
        measurement=measurement, scores=scores,
    )
    print(f"{args.doc_id:28s} {args.system:12s} {_axes_line(scores)}")
    print(f"stored: {run.path}")
    return 0


def cmd_identity(args: argparse.Namespace) -> int:
    settings = load_settings()
    pdfs = sorted(Path(args.testset).glob("*.pdf"))
    if not pdfs:
        print(f"no PDFs under {args.testset}", file=sys.stderr)
        return 1
    for pdf in pdfs:
        measurement = measure_pair(
            settings=settings, source_pdf=pdf, translated_pdf=pdf,
            ocr_language=args.ocr_language,
        )
        scores = score_measurement(measurement)
        save_run(
            data_root=Path(args.data_root), doc_id=pdf.stem, system="identity",
            source_pdf=pdf, translated_pdf=pdf,
            measurement=measurement, scores=scores,
        )
        print(f"{pdf.stem:52s} {_axes_line(scores)}")
    return 0


def cmd_remeasure(args: argparse.Namespace) -> int:
    """Re-run layer (a) — render + layout + OCR — over every stored pdf pair, in place.
    The pairs are the ground truth; measurement.json and scores.json are derived and
    overwritten. Run ids stay, so the comparison history keeps its identity."""
    settings = load_settings()
    for run in list_runs(Path(args.data_root)):
        measurement = measure_pair(
            settings=settings,
            source_pdf=run.path / "source.pdf",
            translated_pdf=run.path / "translated.pdf",
            ocr_language=args.ocr_language,
        )
        run.measurement_path.write_text(
            json.dumps(measurement, ensure_ascii=False), encoding="utf-8"
        )
        scores = score_measurement(measurement)
        write_scores(run, scores)
        print(f"{run.doc_id:36s} {run.system:12s} {run.run_id}  {_axes_line(scores)}")
    return 0


def cmd_rescore(args: argparse.Namespace) -> int:
    changed = 0
    for run in list_runs(Path(args.data_root)):
        before = run.load_scores()
        after = score_measurement(run.load_measurement())
        write_scores(run, after)
        if before is None or (before.get("axes") != after.get("axes")):
            changed += 1
            print(f"changed  {run.doc_id}/{run.system}/{run.run_id}  {_axes_line(after)}")
    print(f"re-scored with scoring rev {SCORING_VERSION}; {changed} run(s) changed")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    for run in list_runs(Path(args.data_root)):
        scores = run.load_scores()
        if scores is None:
            continue
        print(f"{run.doc_id:36s} {run.system:12s} {run.run_id}  {_axes_line(scores)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=str(default_data_root()))
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("measure", help="measure + score one pair, store the run")
    p.add_argument("--source", required=True)
    p.add_argument("--translated", required=True)
    p.add_argument("--doc-id", required=True)
    p.add_argument("--system", required=True)
    p.add_argument("--ocr-language", default="en")
    p.set_defaults(func=cmd_measure)

    p = sub.add_parser("identity", help="identity baseline over a directory of PDFs")
    p.add_argument("--testset", required=True)
    p.add_argument("--ocr-language", default="en")
    p.set_defaults(func=cmd_identity)

    p = sub.add_parser("rescore", help="recompute scores.json for all stored runs")
    p.set_defaults(func=cmd_rescore)

    p = sub.add_parser("remeasure", help="re-run the measurement layer over all stored pairs (GPU)")
    p.add_argument("--ocr-language", default="en")
    p.set_defaults(func=cmd_remeasure)

    p = sub.add_parser("report", help="print stored runs")
    p.set_defaults(func=cmd_report)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
