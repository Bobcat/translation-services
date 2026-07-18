"""Replay a document fixture and diff it: frozen-input checks, per-page image replay, assembly,
optional benchmark-on-replay against the frozen accepted score.

Two diff classes with different re-baseline semantics:

- **Frozen-input diffs** (census, page raster sha, text-layer extraction): a deterministic
  derivation from ``source.pdf`` no longer reproduces what the fixture froze. The frozen hint and
  translations belong to the OLD derivation, so these cannot be accepted in place — ``accept``
  refuses and asks for a fresh live capture.
- **Replay diffs** (align structure, render re-OCR, assembled page geometry): the deterministic
  chain changed. Exactly the image-harness semantics per page; ``accept`` re-baselines them.

Benchmark-on-replay scores the assembled replay PDF against ``source.pdf`` and compares against
the accepted side — both scored with the CURRENT scoring code (the accepted side re-derived from
its frozen measurement), so scoring-code evolution never skews the comparison. On a green replay
the assembled bytes equal the accepted bytes by construction, so any score delta is real.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from app.core.config import AppSettings
from app.pdf.assemble import PageImage
from app.pdf.assemble import assemble_pdf
from app.pdf.document import profile_pdf
from app.pdf.raster import PageRasterizer
from app.pdf.textlayer import PageTextExtractor
from app.regression import fixture as fx
from app.regression.compare import diff_units
from app.regression.compare import reocr_mismatches
from app.regression.pdf import checks
from app.regression.pdf import fixture as dfx
from app.regression.replay import replay_fixture
from app.regression.run import write_snapshot_diff
from app.regression.snapshot import reocr_rows


@dataclass
class _PageReplay:
    page: int
    path: Path
    frozen_diffs: list[str] = field(default_factory=list)
    diffs: list[str] = field(default_factory=list)
    boxes: list[dict[str, Any]] = field(default_factory=list)
    rendered_png: bytes | None = None
    actual_units: list[dict[str, Any]] = field(default_factory=list)
    actual_ignored: list[int] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)


@dataclass
class _DocumentReplay:
    fixture: dfx.DocumentFixture
    frozen_diffs: list[str]  # census-level, before any page ran
    pages: list[_PageReplay]
    assembled_pdf: bytes | None
    assembled_diffs: list[str]

    @property
    def all_frozen_diffs(self) -> list[str]:
        return self.frozen_diffs + [d for page in self.pages for d in page.frozen_diffs]

    @property
    def all_replay_diffs(self) -> list[str]:
        return [d for page in self.pages for d in page.diffs] + self.assembled_diffs


def run_document(
    settings: AppSettings, *, variant_path: Path, score: bool = False
) -> dict[str, Any]:
    """``{passed, frozen_input_diffs, diffs, pages, ...}`` for one document fixture. Writes the
    reviewer artifacts (per-page ``actual.png``/``snapshot_diff.png``, document ``actual.pdf``)
    on failure and clears them on a pass."""
    replay = _replay_document(settings, variant_path)
    score_result: dict[str, Any] | None = None
    score_diffs: list[str] = []
    if score and replay.assembled_pdf is not None:
        score_result, score_diffs = _score_replay(settings, variant_path, replay)

    _manage_document_artifacts(variant_path, replay, score_diffs)
    frozen = replay.all_frozen_diffs
    diffs = replay.all_replay_diffs + score_diffs
    return {
        "passed": not frozen and not diffs,
        "frozen_input_diffs": frozen,
        "diffs": diffs,
        "pages": [
            {
                "page": page.page,
                "passed": not page.frozen_diffs and not page.diffs,
                "diffs": page.frozen_diffs + page.diffs,
                "timings": {key: round(value, 1) for key, value in page.timings.items()},
            }
            for page in replay.pages
        ],
        **({"score": score_result} if score_result is not None else {}),
    }


def accept_document(
    settings: AppSettings, *, variant_path: Path, freeze_score: bool = True
) -> dict[str, Any]:
    """Re-baseline a document: replay it and overwrite the per-page snapshots, ``accepted.pdf``
    and (unless skipped) the accepted score with the current output. Refuses when frozen inputs
    or the assembled geometry diverge — that is not a re-baselinable change but a broken
    derivation; re-capture from a fresh live run instead."""
    replay = _replay_document(settings, variant_path)
    frozen = replay.all_frozen_diffs
    if frozen:
        return {
            "ok": False,
            "needs_recapture": True,
            "error": "frozen inputs no longer reproduce from source.pdf; re-capture from a fresh run",
            "frozen_input_diffs": frozen,
        }
    if replay.assembled_diffs or replay.assembled_pdf is None:
        return {
            "ok": False,
            "needs_recapture": False,
            "error": "assembled output is inconsistent with the census; fix assemble before accepting",
            "frozen_input_diffs": [],
            "diffs": replay.assembled_diffs,
        }

    for page in replay.pages:
        assert page.rendered_png is not None  # frozen diffs were empty, so every page replayed
        snapshot = fx.Snapshot(
            expected_units=page.actual_units,
            ignored_cells=page.actual_ignored,
            reocr=reocr_rows(settings.ocr, page.rendered_png, replay.fixture.target_lang),
        )
        fx.save_snapshot(page.path, snapshot)
        (page.path / "snapshot.png").write_bytes(page.rendered_png)
        _clear(page.path / "actual.png", page.path / "snapshot_diff.png")
    dfx.accepted_pdf_path(variant_path).write_bytes(replay.assembled_pdf)
    _clear(variant_path / "actual.pdf")

    out: dict[str, Any] = {"ok": True, "pages": len(replay.pages)}
    if freeze_score:
        from app.regression.pdf.capture import freeze_accepted_score

        out["accepted_scores"] = freeze_accepted_score(
            settings, variant_path=variant_path, target_lang=replay.fixture.target_lang
        )
    return out


def _replay_document(settings: AppSettings, variant_path: Path) -> _DocumentReplay:
    fixture = dfx.load_document(variant_path)
    source_pdf = dfx.source_pdf_path(variant_path)
    if not source_pdf.exists():
        return _DocumentReplay(fixture, ["fixture has no source.pdf"], [], None, [])
    if fx.sha256(source_pdf.read_bytes()) != fixture.source_sha256:
        return _DocumentReplay(fixture, ["source.pdf sha mismatch (corrupted fixture)"], [], None, [])

    profile = profile_pdf(source_pdf, page_cap=fixture.page_count)
    census_diffs = checks.census_diffs(fixture.census, [page.to_dict() for page in profile.pages])
    if any(diff.startswith("page count") for diff in census_diffs):
        # The page fixtures no longer line up with the document; nothing per-page is meaningful.
        return _DocumentReplay(fixture, census_diffs, [], None, [])

    pages: list[_PageReplay] = []
    with (
        PageRasterizer(source_pdf, dpi=fixture.analysis_dpi) as rasterizer,
        PageTextExtractor(source_pdf, dpi=fixture.analysis_dpi) as extractor,
        TemporaryDirectory(prefix="pdf-replay-") as tmp,
    ):
        for entry in fixture.census:
            page_no = int(entry.get("page") or 0)
            page_path = dfx.page_dir(variant_path, page_no)
            page = _PageReplay(page=page_no, path=page_path)
            pages.append(page)
            if not (page_path / "fixture.json").exists():
                page.frozen_diffs.append(f"page {page_no}: page fixture missing")
                continue
            page_fixture, snapshot = fx.load(page_path)

            raster_bytes = rasterizer.render_png(page_no - 1)
            if fx.sha256(raster_bytes) != page_fixture.image_sha256:
                page.frozen_diffs.append(
                    f"page {page_no}: raster changed — source.pdf at {fixture.analysis_dpi} dpi "
                    "no longer reproduces the frozen page raster"
                )
            if str(entry.get("cell_source")) == "pdf_text_layer":
                extracted = extractor.cells_for_page(page_no - 1)
                page.frozen_diffs.extend(
                    f"page {page_no}: {diff}"
                    for diff in checks.extraction_diffs(page_fixture.cells, extracted.cells)
                )

            raster_path = Path(tmp) / f"page-{page_no:03d}.png"
            raster_path.write_bytes(raster_bytes)
            actual_units, actual_ignored, rendered_png, timings = replay_fixture(
                raster_path, page_fixture
            )
            reocr_started = time.perf_counter()
            actual_rows = reocr_rows(settings.ocr, rendered_png, page_fixture.target_lang)
            timings = {**timings, "reocr_ms": (time.perf_counter() - reocr_started) * 1000.0}
            reocr_diffs, boxes = reocr_mismatches(snapshot.reocr, actual_rows)
            page.diffs = [
                f"page {page_no}: {diff}"
                for diff in (
                    diff_units(
                        snapshot.expected_units, actual_units,
                        snapshot.ignored_cells, actual_ignored,
                    )
                    + reocr_diffs
                )
            ]
            page.boxes = boxes
            page.rendered_png = rendered_png
            page.actual_units = actual_units
            page.actual_ignored = actual_ignored
            page.timings = timings

        assembled: bytes | None = None
        assembled_diffs: list[str] = []
        if all(page.rendered_png is not None for page in pages):
            page_images = []
            for entry, page in zip(fixture.census, pages):
                rendered_path = Path(tmp) / f"rendered-{page.page:03d}.png"
                rendered_path.write_bytes(page.rendered_png or b"")
                page_images.append(
                    PageImage(
                        png_path=rendered_path,
                        width_pt=float(entry.get("width_pt") or 0),
                        height_pt=float(entry.get("height_pt") or 0),
                    )
                )
            assembled = assemble_pdf(page_images)
            assembled_diffs = checks.assembled_pdf_diffs(assembled, fixture.census)

    return _DocumentReplay(fixture, census_diffs, pages, assembled, assembled_diffs)


def _score_replay(
    settings: AppSettings, variant_path: Path, replay: _DocumentReplay
) -> tuple[dict[str, Any], list[str]]:
    """Benchmark the assembled replay PDF and diff it against the accepted side, both scored with
    the current scoring code. Lazy imports: only --score loads the measurement stack."""
    from app.benchmark.measurement import measure_pair
    from app.benchmark.scoring import score_measurement
    from app.regression.snapshot import ocr_language_for_target

    assert replay.assembled_pdf is not None
    with TemporaryDirectory(prefix="pdf-replay-score-") as tmp:
        replay_pdf = Path(tmp) / "replay.pdf"
        replay_pdf.write_bytes(replay.assembled_pdf)
        measurement = measure_pair(
            settings=settings,
            source_pdf=dfx.source_pdf_path(variant_path),
            translated_pdf=replay_pdf,
            ocr_language=ocr_language_for_target(replay.fixture.target_lang),
        )
    scores = score_measurement(measurement)

    accepted_path = dfx.accepted_measurement_path(variant_path)
    if not accepted_path.exists():
        return (
            {"replay": scores, "accepted": None},
            ["no accepted measurement frozen (capture/accept ran with --no-score)"],
        )
    accepted_scores = score_measurement(json.loads(accepted_path.read_text(encoding="utf-8")))
    diffs = [
        f"score.{group}.{key}: {got!r} != accepted {want!r}"
        for group in ("axes", "indicators", "flags")
        for key in sorted(set(accepted_scores.get(group) or {}) | set(scores.get(group) or {}))
        if (got := (scores.get(group) or {}).get(key)) != (want := (accepted_scores.get(group) or {}).get(key))
    ]
    return {"replay": scores, "accepted": accepted_scores}, diffs


def _manage_document_artifacts(
    variant_path: Path, replay: _DocumentReplay, score_diffs: list[str]
) -> None:
    """Per-page ``actual.png``/``snapshot_diff.png`` and document ``actual.pdf``: written where a
    reviewer needs to look, removed once the fixture passes clean again."""
    for page in replay.pages:
        failed = bool(page.frozen_diffs or page.diffs)
        if failed and page.rendered_png is not None:
            (page.path / "actual.png").write_bytes(page.rendered_png)
            write_snapshot_diff(page.path, page.boxes)
        elif not failed:
            _clear(page.path / "actual.png", page.path / "snapshot_diff.png")
    document_failed = bool(
        replay.all_frozen_diffs or replay.all_replay_diffs or score_diffs
    )
    if document_failed and replay.assembled_pdf is not None:
        (variant_path / "actual.pdf").write_bytes(replay.assembled_pdf)
    elif not document_failed:
        _clear(variant_path / "actual.pdf")


def _clear(*paths: Path) -> None:
    for path in paths:
        if path.exists():
            path.unlink()
