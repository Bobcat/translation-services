"""Persistent benchmark-run storage: data/benchmark/<doc-id>/<system>/<run-id>/.

Outside the TTL'd work_root by design: benchmark runs are a lasting dataset.
The pdf pair is the ground truth (a breaking measurement-schema change is a
re-measure pass over stored pairs, never data loss); measurement.json is the
frozen model output; scores.json is derivable and cheap to rewrite.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from app.core.util import safe_token


def default_data_root() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "benchmark"


@dataclass(frozen=True)
class BenchmarkRun:
    path: Path
    doc_id: str
    system: str
    run_id: str

    @property
    def measurement_path(self) -> Path:
        return self.path / "measurement.json"

    @property
    def scores_path(self) -> Path:
        return self.path / "scores.json"

    def load_measurement(self) -> dict[str, Any]:
        return json.loads(self.measurement_path.read_text(encoding="utf-8"))

    def load_scores(self) -> dict[str, Any] | None:
        if not self.scores_path.exists():
            return None
        return json.loads(self.scores_path.read_text(encoding="utf-8"))


def save_run(
    *,
    data_root: Path,
    doc_id: str,
    system: str,
    source_pdf: Path,
    translated_pdf: Path,
    measurement: dict[str, Any],
    scores: dict[str, Any],
) -> BenchmarkRun:
    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + _short_hash(translated_pdf)
    run_dir = (data_root / safe_token(doc_id) / safe_token(system) / run_id).resolve()
    run_dir.relative_to(data_root.resolve())  # path-safety, same guard as the API endpoints
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "source.pdf").write_bytes(source_pdf.read_bytes())
    (run_dir / "translated.pdf").write_bytes(translated_pdf.read_bytes())
    (run_dir / "measurement.json").write_text(
        json.dumps(measurement, ensure_ascii=False), encoding="utf-8"
    )
    run = BenchmarkRun(path=run_dir, doc_id=doc_id, system=system, run_id=run_id)
    write_scores(run, scores)
    return run


def write_scores(run: BenchmarkRun, scores: dict[str, Any]) -> None:
    run.scores_path.write_text(json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8")


def list_runs(data_root: Path) -> list[BenchmarkRun]:
    out: list[BenchmarkRun] = []
    if not data_root.is_dir():
        return out
    for doc_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
        for system_dir in sorted(path for path in doc_dir.iterdir() if path.is_dir()):
            for run_dir in sorted(path for path in system_dir.iterdir() if path.is_dir()):
                if (run_dir / "measurement.json").exists():
                    out.append(
                        BenchmarkRun(
                            path=run_dir,
                            doc_id=doc_dir.name,
                            system=system_dir.name,
                            run_id=run_dir.name,
                        )
                    )
    return out


def find_run(data_root: Path, *, doc_id: str, system: str, run_id: str | None = None) -> BenchmarkRun | None:
    """The named run, or the latest one for (doc, system) when run_id is None.
    Inputs are matched against the safe_token'd directory names the runs were
    stored under, so URL path segments resolve without a second sanitizing pass."""
    candidates = [
        run
        for run in list_runs(data_root)
        if run.doc_id == safe_token(doc_id) and run.system == safe_token(system)
        and (run_id is None or run.run_id == run_id)
    ]
    return candidates[-1] if candidates else None


def runs_index(data_root: Path) -> list[dict[str, Any]]:
    """Flat run list for the comparison matrix: one entry per stored run with its
    scores summary. Grouping/sorting (headroom) is a client concern."""
    out: list[dict[str, Any]] = []
    for run in list_runs(data_root):
        scores = run.load_scores()
        if scores is None:
            continue
        out.append(
            {
                "doc_id": run.doc_id,
                "system": run.system,
                "run_id": run.run_id,
                "scoring_version": scores.get("scoring_version"),
                "axes": dict(scores.get("axes") or {}),
                "indicators": dict(scores.get("indicators") or {}),
                "flags": dict(scores.get("flags") or {}),
            }
        )
    return out


def _short_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:8]
