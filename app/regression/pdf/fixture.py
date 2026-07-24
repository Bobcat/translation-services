"""The document fixture: what a ``translate_pdf`` regression freezes at DOCUMENT level.

A document fixture is a directory ``testset/pdf/_regression/<stem>/<lang>/<vN>/`` holding

- ``source.pdf`` — the exact uploaded bytes (ground truth for every derived input),
- ``document.json`` — this module's :class:`DocumentFixture`: analysis dpi, target language and
  the per-page census (size, rotation, class, cell source) the replay re-derives and diffs,
- ``accepted.pdf`` + ``accepted_scores.json`` — the approved assembled output and its frozen
  benchmark score (the yardstick of the re-baseline decision),
- ``pages/page-NNN/`` — one IMAGE fixture per page (``app.regression.pages.fixture`` schema, reused
  verbatim): frozen cells/hint/translations, approved snapshot, approved render.

Page rasters are not stored: they are deterministically reproducible from ``source.pdf`` at the
recorded dpi, and each page fixture's ``image_sha256`` pins the expected raster bytes.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PDF_REGRESSION_ROOT = Path("testset/pdf/_regression")

# The census keys a replay re-derives from source.pdf (via profile_pdf) and diffs exactly.
# ``cell_source`` is run-time routing (page class + use_pdf_text_layer), frozen as data: it decides
# whether the replay re-runs the text-layer extraction check for that page.
CENSUS_PROFILE_KEYS = (
    "width_pt",
    "height_pt",
    "rotation",
    "text_chars",
    "image_coverage",
    "page_class",
)


@dataclass(frozen=True)
class DocumentFixture:
    source_sha256: str
    analysis_dpi: int
    target_lang: str
    # Per page, 1-based order: {"page": N, <CENSUS_PROFILE_KEYS...>, "cell_source": "ocr"|"pdf_text_layer"}
    census: list[dict[str, Any]]
    schema_version: int = 1

    @property
    def page_count(self) -> int:
        return len(self.census)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "pdf-document-fixture",
            "schema_version": self.schema_version,
            "source_sha256": self.source_sha256,
            "analysis_dpi": self.analysis_dpi,
            "target_lang": self.target_lang,
            "census": self.census,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DocumentFixture":
        return cls(
            source_sha256=str(data["source_sha256"]),
            analysis_dpi=int(data["analysis_dpi"]),
            target_lang=str(data.get("target_lang") or ""),
            census=list(data.get("census") or []),
            schema_version=int(data.get("schema_version") or 1),
        )


def document_path(variant_path: Path) -> Path:
    return variant_path / "document.json"


def source_pdf_path(variant_path: Path) -> Path:
    return variant_path / "source.pdf"


def accepted_pdf_path(variant_path: Path) -> Path:
    return variant_path / "accepted.pdf"


def accepted_scores_path(variant_path: Path) -> Path:
    return variant_path / "accepted_scores.json"


def accepted_measurement_path(variant_path: Path) -> Path:
    # Frozen alongside the accepted score so a scoring-code change can re-derive the accepted
    # side purely (scores = f_scoring(measurement)) — the replay/accepted comparison then always
    # runs both sides through the SAME scoring version, no matter how old the freeze is.
    return variant_path / "accepted_measurement.json"


def page_dir(variant_path: Path, page_number: int) -> Path:
    return variant_path / "pages" / f"page-{int(page_number):03d}"


def page_dirs(variant_path: Path) -> list[Path]:
    root = variant_path / "pages"
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir() and (p / "fixture.json").exists())


def load_document(variant_path: Path) -> DocumentFixture:
    return DocumentFixture.from_dict(json.loads(document_path(variant_path).read_text(encoding="utf-8")))


def save_document(variant_path: Path, fixture: DocumentFixture) -> None:
    variant_path.mkdir(parents=True, exist_ok=True)
    document_path(variant_path).write_text(
        json.dumps(fixture.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8"
    )


def load_accepted_scores(variant_path: Path) -> dict[str, Any] | None:
    path = accepted_scores_path(variant_path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_accepted_scores(variant_path: Path, scores: dict[str, Any]) -> None:
    accepted_scores_path(variant_path).write_text(
        json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def variant_dirs(root: Path = PDF_REGRESSION_ROOT) -> list[tuple[str, str, str, Path]]:
    """``(name, lang, variant, path)`` for every document fixture under ``root``. The last three
    path components are always ``<stem>/<lang>/<vN>``; a fixture may sit under a subset subdir that
    mirrors its source's testset location (``docpack/07_…/nl/v1``), so the walk recurses."""
    out: list[tuple[str, str, str, Path]] = []
    if not root.is_dir():
        return out
    for doc_file in sorted(root.rglob("document.json")):
        variant_path = doc_file.parent
        out.append((variant_path.parent.parent.name, variant_path.parent.name, variant_path.name, variant_path))
    return out


def list_documents(root: Path = PDF_REGRESSION_ROOT) -> list[dict[str, Any]]:
    """Light inventory (no replay): one entry per fixture with page count and the frozen
    accepted score (axes/indicators only — what the Score tab shows without re-running)."""
    out: list[dict[str, Any]] = []
    for name, lang, variant, path in variant_dirs(root):
        try:
            fixture = load_document(path)
        except (OSError, ValueError, KeyError):
            continue
        accepted = load_accepted_scores(path)
        out.append(
            {
                "name": name,
                "target_lang": lang,
                "variant": variant,
                "pages": fixture.page_count,
                "analysis_dpi": fixture.analysis_dpi,
                "has_accepted_scores": accepted is not None,
                "accepted": None if accepted is None else {
                    "scoring_version": accepted.get("scoring_version"),
                    "axes": dict(accepted.get("axes") or {}),
                    "indicators": dict(accepted.get("indicators") or {}),
                    "flags": dict(accepted.get("flags") or {}),
                },
            }
        )
    return out
