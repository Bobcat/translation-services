"""Document-level replay checks: pure diff functions over frozen vs re-derived inputs/outputs.

These guard the parts of ``translate_pdf`` that sit OUTSIDE the per-page align/render chain but
inside our own code: the census (``app.pdf.document``), the raster (``app.pdf.raster`` + engine),
the text-layer extraction (``app.pdf.textlayer``) and the assembly (``app.pdf.assemble``). Each
returns human-readable mismatch strings (empty list = pass), like ``app.regression.compare``.

A non-empty census / raster / extraction diff means a FROZEN INPUT no longer reproduces from
``source.pdf`` — the frozen hint and translations belong to the old derivation, so such a diff is
not re-baselinable in place: it requires a fresh live capture. The accept flow enforces that.
"""
from __future__ import annotations

import json
from typing import Any

import pymupdf

from app.regression.pdf.fixture import CENSUS_PROFILE_KEYS


def census_diffs(census: list[dict[str, Any]], profile_pages: list[dict[str, Any]]) -> list[str]:
    """Frozen census vs a fresh ``profile_pdf`` run on the frozen source. Exact on every profiled
    field: a changed page class re-routes cell sourcing, a changed size re-scales everything."""
    diffs: list[str] = []
    if len(census) != len(profile_pages):
        diffs.append(f"page count {len(profile_pages)} != expected {len(census)}")
        return diffs
    for entry, page in zip(census, profile_pages):
        page_no = int(entry.get("page") or 0)
        for key in CENSUS_PROFILE_KEYS:
            if entry.get(key) != page.get(key):
                diffs.append(
                    f"page {page_no}: census.{key}: {page.get(key)!r} != expected {entry.get(key)!r}"
                )
    return diffs


def extraction_diffs(frozen_cells: list[dict[str, Any]], extracted_cells: list[dict[str, Any]]) -> list[str]:
    """Frozen text-layer cells vs a fresh extraction from the frozen source at the frozen dpi.
    Exact JSON equality per cell: the cells are align/render INPUTS, so any drift (text, geometry,
    style) changes the chain downstream. Reported compactly — count first, then the first few
    differing cells with the fields that differ."""
    diffs: list[str] = []
    if len(frozen_cells) != len(extracted_cells):
        diffs.append(
            f"text-layer extraction: {len(extracted_cells)} cells != expected {len(frozen_cells)}"
        )
    reported = 0
    for index, (frozen, extracted) in enumerate(zip(frozen_cells, extracted_cells)):
        if frozen == extracted:
            continue
        changed = sorted(
            key
            for key in set(frozen) | set(extracted)
            if _canonical(frozen.get(key)) != _canonical(extracted.get(key))
        )
        diffs.append(f"text-layer cell[{index}] differs on: {', '.join(changed) or 'value'}")
        reported += 1
        if reported >= 3:
            remaining = sum(
                1 for f, e in zip(frozen_cells[index + 1:], extracted_cells[index + 1:]) if f != e
            )
            if remaining:
                diffs.append(f"text-layer extraction: {remaining} more differing cell(s)")
            break
    return diffs


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def assembled_pdf_diffs(pdf_bytes: bytes, census: list[dict[str, Any]]) -> list[str]:
    """The assembled output PDF must carry one page per source page on the source's dimensions
    (pt) — the ``assemble_pdf`` contract."""
    diffs: list[str] = []
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    try:
        if doc.page_count != len(census):
            diffs.append(f"assembled pdf: page count {doc.page_count} != expected {len(census)}")
            return diffs
        for entry in census:
            page_no = int(entry.get("page") or 0)
            rect = doc[page_no - 1].rect
            got = (round(float(rect.width), 2), round(float(rect.height), 2))
            want = (float(entry.get("width_pt") or 0), float(entry.get("height_pt") or 0))
            if got != want:
                diffs.append(f"assembled pdf: page {page_no} size {got} != expected {want}")
    finally:
        doc.close()
    return diffs
