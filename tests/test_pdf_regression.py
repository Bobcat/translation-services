"""Unit tests for the document regression harness (app.regression.pdf) — the pure pieces.

Documents are generated in-test with pymupdf; no models, no GPU. Covers the document-fixture
model and discovery, the frozen-input checks (census, extraction, assembled geometry), the
translation merge that rebuilds ``translation_units`` from run artifacts, and the capture-side
grouping-model resolution. The full capture/replay loop (align + render + re-OCR) runs live on
the GPU box via ``scripts/pdf_regress.py``, not here.
"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pymupdf
from PIL import Image

from app.pdf.assemble import PageImage
from app.pdf.assemble import assemble_pdf
from app.pdf.document import profile_pdf
from app.pdf.textlayer import PageTextExtractor
from app.regression.capture import build_fixture
from app.regression.pdf import checks
from app.regression.pdf import fixture as dfx
from app.regression.pdf.capture import _grouping_model
from app.regression.pdf.capture import _merge_translations


def _png_bytes(width: int, height: int, color=(250, 250, 250)) -> bytes:
    out = BytesIO()
    Image.new("RGB", (width, height), color).save(out, format="PNG")
    return out.getvalue()


def _text_pdf(tmp_path: Path, name: str = "doc.pdf") -> Path:
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 100), "A born-digital line of text", fontsize=12)
    path = tmp_path / name
    path.write_bytes(doc.tobytes())
    doc.close()
    return path


def _census_for(path: Path, cell_source: str = "pdf_text_layer") -> list[dict]:
    profile = profile_pdf(path, page_cap=25)
    return [
        {"page": page.index + 1, **page.to_dict(), "cell_source": cell_source}
        for page in profile.pages
    ]


# --- census check -------------------------------------------------------------------------


def test_census_diffs_pass_on_reprofile(tmp_path: Path) -> None:
    path = _text_pdf(tmp_path)
    census = _census_for(path)
    profile = profile_pdf(path, page_cap=len(census))
    assert checks.census_diffs(census, [p.to_dict() for p in profile.pages]) == []


def test_census_diffs_flag_changed_class_and_count(tmp_path: Path) -> None:
    path = _text_pdf(tmp_path)
    census = _census_for(path)
    profile = profile_pdf(path, page_cap=25)
    pages = [p.to_dict() for p in profile.pages]

    mutated = [dict(census[0], page_class="scanned")]
    diffs = checks.census_diffs(mutated, pages)
    assert len(diffs) == 1 and "page_class" in diffs[0]

    assert checks.census_diffs(census, pages + pages) == ["page count 2 != expected 1"]


# --- text-layer extraction check ----------------------------------------------------------


def test_extraction_diffs_pass_on_reextract(tmp_path: Path) -> None:
    path = _text_pdf(tmp_path)
    with PageTextExtractor(path, dpi=160) as extractor:
        first = extractor.cells_for_page(0).cells
        second = extractor.cells_for_page(0).cells
    assert checks.extraction_diffs(first, second) == []


def test_extraction_diffs_name_the_changed_fields(tmp_path: Path) -> None:
    path = _text_pdf(tmp_path)
    with PageTextExtractor(path, dpi=160) as extractor:
        cells = extractor.cells_for_page(0).cells
    mutated = [dict(cells[0], text="Different text", size_pt=13.0)]
    diffs = checks.extraction_diffs(mutated, cells)
    assert len(diffs) == 1
    assert "cell[0]" in diffs[0] and "size_pt" in diffs[0] and "text" in diffs[0]


def test_extraction_diffs_report_count_and_cap_detail() -> None:
    frozen = [{"text": f"cell {i}"} for i in range(6)]
    extracted = [{"text": f"CELL {i}"} for i in range(5)]
    diffs = checks.extraction_diffs(frozen, extracted)
    assert diffs[0] == "text-layer extraction: 5 cells != expected 6"
    assert any("more differing cell" in d for d in diffs)  # detail capped at 3 cells


# --- assembled-pdf check ------------------------------------------------------------------


def test_assembled_pdf_diffs_pass_and_flag_size(tmp_path: Path) -> None:
    png = tmp_path / "page.png"
    png.write_bytes(_png_bytes(64, 91))
    census = [
        {"page": 1, "width_pt": 595.0, "height_pt": 842.0},
        {"page": 2, "width_pt": 612.0, "height_pt": 792.0},
    ]
    assembled = assemble_pdf([
        PageImage(png_path=png, width_pt=595.0, height_pt=842.0),
        PageImage(png_path=png, width_pt=612.0, height_pt=792.0),
    ])
    assert checks.assembled_pdf_diffs(assembled, census) == []

    wrong = assemble_pdf([PageImage(png_path=png, width_pt=595.0, height_pt=842.0)])
    assert checks.assembled_pdf_diffs(wrong, census) == ["assembled pdf: page count 1 != expected 2"]

    census_wrong = [dict(census[0]), dict(census[1], height_pt=800.0)]
    diffs = checks.assembled_pdf_diffs(assembled, census_wrong)
    assert len(diffs) == 1 and "page 2 size" in diffs[0]


# --- translation merge (run artifacts -> translation_units) -------------------------------


_GROUPING = {
    "units": [
        {"id": 1, "hint_index": 0, "members": [{"cell_id": 10, "order": 0}]},
        {"id": 2, "hint_index": None, "members": [{"cell_id": 11, "order": 1}, {"cell_id": 12, "order": 0}]},
        {"id": 3, "hint_index": 2, "members": [{"cell_id": 13, "order": 2}]},
    ],
    "cells": [{"id": 10}, {"id": 11}, {"id": 12}, {"id": 13}],
    "layout_regions": [],
}


def test_merge_translations_filters_and_merges() -> None:
    translation = [
        {"unit_id": 1, "translated_text": "EEN", "field_translations": None},
        {"unit_id": 2, "translated_text": "TWEE", "field_translations": [["a", "b"]]},
        # unit 3 absent: filtered out before translation (preserve_heuristic_text)
    ]
    units = _merge_translations(_GROUPING, translation, page_no=1)
    assert [u["id"] for u in units] == [1, 2]
    assert units[0]["translated_text"] == "EEN"
    assert units[1]["field_translations"] == [["a", "b"]]


def test_merge_translations_flags_orphan_translation_ids() -> None:
    translation = [{"unit_id": 9, "translated_text": "X", "field_translations": None}]
    out = _merge_translations(_GROUPING, translation, page_no=2)
    assert isinstance(out, str) and "page 2" in out and "[9]" in out


def test_merged_units_feed_build_fixture_split() -> None:
    """End of the capture path: hint-matched units key by hint_index, leftover units by their
    anchor cell (lowest order), exactly like the image capture."""
    translation = [
        {"unit_id": 1, "translated_text": "EEN", "field_translations": None},
        {"unit_id": 2, "translated_text": "TWEE", "field_translations": None},
    ]
    units = _merge_translations(_GROUPING, translation, page_no=1)
    response_like = {
        "ocr": {"cells": _GROUPING["cells"], "translation_units": units},
        "metadata": {"target_lang_code": "nl", "grouping_model": "m"},
        "llm_calls": [{"role": "grouping_vlm", "response": {"output_text": "*b:* line"}}],
    }
    fixture = build_fixture(response_like, source_bytes=b"png")
    assert fixture.hint_translations == {"0": {"translated_text": "EEN", "field_translations": None}}
    assert fixture.leftover_translations == {"12": {"translated_text": "TWEE", "field_translations": None}}
    assert fixture.raw_hint == "*b:* line"


def test_grouping_model_prefers_resolved_call_model() -> None:
    calls = [{"role": "grouping_vlm", "payload": {"model": "resolved-model"}}]
    assert _grouping_model(calls, {"grouping_model": "requested"}) == "resolved-model"
    assert _grouping_model([], {"grouping_model": "requested"}) == "requested"
    assert _grouping_model([], {}) == ""


# --- document fixture model + discovery ---------------------------------------------------


def test_document_fixture_roundtrip_and_discovery(tmp_path: Path) -> None:
    root = tmp_path / "_regression"
    variant = root / "01_doc" / "nl" / "v1"
    fixture = dfx.DocumentFixture(
        source_sha256="ab" * 32,
        analysis_dpi=160,
        target_lang="nl",
        census=[{"page": 1, "width_pt": 595.0, "height_pt": 842.0, "cell_source": "ocr"}],
    )
    dfx.save_document(variant, fixture)
    loaded = dfx.load_document(variant)
    assert loaded == fixture
    assert loaded.page_count == 1

    assert dfx.variant_dirs(root) == [("01_doc", "nl", "v1", variant)]
    listed = dfx.list_documents(root)
    assert listed == [{
        "name": "01_doc", "target_lang": "nl", "variant": "v1",
        "pages": 1, "analysis_dpi": 160, "has_accepted_scores": False, "accepted": None,
    }]


def test_page_dirs_only_yields_page_fixtures(tmp_path: Path) -> None:
    variant = tmp_path / "v1"
    good = dfx.page_dir(variant, 1)
    good.mkdir(parents=True)
    (good / "fixture.json").write_text("{}")
    empty = dfx.page_dir(variant, 2)
    empty.mkdir(parents=True)  # no fixture.json -> not a page fixture
    assert dfx.page_dirs(variant) == [good]


# --- /v1/pdf-regression endpoints: the cheap guard paths (no replay, no GPU) --------------


def test_pdf_regression_endpoint_guards(tmp_path: Path) -> None:
    import json

    from fastapi.testclient import TestClient

    from app.main import create_app
    from tests.test_api import _settings_path

    app = create_app(_settings_path(tmp_path))
    with TestClient(app) as client:
        listing = client.get("/v1/pdf-regression/fixtures")
        assert listing.status_code == 200
        assert isinstance(listing.json()["documents"], list)

        assert client.post("/v1/pdf-regression/run", json={}).status_code == 400
        traversal = client.post(
            "/v1/pdf-regression/run", json={"name": "..", "lang": "nl", "variant": "v1"}
        )
        assert traversal.json()["code"] in {"REGRESSION_PATH_INVALID", "REGRESSION_FIXTURE_NOT_FOUND"}
        assert traversal.status_code in {400, 404}
        missing = client.post(
            "/v1/pdf-regression/run", json={"name": "no-such-doc", "lang": "nl", "variant": "v1"}
        )
        assert missing.status_code == 404

        assert client.post("/v1/pdf-regression/capture", json={}).status_code == 400
        assert client.post(
            "/v1/pdf-regression/accept", json={"name": "no-such-doc", "lang": "nl", "variant": "v1"}
        ).status_code == 404
        assert client.delete("/v1/pdf-regression/fixtures/no-such-doc/nl/v1").status_code == 404
        unknown = client.get("/v1/pdf-regression/fixtures/no-such-doc/nl/v1/artifact/weird.bin")
        assert unknown.status_code == 404
        assert unknown.json()["code"] == "REGRESSION_ARTIFACT_UNKNOWN"
