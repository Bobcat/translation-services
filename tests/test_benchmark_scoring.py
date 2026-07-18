"""Unit tests for app.benchmark.scoring — pure functions over synthetic measurements.

The identity case anchors the scale corners (layout 100, retention 100,
unchanged 100% — axes are observations, intent is not scored); the mutation
cases each move exactly one axis or indicator. The v3 matching cases cover the
detector-granularity artifacts measured on real runs: duplicate detections,
splits/merges (covered), nested detections, and genuinely invented content.
"""
from __future__ import annotations

import copy
import math
from typing import Any

from app.benchmark.scoring import score_measurement


def _segment(text: str, left: float, top: float, width: float = 180.0, height: float = 16.0) -> dict[str, Any]:
    return {"text": text, "bbox": {"left": left, "top": top, "width": width, "height": height}, "confidence": 0.99}


def _page(index: int = 0) -> dict[str, Any]:
    """One page: a title region + a text region with two prose segments, and an image region."""
    return {
        "index": index,
        "width_pt": 595.0,
        "height_pt": 842.0,
        "width_px": 1323,
        "height_px": 1871,
        "regions": [
            {"label": "doc_title", "score": 0.95, "coordinate": [100.0, 80.0, 900.0, 140.0]},
            {"label": "text", "score": 0.92, "coordinate": [100.0, 200.0, 1200.0, 600.0]},
            {"label": "image", "score": 0.9, "coordinate": [100.0, 700.0, 600.0, 1100.0]},
        ],
        "segments": [
            _segment("Drug Safety Update", 120.0, 90.0),
            _segment("There is a small increase in the risk", 120.0, 220.0, width=600.0),
            _segment("Healthcare professionals should advise", 120.0, 260.0, width=600.0),
        ],
    }


def _weight(box: list[float]) -> float:
    return math.sqrt((box[2] - box[0]) * (box[3] - box[1]))


_TITLE_W = _weight([100.0, 80.0, 900.0, 140.0])
_TEXT_W = _weight([100.0, 200.0, 1200.0, 600.0])
_IMAGE_W = _weight([100.0, 700.0, 600.0, 1100.0])


def _measurement(source_pages: list[dict[str, Any]], target_pages: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "analysis_dpi": 160,
        "models": {"layout": "PP-DocLayout_plus-L", "ocr_language": "en"},
        "source": {"page_count": len(source_pages), "pages": source_pages},
        "translated": {"page_count": len(target_pages), "pages": target_pages},
    }


def _translated_page() -> dict[str, Any]:
    """The same page 'translated': same regions, Dutch text of similar shape."""
    page = _page()
    page["segments"] = [
        _segment("Update over medicijnveiligheid", 120.0, 90.0),
        _segment("Er is een kleine toename van het risico", 120.0, 220.0, width=600.0),
        _segment("Zorgverleners moeten adviseren", 120.0, 260.0, width=600.0),
    ]
    return page


def test_identity_anchors_the_corners() -> None:
    scores = score_measurement(_measurement([_page()], [copy.deepcopy(_page())]))
    assert scores["axes"]["layout"] == 100.0
    assert scores["axes"]["retention"] == 100.0  # nothing lost — the source IS present
    assert scores["axes"]["typography"] == 100.0
    assert scores["indicators"]["unchanged_share"] == 100.0  # ...but nothing changed either
    assert scores["indicators"]["changed_segments"] == 0
    assert scores["flags"]["page_count_equal"] is True
    assert scores["flags"]["image_regions_equal"] is True


def test_clean_translation_changes_everything() -> None:
    scores = score_measurement(_measurement([_page()], [_translated_page()]))
    assert scores["axes"]["layout"] == 100.0
    assert scores["axes"]["retention"] == 100.0
    assert scores["indicators"]["unchanged_share"] == 0.0
    assert scores["indicators"]["changed_segments"] == 3


def test_one_unchanged_segment_moves_the_indicator_not_retention() -> None:
    target = _translated_page()
    target["segments"][1] = _segment("There is a small increase in the risk", 120.0, 220.0, width=600.0)
    scores = score_measurement(_measurement([_page()], [target]))
    raw = scores["per_page"][0]["raw"]
    assert raw["unchanged_segments"] == 1
    assert raw["eligible_source_segments"] == 3
    assert scores["axes"]["retention"] == 100.0
    assert scores["indicators"]["unchanged_share"] == round(100 / 3, 2)


def test_lost_region_lowers_layout_by_weight() -> None:
    target = _translated_page()
    target["regions"] = [region for region in target["regions"] if region["label"] != "image"]
    scores = score_measurement(_measurement([_page()], [target]))
    raw = scores["per_page"][0]["raw"]
    assert raw["regions_lost"] == 1
    assert raw["regions_covered_source"] == 0
    expected = 100 * (_TITLE_W + _TEXT_W) / (_TITLE_W + _TEXT_W + _IMAGE_W)
    assert scores["axes"]["layout"] == round(expected, 2)
    assert scores["flags"]["image_regions_equal"] is False


def test_missing_text_region_content_lowers_retention() -> None:
    target = _translated_page()
    # The body region keeps its box but loses all its text.
    target["segments"] = [seg for seg in target["segments"] if seg["bbox"]["top"] < 200]
    scores = score_measurement(_measurement([_page()], [target]))
    raw = scores["per_page"][0]["raw"]
    assert raw["missing_segments"] == 2
    assert scores["axes"]["retention"] == round(100 * (1 - 2 / 3), 2)
    assert scores["indicators"]["missing_share"] == round(100 * 2 / 3, 2)


def test_moved_region_lowers_layout_via_iou() -> None:
    target = _translated_page()
    target["regions"][1] = {"label": "text", "score": 0.92, "coordinate": [100.0, 300.0, 1200.0, 700.0]}
    scores = score_measurement(_measurement([_page()], [target]))
    assert 0 < scores["axes"]["layout"] < 100.0
    assert scores["per_page"][0]["raw"]["regions_matched"] == 3


def test_split_region_is_covered_not_invented() -> None:
    target = _translated_page()
    # The detector cut the translated body block in two halves.
    target["regions"][1] = {"label": "text", "score": 0.92, "coordinate": [100.0, 200.0, 1200.0, 395.0]}
    target["regions"].append({"label": "text", "score": 0.9, "coordinate": [100.0, 405.0, 1200.0, 600.0]})
    scores = score_measurement(_measurement([_page()], [target]))
    raw = scores["per_page"][0]["raw"]
    assert raw["regions_invented"] == 0
    assert raw["regions_covered_translated"] == 1
    assert raw["regions_matched"] == 3  # title, one half, image
    # The split still costs via the halved IoU of the matched pair, nothing more.
    assert 70 < scores["axes"]["layout"] < 100


def test_nested_detection_is_covered_not_lost() -> None:
    source = _page()
    # The detector reported the body block AND one of its lines separately.
    source["regions"].append({"label": "text", "score": 0.88, "coordinate": [100.0, 210.0, 1190.0, 260.0]})
    scores = score_measurement(_measurement([source], [_translated_page()]))
    raw = scores["per_page"][0]["raw"]
    assert raw["regions_lost"] == 0
    assert raw["regions_covered_source"] == 1
    assert scores["axes"]["layout"] == 100.0


def test_uncovered_extra_region_is_invented() -> None:
    target = _translated_page()
    # A genuinely new block (a translator watermark) far from any source region.
    target["regions"].append({"label": "text", "score": 0.9, "coordinate": [1000.0, 1600.0, 1800.0, 1700.0]})
    scores = score_measurement(_measurement([_page()], [target]))
    raw = scores["per_page"][0]["raw"]
    assert raw["regions_invented"] == 1
    assert raw["regions_covered_translated"] == 0
    assert scores["axes"]["layout"] < 100.0


def test_duplicate_detection_is_deduped() -> None:
    source = _page()
    source["regions"].append({"label": "doc_title", "score": 0.55, "coordinate": [100.0, 80.0, 900.0, 140.0]})
    scores = score_measurement(_measurement([source], [_translated_page()]))
    raw = scores["per_page"][0]["raw"]
    assert raw["regions_source"] == 3  # the duplicate never enters the matching
    assert raw["regions_lost"] == 0
    assert scores["axes"]["layout"] == 100.0


def test_document_layout_aggregates_by_region_weight() -> None:
    # Page A: rich page, all matched. Page B: two tiny regions, one lost.
    page_a_src, page_a_tgt = _page(0), _translated_page()
    page_b_src = {
        "index": 1, "width_pt": 595.0, "height_pt": 842.0, "width_px": 1323, "height_px": 1871,
        "regions": [
            {"label": "text", "score": 0.9, "coordinate": [100.0, 100.0, 200.0, 120.0]},
            {"label": "text", "score": 0.9, "coordinate": [100.0, 900.0, 200.0, 920.0]},
        ],
        "segments": [],
    }
    page_b_tgt = copy.deepcopy(page_b_src)
    page_b_tgt["regions"] = page_b_tgt["regions"][:1]
    scores = score_measurement(_measurement([page_a_src, page_b_src], [page_a_tgt, page_b_tgt]))
    # Unweighted page mean would be (100 + 50) / 2 = 75; region weighting keeps
    # the document score near the rich page.
    assert scores["axes"]["layout"] > 95.0
    assert scores["per_page"][1]["raw"]["regions_lost"] == 1


def test_size_ratio_drift_lowers_typography() -> None:
    source = _page()
    # Second text region so a ratio spread can exist.
    source["regions"].append({"label": "text", "score": 0.9, "coordinate": [700.0, 700.0, 1200.0, 1100.0]})
    source["segments"].append(_segment("A second block of body text here", 720.0, 800.0, width=400.0))
    target = copy.deepcopy(source)
    target["segments"][1]["text"] = "Vertaalde eerste alinea van de tekst"
    target["segments"][2]["text"] = "Vertaalde tweede alinea van de tekst"
    target["segments"][3] = _segment("Vertaald tweede blok lichaamstekst", 720.0, 800.0, width=400.0, height=8.0)
    scores = score_measurement(_measurement([source], [target]))
    raw = scores["per_page"][0]["raw"]
    assert raw["size_ratio_drift"] > 0.5
    assert scores["axes"]["typography"] < 100.0


def test_page_count_mismatch_flags() -> None:
    scores = score_measurement(_measurement([_page(0), _page(1)], [_translated_page()]))
    assert scores["flags"]["page_count_equal"] is False
    assert scores["flags"]["page_count_source"] == 2
    assert scores["flags"]["page_count_translated"] == 1


def test_stray_text_lowers_typography() -> None:
    target = _translated_page()
    # A big text segment far outside every region.
    target["segments"].append(_segment("Zwevende tekst buiten alle regio's", 1000.0, 1600.0, width=800.0, height=200.0))
    scores = score_measurement(_measurement([_page()], [target]))
    raw = scores["per_page"][0]["raw"]
    assert raw["stray_text_share"] > 0.3
    assert scores["axes"]["typography"] < 90.0
