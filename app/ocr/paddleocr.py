from __future__ import annotations

import math
from pathlib import Path
import threading
from typing import Any

import numpy as np
from PIL import Image

from app.core.config import OcrSettings
from app.ocr.merging import merge_same_line_segments
from app.ocr.segment import bbox_polygon
from app.ocr.segment import OcrSegment


# PaddleX's get_rotate_crop_image rotates any recognition crop whose
# height/width >= this ratio by 90 degrees, on the assumption that a tall-thin
# crop is a vertical text line. For isolated tall glyphs (e.g. a single '1' in a
# receipt's quantity column) that assumption is wrong: the rotated digit reads as
# 'T'/'I'/'V'/'7' at low confidence and gets dropped. We re-recognize exactly
# those crops upright. See _correct_rotated_tall_text.
_ROTATED_TALL_RATIO = 1.5


# Languages whose script the multilingual PP-OCRv5 server models cover (Simplified/
# Traditional Chinese, Japanese). These get the server det+rec pair; everything
# else uses the en mobile recognizer — exactly two models, by design (other
# scripts such as Hangul are out of scope). The server recognizer must NOT be used
# for Latin images: it is trained on 48x320 crops (~25 chars max) and degrades
# out-of-distribution on long low-resolution Latin lines — characters garble and
# inter-word spaces disappear — beyond what hint alignment can absorb.
_SERVER_PAIR_LANGUAGES = {"japan", "ch", "chinese_cht"}
_SERVER_PAIR_MODELS = ("PP-OCRv5_server_det", "PP-OCRv5_server_rec")

# Han (U+4E00..U+9FFF) + Hiragana/Katakana (U+3040..U+30FF). A single glyph could
# be VLM hallucination on a Latin page — and the cost of that false positive is
# the server pair's long-line degradation — so require at least two.
_CJK_ROUTING_MIN_GLYPHS = 2


# One lock PER ENGINE: Paddle predictors are not assumed thread-safe, so prediction on one
# engine stays serialized — but two runner slots on different engines (en vs the server pair)
# must not block each other, and building an engine (first server-pair init downloads and
# compiles for seconds to minutes) must not stall a warm engine's prediction. The module lock
# only guards the two dicts and is never held during predict or engine construction.
_PADDLEOCR_DICTS_LOCK = threading.Lock()
_PADDLEOCR_CACHE: dict[tuple[Any, ...], Any] = {}
_PADDLEOCR_ENGINE_LOCKS: dict[tuple[Any, ...], threading.Lock] = {}


def resolve_paddleocr_language(settings: OcrSettings, hint_texts: list[str]) -> str:
    """The PaddleOCR language (= model choice) for this image.

    A configured ``ocr.language`` pins it: the code goes to PaddleOCR verbatim and
    its own lookup picks the models. Otherwise the VLM grouping hint decides —
    enough Han/Kana glyphs in the hint route to the multilingual server pair,
    everything else to the en recognizer. The request's source language plays no
    role: the hint reflects what is actually printed on the image.
    """
    configured = str(settings.language or "").strip()
    if configured:
        return configured
    glyphs = sum(_count_cjk_glyphs(text) for text in hint_texts)
    return "ch" if glyphs >= _CJK_ROUTING_MIN_GLYPHS else "en"


def _count_cjk_glyphs(text: Any) -> int:
    # Han (incl. Extension A) + Kana + HALFWIDTH katakana (U+FF66-FF9F, common on Japanese
    # receipts and a plausible verbatim VLM transcription — without it such an image routes to
    # the en recognizer and the whole read is mangled).
    return sum(
        1
        for ch in str(text or "")
        if "一" <= ch <= "鿿" or "぀" <= ch <= "ヿ" or "㐀" <= ch <= "䶿" or "ｦ" <= ch <= "ﾟ"
    )


def run_paddleocr(
    settings: OcrSettings,
    input_path: Path,
    *,
    language: str | None = None,
    merge_lines: bool = True,
) -> list[OcrSegment]:
    resolved_language = str(language or settings.language or "").strip() or "en"
    engine, engine_lock = _get_paddleocr_engine(settings, resolved_language)

    segments: list[OcrSegment] = []
    with engine_lock:
        try:
            results = engine.predict(str(input_path))
        except Exception as exc:
            raise RuntimeError(f"paddleocr failed: {exc}") from exc

        rec_model: Any = None  # the engine's own rec sub-model
        image_bgr: Any = None  # both resolved lazily, only when a tall crop needs re-recognition
        for result in results or []:
            payload = _paddleocr_result_payload(result)
            texts = _as_list(payload.get("rec_texts"))
            scores = _as_list(payload.get("rec_scores"))
            boxes = _as_list(payload.get("rec_boxes"))
            polys = _as_list(payload.get("rec_polys"))
            if not polys:
                polys = _as_list(payload.get("dt_polys"))

            for idx, raw_text in enumerate(texts):
                text = str(raw_text or "").strip()
                confidence = _parse_paddleocr_confidence(_list_item(scores, idx))
                if confidence is None:
                    # PaddleX 3.6 always emits rec_scores; a missing one means result-shape
                    # drift. Fail SAFE: 0.0 drops the segment at min_confidence rather than
                    # waving unknown-quality text past every filter (1.0 would also make the
                    # upright rescue unreachable).
                    confidence = 0.0
                polygon = _paddleocr_polygon(_list_item(polys, idx))
                bbox = _paddleocr_bbox(_list_item(boxes, idx))
                if bbox is None:
                    bbox = _paddleocr_bbox(polygon)
                if bbox is None:
                    continue
                if polygon is None:
                    polygon = bbox_polygon(bbox)
                # The upright rescue exists for one case: an ISOLATED tall glyph (a receipt
                # quantity digit) that PaddleX rotated 90° and mangled. It must not run on
                # genuine vertical text, judged by the READ itself: a rotated read longer than
                # 2 chars, or one carrying CJK glyphs (a vertical CJK column — its rotated read
                # is the correct one), keeps its result. Gating on content rather than on the
                # routed engine keeps the digit rescue alive on CJK-routed receipts, whose
                # quantity columns are the original motivating case.
                if (
                    len(text) <= 2
                    and not _count_cjk_glyphs(text)
                    and _polygon_is_rotated_tall(polygon)
                ):
                    if rec_model is None:
                        rec_model = engine.paddlex_pipeline.text_rec_model
                        image_bgr = _load_bgr(input_path)
                    try:
                        upright = _recognize_upright(rec_model, image_bgr, bbox)
                    except Exception:
                        upright = None  # a degenerate (sub-pixel) crop must not fail the request
                    # Only a NON-EMPTY upright read may replace the rotated one: an empty read
                    # with a >= score would otherwise delete a valid segment.
                    if upright is not None and upright[0] and upright[1] >= confidence:
                        text, confidence = upright
                if not text:
                    continue
                if confidence < settings.min_confidence:
                    continue
                segments.append(OcrSegment(text=text, bbox=bbox, confidence=confidence, polygon=polygon))
    if not merge_lines:
        return segments
    return merge_same_line_segments(segments)


def _polygon_is_rotated_tall(polygon: list[dict[str, int]] | None) -> bool:
    """Whether PaddleX would have rotated this crop 90 degrees before recognition.

    Mirrors get_rotate_crop_image: it rotates when crop height/width >= 1.5, where
    width/height are the rotated-rectangle edge lengths of the detection polygon
    (top-left, top-right, bottom-right, bottom-left order).
    """
    if not polygon or len(polygon) < 4:
        return False
    pts = [(float(p["x"]), float(p["y"])) for p in polygon[:4]]
    # PaddleX TRUNCATES the edge norms to int before comparing (get_rotate_crop_image:
    # ``int(max(norm(...)))``); mirror that, or a crop in the ~1.40-1.60 band rotates in
    # PaddleX while this predicate says it did not — and the mangled read is never rescued.
    width = int(max(_distance(pts[0], pts[1]), _distance(pts[3], pts[2])))
    height = int(max(_distance(pts[0], pts[3]), _distance(pts[1], pts[2])))
    if width <= 0:
        return False
    return (height / width) >= _ROTATED_TALL_RATIO


def _recognize_upright(rec_model: Any, image_bgr: Any, bbox: dict[str, int]) -> tuple[str, float] | None:
    """Re-run recognition on the axis-aligned (upright) crop of bbox."""
    height, width = image_bgr.shape[:2]
    left = max(0, min(int(bbox["left"]), width - 1))
    top = max(0, min(int(bbox["top"]), height - 1))
    right = max(left + 1, min(int(bbox["left"]) + int(bbox["width"]), width))
    bottom = max(top + 1, min(int(bbox["top"]) + int(bbox["height"]), height))
    crop = image_bgr[top:bottom, left:right]
    if crop.size == 0:
        return None
    for result in rec_model(crop):
        text = str(result.get("rec_text") or "").strip()
        score = _parse_paddleocr_confidence(result.get("rec_score"))
        return text, (score if score is not None else 0.0)
    return None


def _load_bgr(input_path: Path) -> Any:
    rgb = np.asarray(Image.open(input_path).convert("RGB"))
    return np.ascontiguousarray(rgb[:, :, ::-1])


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _get_paddleocr_engine(settings: OcrSettings, language: str) -> tuple[Any, threading.Lock]:
    """The engine for this language/config plus ITS lock — the caller must hold that lock for
    every ``predict`` (see the note at ``_PADDLEOCR_DICTS_LOCK``). Construction happens under
    the engine's own lock (double-checked), so a slow first build never blocks other engines."""
    det_model = str(settings.det_model or "").strip()
    rec_model = str(settings.rec_model or "").strip()
    lang = str(language or "en")
    if not det_model and not rec_model and lang in _SERVER_PAIR_LANGUAGES:
        det_model, rec_model = _SERVER_PAIR_MODELS
    # Explicit model names override PaddleOCR's lang-based selection. When both are
    # set the engine no longer depends on the source language, so collapse it in the
    # cache key to avoid spinning up one (heavy) engine per source language.
    lang_key = "*" if (det_model and rec_model) else lang
    key = (
        lang_key,
        det_model,
        rec_model,
        str(settings.ocr_version or "PP-OCRv5"),
        str(settings.device or "cpu"),
        int(settings.text_det_limit_side_len),
        str(settings.text_det_limit_type),
    )
    with _PADDLEOCR_DICTS_LOCK:
        cached = _PADDLEOCR_CACHE.get(key)
        engine_lock = _PADDLEOCR_ENGINE_LOCKS.setdefault(key, threading.Lock())
    if cached is not None:
        return cached, engine_lock

    with engine_lock:
        cached = _PADDLEOCR_CACHE.get(key)  # double-check: a racing caller may have built it
        if cached is not None:
            return cached, engine_lock
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise RuntimeError("OCR backend paddleocr is configured but paddleocr is not installed") from exc

        try:
            engine = PaddleOCR(
                lang=lang,
                ocr_version=str(settings.ocr_version or "PP-OCRv5"),
                device=str(settings.device or "cpu"),
                text_detection_model_name=det_model or None,
                text_recognition_model_name=rec_model or None,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                text_det_limit_side_len=int(settings.text_det_limit_side_len),
                text_det_limit_type=str(settings.text_det_limit_type),
            )
        except Exception as exc:
            raise RuntimeError(f"failed to initialize paddleocr: {exc}") from exc
        with _PADDLEOCR_DICTS_LOCK:
            _PADDLEOCR_CACHE[key] = engine
        return engine, engine_lock


def _paddleocr_result_payload(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        return result
    try:
        return dict(result)
    except (TypeError, ValueError):
        pass

    payload: dict[str, Any] = {}
    for key in ("rec_texts", "rec_scores", "rec_boxes", "rec_polys", "dt_polys"):
        try:
            payload[key] = result[key]
            continue
        except (KeyError, TypeError, AttributeError):
            pass
        if hasattr(result, key):
            payload[key] = getattr(result, key)
    return payload


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def _list_item(values: list[Any], idx: int) -> Any:
    if idx < len(values):
        return values[idx]
    return None


def _parse_paddleocr_confidence(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _paddleocr_bbox(value: Any) -> dict[str, int] | None:
    value = _to_plain_value(value)
    if not isinstance(value, list) or not value:
        return None

    if len(value) == 4 and all(_is_number(item) for item in value):
        left, top, right, bottom = [float(item) for item in value]
    else:
        points: list[tuple[float, float]] = []
        for raw_point in value:
            point = _to_plain_value(raw_point)
            if isinstance(point, (list, tuple)) and len(point) >= 2 and _is_number(point[0]) and _is_number(point[1]):
                points.append((float(point[0]), float(point[1])))
            elif isinstance(point, dict) and _is_number(point.get("x")) and _is_number(point.get("y")):
                points.append((float(point["x"]), float(point["y"])))
        if not points:
            return None
        left = min(point[0] for point in points)
        top = min(point[1] for point in points)
        right = max(point[0] for point in points)
        bottom = max(point[1] for point in points)

    left_i = max(0, int(round(min(left, right))))
    top_i = max(0, int(round(min(top, bottom))))
    right_i = max(left_i, int(round(max(left, right))))
    bottom_i = max(top_i, int(round(max(top, bottom))))
    return {
        "left": left_i,
        "top": top_i,
        "width": max(0, right_i - left_i),
        "height": max(0, bottom_i - top_i),
    }


def _paddleocr_polygon(value: Any) -> list[dict[str, int]] | None:
    value = _to_plain_value(value)
    if not isinstance(value, list) or not value:
        return None

    if len(value) == 4 and all(_is_number(item) for item in value):
        bbox = _paddleocr_bbox(value)
        return bbox_polygon(bbox) if bbox is not None else None

    points: list[dict[str, int]] = []
    for raw_point in value:
        point = _to_plain_value(raw_point)
        if isinstance(point, (list, tuple)) and len(point) >= 2 and _is_number(point[0]) and _is_number(point[1]):
            points.append({"x": max(0, int(round(float(point[0])))), "y": max(0, int(round(float(point[1]))))})
    if len(points) < 4:
        return None
    return points


def _to_plain_value(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _is_number(value: Any) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False
