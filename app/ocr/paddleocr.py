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


_PADDLEOCR_LANGUAGE_BY_SOURCE = {
    "en": "en",
    "ja": "japan",
    "jp": "japan",
    "ko": "korean",
    "zh": "ch",
    "zh-cn": "ch",
    "zh-tw": "chinese_cht",
}


_PADDLEOCR_LOCK = threading.Lock()
_PADDLEOCR_CACHE: dict[tuple[Any, ...], Any] = {}


def resolve_paddleocr_language(settings: OcrSettings, source_lang_code: str | None) -> str:
    source = str(source_lang_code or "").strip().lower()
    configured = str(settings.language or "").strip()
    if configured:
        return configured
    if source in _PADDLEOCR_LANGUAGE_BY_SOURCE:
        return _PADDLEOCR_LANGUAGE_BY_SOURCE[source]
    if source and len(source) <= 3:
        return source
    return "en"


def run_paddleocr(
    settings: OcrSettings,
    input_path: Path,
    *,
    language: str | None = None,
    merge_lines: bool = True,
) -> list[OcrSegment]:
    resolved_language = str(language or settings.language or "").strip() or "en"
    engine = _get_paddleocr_engine(settings, resolved_language)

    segments: list[OcrSegment] = []
    with _PADDLEOCR_LOCK:
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
                    confidence = 1.0
                polygon = _paddleocr_polygon(_list_item(polys, idx))
                bbox = _paddleocr_bbox(_list_item(boxes, idx))
                if bbox is None:
                    bbox = _paddleocr_bbox(polygon)
                if bbox is None:
                    continue
                if polygon is None:
                    polygon = bbox_polygon(bbox)
                if _polygon_is_rotated_tall(polygon):
                    if rec_model is None:
                        rec_model = engine.paddlex_pipeline.text_rec_model
                        image_bgr = _load_bgr(input_path)
                    upright = _recognize_upright(rec_model, image_bgr, bbox)
                    if upright is not None and upright[1] >= confidence:
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
    width = max(_distance(pts[0], pts[1]), _distance(pts[3], pts[2]))
    height = max(_distance(pts[0], pts[3]), _distance(pts[1], pts[2]))
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


def _get_paddleocr_engine(settings: OcrSettings, language: str) -> Any:
    det_model = str(settings.det_model or "").strip()
    rec_model = str(settings.rec_model or "").strip()
    lang = str(language or "en")
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
    with _PADDLEOCR_LOCK:
        cached = _PADDLEOCR_CACHE.get(key)
        if cached is not None:
            return cached
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
        _PADDLEOCR_CACHE[key] = engine
        return engine


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
