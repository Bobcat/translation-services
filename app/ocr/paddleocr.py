from __future__ import annotations

from pathlib import Path
import threading
from typing import Any

from app.core.config import OcrSettings
from app.ocr.merging import merge_same_line_segments
from app.ocr.segment import bbox_polygon
from app.ocr.segment import OcrSegment


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
_PADDLEOCR_CACHE: dict[tuple[str, str, str, bool, bool, bool], Any] = {}


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
    use_doc_unwarping: bool | None = None,
) -> list[OcrSegment]:
    resolved_language = str(language or settings.language or "").strip() or "en"
    engine = _get_paddleocr_engine(settings, resolved_language, use_doc_unwarping=use_doc_unwarping)
    try:
        with _PADDLEOCR_LOCK:
            results = engine.predict(str(input_path))
    except Exception as exc:
        raise RuntimeError(f"paddleocr failed: {exc}") from exc

    segments: list[OcrSegment] = []
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
            if not text:
                continue
            confidence = _parse_paddleocr_confidence(_list_item(scores, idx))
            if confidence is None:
                confidence = 1.0
            if confidence < settings.min_confidence:
                continue
            polygon = _paddleocr_polygon(_list_item(polys, idx))
            bbox = _paddleocr_bbox(_list_item(boxes, idx))
            if bbox is None:
                bbox = _paddleocr_bbox(polygon)
            if bbox is None:
                continue
            if polygon is None:
                polygon = bbox_polygon(bbox)
            segments.append(OcrSegment(text=text, bbox=bbox, confidence=confidence, polygon=polygon))
    if not merge_lines:
        return segments
    return merge_same_line_segments(segments)


def _get_paddleocr_engine(settings: OcrSettings, language: str, *, use_doc_unwarping: bool | None = None) -> Any:
    doc_unwarping = bool(settings.use_doc_unwarping if use_doc_unwarping is None else use_doc_unwarping)
    key = (
        str(language or "en"),
        str(settings.ocr_version or "PP-OCRv5"),
        str(settings.device or "cpu"),
        bool(settings.use_doc_orientation_classify),
        doc_unwarping,
        bool(settings.use_textline_orientation),
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
                lang=key[0],
                ocr_version=key[1],
                device=key[2],
                use_doc_orientation_classify=key[3],
                use_doc_unwarping=key[4],
                use_textline_orientation=key[5],
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
