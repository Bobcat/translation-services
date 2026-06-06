from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import tempfile
import threading
from typing import Any

from app.core.config import OcrSettings


_PPSTRUCTURE_LOCK = threading.Lock()
_PPSTRUCTURE_CACHE: dict[tuple[str, str, bool], Any] = {}
_USE_DOCUMENT_UNWARP_PADDING = False


@dataclass(frozen=True)
class DocumentUnwarpedDebug:
    image: bytes | None
    metadata: dict[str, Any]
    cells: list[dict[str, Any]]
    layout_regions: list[dict[str, Any]]
    mime_type: str = "image/png"


def render_paddle_document_unwarped_debug(
    *,
    settings: OcrSettings,
    input_path: Path,
    language: str,
    use_doc_unwarping: bool,
) -> DocumentUnwarpedDebug:
    if settings.backend != "paddleocr":
        return _result(image=None, reason="ocr_backend_is_not_paddleocr")

    padded_input_path: Path | None = None
    try:
        engine = _get_engine(settings=settings, language=language, use_doc_unwarping=use_doc_unwarping)
        if _USE_DOCUMENT_UNWARP_PADDING:
            padded_input_path = _create_padded_input(input_path)
        predict_path = padded_input_path or input_path
        with _PPSTRUCTURE_LOCK:
            result = next(iter(engine.predict(str(predict_path))), None)
    except Exception as exc:
        return _result(image=None, reason="ppstructurev3_failed", error=str(exc))
    finally:
        if padded_input_path is not None:
            padded_input_path.unlink(missing_ok=True)

    if result is None:
        return _result(image=None, reason="ppstructurev3_returned_no_result")

    doc_preprocessor = _dict_like_get(result, "doc_preprocessor_res") or {}
    output_img = _dict_like_get(doc_preprocessor, "output_img")

    overall_ocr = _dict_like_get(result, "overall_ocr_res") or {}
    rec_polys = _as_list(_dict_like_get(overall_ocr, "rec_polys"))
    if not rec_polys:
        rec_polys = _as_list(_dict_like_get(overall_ocr, "dt_polys"))
    rec_texts = _as_list(_dict_like_get(overall_ocr, "rec_texts"))
    rec_scores = _as_list(_dict_like_get(overall_ocr, "rec_scores"))
    layout_det = _dict_like_get(result, "layout_det_res") or {}
    layout_boxes = _as_list(_dict_like_get(layout_det, "boxes"))
    cells = _document_cells(texts=rec_texts, polygons=rec_polys, scores=rec_scores)
    layout_regions = _document_layout_regions(layout_boxes)

    try:
        image = _image_from_array(output_img) if output_img is not None else _image_from_path(input_path)
        debug_image = _draw_unwarped_debug(image=image, polygons=rec_polys, layout_boxes=layout_boxes)
    except Exception as exc:
        return _result(image=None, reason="document_unwarped_debug_render_failed", error=str(exc))

    return _result(
        image=debug_image,
        reason="document_unwarped_debug_applied",
        ocr_segment_count=len(rec_texts),
        polygon_count=len(rec_polys),
        layout_box_count=len(layout_boxes),
        unwarping_enabled=use_doc_unwarping,
        unwarped_image_returned=output_img is not None,
        padding_applied=padded_input_path is not None,
        cells=cells,
        layout_regions=layout_regions,
    )


def _create_padded_input(input_path: Path) -> Path:
    from PIL import Image

    image = Image.open(input_path).convert("RGB")
    width, height = image.size
    padding = max(32, round(max(width, height) * 0.10))
    padded = Image.new("RGB", (width + padding * 2, height + padding * 2), (255, 255, 255))
    padded.paste(image, (padding, padding))

    with tempfile.NamedTemporaryFile(prefix="ocr-unwarp-", suffix=".png", delete=False) as tmp:
        padded.save(tmp, format="PNG")
        return Path(tmp.name)


def _get_engine(*, settings: OcrSettings, language: str, use_doc_unwarping: bool) -> Any:
    key = (str(language or settings.language or "en"), str(settings.device or "cpu"), bool(use_doc_unwarping))
    with _PPSTRUCTURE_LOCK:
        cached = _PPSTRUCTURE_CACHE.get(key)
        if cached is not None:
            return cached
        try:
            from paddleocr import PPStructureV3
        except ImportError as exc:
            raise RuntimeError("PPStructureV3 is not available in paddleocr") from exc

        engine = PPStructureV3(
            lang=key[0],
            device=key[1],
            use_doc_orientation_classify=False,
            use_doc_unwarping=key[2],
            use_textline_orientation=False,
            use_table_recognition=False,
            use_formula_recognition=False,
            use_chart_recognition=False,
            use_seal_recognition=False,
            use_region_detection=False,
        )
        _PPSTRUCTURE_CACHE[key] = engine
        return engine


def _draw_unwarped_debug(*, image: Any, polygons: list[Any], layout_boxes: list[Any]) -> bytes:
    from PIL import Image
    from PIL import ImageDraw

    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for raw_box in layout_boxes:
        box = _layout_box(raw_box)
        if box is None:
            continue
        draw.rectangle(box, outline=(236, 72, 153, 220), width=4)
    for raw_polygon in polygons:
        polygon = _polygon(raw_polygon)
        if polygon is None:
            continue
        draw.polygon(polygon, fill=(255, 255, 255, 72), outline=(34, 197, 94, 230))
        draw.line(polygon + [polygon[0]], fill=(34, 197, 94, 255), width=2)

    out = BytesIO()
    Image.alpha_composite(base, overlay).convert("RGB").save(out, format="PNG")
    return out.getvalue()


def _document_cells(*, texts: list[Any], polygons: list[Any], scores: list[Any]) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for idx, raw_text in enumerate(texts):
        text = str(raw_text or "").strip()
        if not text:
            continue
        polygon = _polygon(polygons[idx]) if idx < len(polygons) else None
        payload: dict[str, Any] = {
            "id": len(cells) + 1,
            "text": text,
        }
        if polygon is not None:
            payload["polygon"] = [{"x": round(x, 2), "y": round(y, 2)} for x, y in polygon]
            payload["bbox"] = _bbox_from_polygon(polygon)
        if idx < len(scores) and _is_number(scores[idx]):
            payload["confidence"] = round(float(scores[idx]), 4)
        cells.append(payload)
    return cells


def _document_layout_regions(layout_boxes: list[Any]) -> list[dict[str, Any]]:
    regions: list[dict[str, Any]] = []
    for raw_box in layout_boxes:
        box = _layout_box(raw_box)
        if box is None:
            continue
        region: dict[str, Any] = {
            "id": len(regions) + 1,
            "bbox": {
                "left": round(box[0], 2),
                "top": round(box[1], 2),
                "width": round(box[2] - box[0], 2),
                "height": round(box[3] - box[1], 2),
            },
        }
        label = _dict_like_get(raw_box, "label") or _dict_like_get(raw_box, "category")
        if label:
            region["label"] = str(label)
        regions.append(region)
    return regions


def _bbox_from_polygon(polygon: list[tuple[float, float]]) -> dict[str, float]:
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    left = min(xs)
    top = min(ys)
    right = max(xs)
    bottom = max(ys)
    return {
        "left": round(left, 2),
        "top": round(top, 2),
        "width": round(right - left, 2),
        "height": round(bottom - top, 2),
    }


def _image_from_array(value: Any) -> Any:
    from PIL import Image

    if hasattr(value, "astype"):
        value = value.astype("uint8")
    return Image.fromarray(value)


def _image_from_path(path: Path) -> Any:
    from PIL import Image

    return Image.open(path).convert("RGB")


def _polygon(value: Any) -> list[tuple[float, float]] | None:
    value = _plain(value)
    if not isinstance(value, list) or len(value) < 4:
        return None
    points: list[tuple[float, float]] = []
    for raw_point in value[:4]:
        point = _plain(raw_point)
        if isinstance(point, (list, tuple)) and len(point) >= 2 and _is_number(point[0]) and _is_number(point[1]):
            points.append((float(point[0]), float(point[1])))
    return points if len(points) == 4 else None


def _layout_box(value: Any) -> tuple[float, float, float, float] | None:
    value = _plain(value)
    if isinstance(value, dict):
        value = value.get("coordinate")
    value = _plain(value)
    if not isinstance(value, list) or len(value) < 4:
        return None
    if not all(_is_number(item) for item in value[:4]):
        return None
    left, top, right, bottom = [float(item) for item in value[:4]]
    return (left, top, right, bottom)


def _dict_like_get(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key)
    try:
        return value[key]
    except (KeyError, TypeError, AttributeError):
        return getattr(value, key, None)


def _as_list(value: Any) -> list[Any]:
    value = _plain(value)
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def _plain(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _is_number(value: Any) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _result(
    *,
    image: bytes | None,
    reason: str,
    ocr_segment_count: int = 0,
    polygon_count: int = 0,
    layout_box_count: int = 0,
    unwarping_enabled: bool = False,
    unwarped_image_returned: bool = False,
    padding_applied: bool = False,
    cells: list[dict[str, Any]] | None = None,
    layout_regions: list[dict[str, Any]] | None = None,
    error: str | None = None,
) -> DocumentUnwarpedDebug:
    cells = list(cells or [])
    layout_regions = list(layout_regions or [])
    metadata: dict[str, Any] = {
        "document_unwarped_debug_applied": image is not None,
        "document_unwarped_debug_reason": reason,
        "document_unwarped_debug_source": "ppstructurev3",
        "document_unwarped_debug_ocr_segment_count": int(ocr_segment_count),
        "document_unwarped_debug_polygon_count": int(polygon_count),
        "document_unwarped_debug_layout_box_count": int(layout_box_count),
        "document_unwarped_debug_unwarping_enabled": bool(unwarping_enabled),
        "document_unwarped_debug_unwarped_image_returned": bool(unwarped_image_returned),
        "document_unwarped_debug_padding_applied": padding_applied,
        "document_cell_count": len(cells),
        "document_layout_region_count": len(layout_regions),
    }
    if error:
        metadata["document_unwarped_debug_error"] = error
    return DocumentUnwarpedDebug(image=image, metadata=metadata, cells=cells, layout_regions=layout_regions)
