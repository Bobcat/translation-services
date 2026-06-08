from __future__ import annotations

from pathlib import Path

from app.core.config import OcrSettings
from app.ocr.paddleocr import resolve_paddleocr_language
from app.ocr.paddleocr import run_paddleocr
from app.ocr.segment import OcrSegment


def run_raw_ocr(
    settings: OcrSettings,
    input_path: Path,
    *,
    language: str | None = None,
) -> list[OcrSegment]:
    _require_paddleocr(settings)
    return run_paddleocr(
        settings,
        input_path,
        language=language,
        merge_lines=False,
    )


def resolve_ocr_language(settings: OcrSettings, source_lang_code: str | None) -> str:
    _require_paddleocr(settings)
    return resolve_paddleocr_language(settings, source_lang_code)


def _require_paddleocr(settings: OcrSettings) -> None:
    backend = str(settings.backend or "").strip().lower()
    if backend != "paddleocr":
        raise RuntimeError(
            f"unsupported OCR backend: {backend or 'unknown'} (only paddleocr is supported)"
        )
