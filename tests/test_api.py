from __future__ import annotations

from io import BytesIO
import json
import time
from pathlib import Path

from fastapi.testclient import TestClient

from app.core.config import AppSettings
from app.core.config import LlmPoolSettings
from app.core.config import OcrSettings
from app.core.config import TranslationRouteSettings
from app.main import create_app
from app.ocr import OcrSegment
from app.ocr import resolve_ocr_language
from app.ocr.merging import merge_same_line_segments
from app.ocr.overlay import render_original_ocr_overlay_debug
from app.ocr.paddleocr import run_paddleocr
from app.tasks.translate_image import TranslateImageResult
from app.translation.routing import resolve_translation_route


def _png_bytes() -> bytes:
    from PIL import Image

    out = BytesIO()
    Image.new("RGB", (1, 1), (255, 255, 255)).save(out, format="PNG")
    return out.getvalue()


PNG_BYTES = _png_bytes()


def _settings_path(tmp_path: Path) -> Path:
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(
        json.dumps(
            {
                "service": {
                    "host": "127.0.0.1",
                    "port": 8030,
                    "work_root": str(tmp_path / "work"),
                },
                "scheduler": {
                    "runner_slots": 1,
                    "queue_limit": 4,
                    "records_max": 100,
                    "records_ttl_s": {"completed": 900, "failed": 900, "cancelled": 900},
                },
                "llm_pool": {
                    "base_url": "http://127.0.0.1:8011",
                    "translator_model": "translategemma-4b-it-q5-k-m-gguf",
                    "translator_mode": "translategemma",
                },
                "ocr": {
                    "backend": "paddleocr",
                    "language": "en",
                    "min_confidence": 0.35,
                },
            }
        ),
        encoding="utf-8",
    )
    return settings_path


def _submit(client: TestClient, request_payload: dict[str, object]) -> dict[str, object]:
    response = client.post(
        "/v1/requests",
        files={
            "request_json": (None, json.dumps(request_payload), "application/json"),
            "image_file": ("input.png", PNG_BYTES, "image/png"),
        },
    )
    assert response.status_code == 202
    return response.json()


def _wait_completed(client: TestClient, request_id: str) -> dict[str, object]:
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        response = client.get(f"/v1/requests/{request_id}")
        assert response.status_code == 200
        body = response.json()
        if body["state"] == "completed":
            return body
        time.sleep(0.05)
    raise AssertionError("request did not complete")


def test_translate_image_pipeline_response(tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_pipeline(*, settings, input_path, input_mime_type, request):
        captured["translator_model"] = settings.llm_pool.translator_model
        captured["input_mime_type"] = input_mime_type
        captured["request"] = request
        return TranslateImageResult(
            image=PNG_BYTES,
            mime_type="image/png",
            debug_image=PNG_BYTES,
            debug_mime_type="image/png",
            rectified_debug_image=PNG_BYTES,
            rectified_debug_mime_type="image/png",
            projected_overlay_debug_image=PNG_BYTES,
            projected_overlay_debug_mime_type="image/png",
            segments=[
                {
                    "id": 1,
                    "text": "Hello world",
                    "translated_text": "Hallo wereld",
                    "bbox": {"left": 1, "top": 2, "width": 30, "height": 10},
                    "confidence": 0.91,
                }
            ],
            metadata={"ocr_backend": "paddleocr", "translator_model": "translategemma-4b-it-q5-k-m-gguf"},
            metrics={"ocr_segment_count": 1, "llm_pool_request_count": 1},
        )

    monkeypatch.setattr("app.runtime.service.run_translate_image_pipeline", fake_pipeline)
    app = create_app(_settings_path(tmp_path))
    with TestClient(app) as client:
        submitted = _submit(
            client,
            {
                "request_id": "req_translate",
                "task": "translate_image",
                "source_lang_code": "en",
                "target_lang_code": "nl",
            },
        )
        completed = _wait_completed(client, "req_translate")
        assert completed["response"]["segments"][0]["translated_text"] == "Hallo wereld"
        assert completed["response"]["artifacts"]["output"]["mime_type"] == "image/png"
        assert completed["response"]["artifacts"]["debug_overlay"]["mime_type"] == "image/png"
        assert completed["response"]["artifacts"]["rectified_debug"]["mime_type"] == "image/png"
        assert completed["response"]["artifacts"]["projected_overlay_debug"]["mime_type"] == "image/png"
        assert completed["response"]["artifacts"]["segments"]["mime_type"] == "application/json"
        assert captured["translator_model"] == "translategemma-4b-it-q5-k-m-gguf"

        output_artifact = client.get("/v1/requests/req_translate/artifacts/output")
        assert output_artifact.status_code == 200
        assert output_artifact.content.startswith(b"\x89PNG\r\n\x1a\n")

        artifact = client.get("/v1/requests/req_translate/artifacts/segments")
        assert artifact.status_code == 200
        assert artifact.json()["segments"][0]["text"] == "Hello world"

        completions = client.get("/v1/completions")
        assert completions.status_code == 200
        events = completions.json()["events"]
        assert len(events) == 1
        assert events[0]["request_id"] == "req_translate"
        assert events[0]["state"] == "completed"


def test_status(tmp_path: Path) -> None:
    app = create_app(_settings_path(tmp_path))
    with TestClient(app) as client:
        resp = client.get("/v1/status")
        assert resp.status_code == 200
        assert resp.json()["runner_slots"] == 1


def test_upload_canonicalizes_exif_orientation(tmp_path: Path, monkeypatch) -> None:
    from PIL import Image

    def fake_pipeline(*, settings, input_path, input_mime_type, request):
        del settings, input_path, input_mime_type, request
        return TranslateImageResult(image=PNG_BYTES, mime_type="image/png", segments=[], metadata={}, metrics={})

    monkeypatch.setattr("app.runtime.service.run_translate_image_pipeline", fake_pipeline)

    image = Image.new("RGB", (2, 1), (255, 255, 255))
    image.putpixel((0, 0), (255, 0, 0))
    image.putpixel((1, 0), (0, 0, 255))
    exif = Image.Exif()
    exif[274] = 6
    raw = BytesIO()
    image.save(raw, format="JPEG", exif=exif)

    app = create_app(_settings_path(tmp_path))
    with TestClient(app) as client:
        response = client.post(
            "/v1/requests",
            files={
                "request_json": (
                    None,
                    json.dumps(
                        {
                            "request_id": "req_exif",
                            "task": "translate_image",
                            "source_lang_code": "en",
                            "target_lang_code": "nl",
                        }
                    ),
                    "application/json",
                ),
                "image_file": ("phone.jpg", raw.getvalue(), "image/jpeg"),
            },
        )
        assert response.status_code == 202
        completed = _wait_completed(client, "req_exif")
        input_artifact = client.get("/v1/requests/req_exif/artifacts/input")
        assert input_artifact.status_code == 200
        with Image.open(BytesIO(input_artifact.content)) as stored:
            assert stored.size == (1, 2)
            assert stored.getexif().get(274) is None
        assert completed["response"]["artifacts"]["input"]["mime_type"] == "image/jpeg"


def test_resolve_ocr_language_maps_paddleocr_codes() -> None:
    settings = OcrSettings(backend="paddleocr", language="")

    assert resolve_ocr_language(settings, "en") == "en"
    assert resolve_ocr_language(settings, "nl") == "nl"
    assert resolve_ocr_language(settings, "ja") == "japan"
    assert resolve_ocr_language(settings, "zh-tw") == "chinese_cht"
    assert resolve_ocr_language(settings, "unknown") == "en"
    assert resolve_ocr_language(OcrSettings(backend="paddleocr", language="latin"), "en") == "latin"


def test_paddleocr_backend_normalizes_chunks_to_visual_lines(tmp_path: Path, monkeypatch) -> None:
    class FakePaddleOCR:
        def predict(self, input_path: str):
            assert input_path.endswith("input.jpg")
            return [
                {
                    "rec_texts": ["THE", "SHOE", "WORKS", "IF", "SKEW", "noise"],
                    "rec_scores": [0.999, 0.998, 0.997, 0.996, 0.995, 0.1],
                    "rec_boxes": [
                        [118, 24, 326, 139],
                        [348, 26, 628, 139],
                        [114, 124, 514, 241],
                        [537, 132, 638, 235],
                        [10, 260, 110, 300],
                        [1, 300, 10, 310],
                    ],
                    "rec_polys": [
                        [[118, 24], [326, 24], [326, 139], [118, 139]],
                        [[348, 26], [628, 26], [628, 139], [348, 139]],
                        [[114, 124], [514, 124], [514, 241], [114, 241]],
                        [[537, 132], [638, 132], [638, 235], [537, 235]],
                        [[10, 260], [110, 250], [120, 290], [20, 300]],
                        [[1, 300], [10, 300], [10, 310], [1, 310]],
                    ],
                }
            ]

    languages: list[str] = []

    def fake_get_paddleocr_engine(settings, language, *, use_doc_unwarping=None):
        del use_doc_unwarping
        languages.append(language)
        return FakePaddleOCR()

    monkeypatch.setattr("app.ocr.paddleocr._get_paddleocr_engine", fake_get_paddleocr_engine)
    input_path = tmp_path / "input.jpg"
    input_path.write_bytes(b"fake")

    segments = run_paddleocr(
        OcrSettings(backend="paddleocr", language="en", min_confidence=0.35),
        input_path,
        language="en",
    )

    assert languages == ["en"]
    assert [segment.text for segment in segments] == ["THE SHOE", "WORKS IF", "SKEW"]
    assert segments[0].bbox == {"left": 118, "top": 24, "width": 510, "height": 115}
    assert segments[1].bbox == {"left": 114, "top": 124, "width": 524, "height": 117}
    assert segments[2].polygon == [
        {"x": 10, "y": 260},
        {"x": 110, "y": 250},
        {"x": 120, "y": 290},
        {"x": 20, "y": 300},
    ]


def test_merge_same_line_segments_preserves_separate_rows() -> None:
    segments = merge_same_line_segments(
        [
            OcrSegment(text="THE", bbox={"left": 118, "top": 24, "width": 208, "height": 115}, confidence=0.99),
            OcrSegment(text="SHOE", bbox={"left": 348, "top": 26, "width": 280, "height": 113}, confidence=0.98),
            OcrSegment(text="body line one", bbox={"left": 51, "top": 683, "width": 650, "height": 28}, confidence=0.94),
            OcrSegment(text="body line two", bbox={"left": 51, "top": 724, "width": 650, "height": 28}, confidence=0.91),
        ]
    )

    assert [segment.text for segment in segments] == ["THE SHOE", "body line one", "body line two"]


def test_merge_same_line_segments_preserves_skewed_row_polygon() -> None:
    segments = merge_same_line_segments(
        [
            OcrSegment(
                text="LEFT",
                bbox={"left": 10, "top": 10, "width": 44, "height": 22},
                confidence=0.9,
                polygon=[
                    {"x": 10, "y": 10},
                    {"x": 54, "y": 18},
                    {"x": 52, "y": 32},
                    {"x": 8, "y": 24},
                ],
            ),
            OcrSegment(
                text="RIGHT",
                bbox={"left": 60, "top": 19, "width": 50, "height": 21},
                confidence=0.9,
                polygon=[
                    {"x": 60, "y": 19},
                    {"x": 110, "y": 28},
                    {"x": 108, "y": 40},
                    {"x": 58, "y": 31},
                ],
            ),
        ]
    )

    assert [segment.text for segment in segments] == ["LEFT RIGHT"]
    assert segments[0].polygon is not None
    assert segments[0].polygon[1]["y"] > segments[0].polygon[0]["y"]


def test_resolve_translation_route_uses_source_target_overrides() -> None:
    settings = AppSettings(
        llm_pool=LlmPoolSettings(
            literal_translator_model="google_gemma-4-E4B-it-Q8_0-gguf",
            literal_translator_mode="generic",
            translation_routes={
                "*:nl": TranslationRouteSettings(
                    literal_translator_model="eurollm-9b-instruct-q5-k-m-gguf",
                    literal_translator_mode="generic",
                ),
                "en:nl": TranslationRouteSettings(
                    literal_translator_model="eurollm-22b-instruct-2512-q5-k-m-gguf",
                    literal_translator_mode="generic",
                ),
            },
        )
    )

    exact = resolve_translation_route(
        settings=settings,
        translator_model="translategemma-4b-it-q5-k-m-gguf",
        translator_mode="auto",
        source_lang_code="en",
        target_lang_code="nl",
        source_text="THE SHOE WORKS IF YOU DO.",
    )
    wildcard = resolve_translation_route(
        settings=settings,
        translator_model="translategemma-4b-it-q5-k-m-gguf",
        translator_mode="auto",
        source_lang_code="de",
        target_lang_code="nl",
        source_text="THE SHOE WORKS IF YOU DO.",
    )

    assert exact.translator_model == "eurollm-22b-instruct-2512-q5-k-m-gguf"
    assert exact.translator_mode == "generic"
    assert exact.translation_route == "literal_generic"
    assert exact.route_key == "en:nl"
    assert wildcard.translator_model == "eurollm-9b-instruct-q5-k-m-gguf"
    assert wildcard.route_key == "*:nl"


def test_original_ocr_overlay_debug_draws_original_boxes(tmp_path: Path) -> None:
    from PIL import Image

    input_path = tmp_path / "input.png"
    Image.new("RGB", (80, 80), (20, 30, 40)).save(input_path)

    overlay = render_original_ocr_overlay_debug(
        input_path=input_path,
        ocr_segments=[
            OcrSegment(
                text="DANGER",
                bbox={"left": 20, "top": 20, "width": 30, "height": 18},
                confidence=0.99,
            )
        ],
    )

    assert overlay.image is not None
    assert overlay.metadata["projected_overlay_debug_applied"] is True
    assert overlay.metadata["projected_overlay_debug_reason"] == "original_ocr_overlay_debug_applied"
    assert overlay.metadata["projected_overlay_debug_source"] == "original_ocr"
    assert overlay.metadata["projected_overlay_debug_segment_count"] == 1

    output_path = tmp_path / "overlay.png"
    output_path.write_bytes(overlay.image)
    with Image.open(output_path) as image:
        assert image.getpixel((5, 5)) == (20, 30, 40)
        assert image.getpixel((25, 25)) != (20, 30, 40)
