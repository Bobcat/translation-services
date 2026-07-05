from __future__ import annotations

from io import BytesIO
import json
import threading
import time
from pathlib import Path

from fastapi.testclient import TestClient

from app.core.config import AppSettings
from app.core.config import OcrSettings
from app.main import create_app
from app.ocr import OcrSegment
from app.ocr import resolve_ocr_language
from app.ocr.merging import merge_same_line_segments
from app.ocr.overlay import render_original_ocr_overlay_debug
from app.ocr.paddleocr import _polygon_is_rotated_tall
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

    def fake_pipeline(*, settings, input_path, input_mime_type, request, checkpoint):
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
                "preserve_heuristic_text": False,
                "preserve_unchanged_text": True,
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
        assert captured["request"]["preserve_heuristic_text"] is False
        assert captured["request"]["preserve_unchanged_text"] is True

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


def test_cancel_mid_run_stops_at_next_checkpoint(tmp_path: Path, monkeypatch) -> None:
    # Cancelling a RUNNING request must stop the pipeline at its next stage boundary (freeing
    # the runner slot) and resolve the record to "cancelled" — not run to completion, and not
    # let the checkpoint's stop-exception masquerade as a failure.
    started = threading.Event()
    release = threading.Event()

    def fake_pipeline(*, settings, input_path, input_mime_type, request, checkpoint):
        del settings, input_path, input_mime_type, request
        started.set()
        assert release.wait(3.0)
        checkpoint()  # cancel was requested while we "worked" -> raises PipelineCancelled
        raise AssertionError("checkpoint did not stop a cancelled run")

    monkeypatch.setattr("app.runtime.service.run_translate_image_pipeline", fake_pipeline)
    app = create_app(_settings_path(tmp_path))
    with TestClient(app) as client:
        _submit(
            client,
            {"request_id": "req_cancel", "task": "translate_image",
             "source_lang_code": "en", "target_lang_code": "nl"},
        )
        assert started.wait(3.0)
        response = client.post("/v1/requests/req_cancel/cancel")
        assert response.status_code == 200
        assert response.json()["state"] == "cancel_requested"
        release.set()
        deadline = time.monotonic() + 3.0
        body: dict[str, object] = {}
        while time.monotonic() < deadline:
            body = client.get("/v1/requests/req_cancel").json()
            if body["state"] == "cancelled":
                break
            time.sleep(0.05)
        assert body["state"] == "cancelled"
        events = client.get("/v1/completions").json()["events"]
        assert events and events[-1]["state"] == "cancelled"


def test_status(tmp_path: Path) -> None:
    app = create_app(_settings_path(tmp_path))
    with TestClient(app) as client:
        resp = client.get("/v1/status")
        assert resp.status_code == 200
        assert resp.json()["runner_slots"] == 1


def test_upload_canonicalizes_exif_orientation(tmp_path: Path, monkeypatch) -> None:
    from PIL import Image

    def fake_pipeline(*, settings, input_path, input_mime_type, request, checkpoint):
        del settings, input_path, input_mime_type, request, checkpoint
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


def test_resolve_ocr_language_routes_on_hint_script() -> None:
    settings = OcrSettings(backend="paddleocr", language="")

    assert resolve_ocr_language(settings, ["DINER", "€8,50", "Franse vissoep"]) == "en"
    assert resolve_ocr_language(settings, ["HÆTTA!", "DANGER!", "危險"]) == "ch"
    assert resolve_ocr_language(settings, ["ラーメン"]) == "ch"
    assert resolve_ocr_language(settings, ["one stray 危 glyph"]) == "en"
    assert resolve_ocr_language(settings, []) == "en"
    assert resolve_ocr_language(OcrSettings(backend="paddleocr", language="latin"), ["危險"]) == "latin"


def test_halfwidth_katakana_routes_to_the_server_pair() -> None:
    # Halfwidth katakana (common on Japanese receipts) must count as CJK for routing —
    # en_mobile mangles such an image wholesale.
    settings = OcrSettings(backend="paddleocr", language="")
    assert resolve_ocr_language(settings, ["ｶﾞｿﾘﾝ ﾚｼｰﾄ"]) == "ch"


def test_rotation_predicate_truncates_edge_lengths_like_paddlex() -> None:
    # A ~10.9 x 16.2 crop: the float ratio is 1.486 (< 1.5), but PaddleX truncates the edge
    # norms to int (16/10 = 1.6) and DOES rotate — the predicate must fire there too, or the
    # mangled read in the 1.40-1.60 band is never rescued.
    poly = [{"x": 0, "y": 0}, {"x": 10.9, "y": 0}, {"x": 10.9, "y": 16.2}, {"x": 0, "y": 16.2}]
    assert _polygon_is_rotated_tall(poly) is True


def test_upright_rescue_gates_on_short_reads_and_nonempty_result(tmp_path: Path, monkeypatch) -> None:
    # The rescue exists for one case: an isolated tall glyph PaddleX rotated and mangled.
    # A longer read (genuine vertical/tall text) keeps its rotated result; a rescued read
    # must be non-empty to replace the original.
    from PIL import Image
    from types import SimpleNamespace

    input_path = tmp_path / "input.png"
    Image.new("RGB", (200, 200), (255, 255, 255)).save(input_path)

    class FakePaddleOCR:
        paddlex_pipeline = SimpleNamespace(
            text_rec_model=lambda crop: [{"rec_text": "1", "rec_score": 0.99}]
        )

        def predict(self, input_path: str):
            return [
                {
                    # all crops are tall (10 x 40); "l" is a mangled isolated glyph, "WORD" is a
                    # genuine long tall read, "注意" a vertical CJK column whose rotated read is
                    # correct — only the first may be re-recognized
                    "rec_texts": ["l", "WORD", "注意"],
                    "rec_scores": [0.5, 0.9, 0.6],
                    "rec_boxes": [[20, 20, 30, 60], [50, 20, 60, 60], [80, 20, 90, 60]],
                    "rec_polys": [
                        [[20, 20], [30, 20], [30, 60], [20, 60]],
                        [[50, 20], [60, 20], [60, 60], [50, 60]],
                        [[80, 20], [90, 20], [90, 60], [80, 60]],
                    ],
                }
            ]

    def fake_get_paddleocr_engine(settings, language):
        return FakePaddleOCR(), threading.Lock()

    monkeypatch.setattr("app.ocr.paddleocr._get_paddleocr_engine", fake_get_paddleocr_engine)
    segments = run_paddleocr(
        OcrSettings(backend="paddleocr", language="en", min_confidence=0.35),
        input_path,
        language="en",
        merge_lines=False,  # the production entry (run_raw_ocr) runs unmerged
    )
    assert [segment.text for segment in segments] == ["1", "WORD", "注意"]


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

    def fake_get_paddleocr_engine(settings, language):
        languages.append(language)
        return FakePaddleOCR(), threading.Lock()

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


def test_resolve_translation_route_returns_configured_model_and_mode() -> None:
    # Single direct A->B path: routing just echoes the configured model + mode.
    decision = resolve_translation_route(
        settings=AppSettings(),
        translator_model="google_gemma-4-26B-A4B-it-Q4_K_M-gguf",
        translator_mode="generic",
        source_lang_code="en",
        target_lang_code="nl",
        source_text="THE SHOE WORKS IF YOU DO.",
    )
    assert decision.translator_model == "google_gemma-4-26B-A4B-it-Q4_K_M-gguf"
    assert decision.translator_mode == "generic"
    assert decision.translation_route == "configured_generic"


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


def test_retranslate_feeds_the_source_runs_hint_variant(tmp_path: Path, monkeypatch) -> None:
    # RT1: the source run fed the geometry-adjusted hint lines to translation (the default);
    # a retranslate must feed the SAME variant, or a prompt A/B silently varies the input too.
    from app.core.config import AppSettings
    from app.grouping.units import TranslationUnit
    from app.tasks.retranslate_image import run_retranslate_image_pipeline

    captured: dict[str, object] = {}

    def fake_translate_units(**kwargs):
        captured["hint_units"] = kwargs["hint_units"]
        return []

    monkeypatch.setattr("app.tasks.retranslate_image.translate_units", fake_translate_units)
    monkeypatch.setattr(
        "app.tasks.retranslate_image.render_translated_image", lambda p, u, **k: PNG_BYTES
    )
    input_path = tmp_path / "in.png"
    input_path.write_bytes(PNG_BYTES)
    unit = TranslationUnit(id=1, order=1, members=[], bbox={}, source_text="KAAS 4,50", hint_index=0)
    grouping = {
        "units": [unit.to_dict()],
        "hint_units": ["KAAS 4,50"],
        "hint_units_adjusted": ["KAAS | 4,50"],
        "hint_block_ids": [0],
        "category": "receipt",
    }
    base = {"source_lang_code": "nl", "target_lang_code": "en", "source_request_id": "src"}

    run_retranslate_image_pipeline(
        settings=AppSettings(), input_path=input_path, source_grouping=grouping, request=dict(base)
    )
    assert captured["hint_units"] == ["KAAS | 4,50"]  # adjusted: what the source run fed

    run_retranslate_image_pipeline(
        settings=AppSettings(),
        input_path=input_path,
        source_grouping=grouping,
        request={**base, "use_geometry_columns": False},
    )
    assert captured["hint_units"] == ["KAAS 4,50"]  # explicit opt-out -> raw


def test_retranslate_inherits_source_flags_and_body_overrides(tmp_path: Path, monkeypatch) -> None:
    # RT2: the retranslate payload carries the SOURCE run's flags unless the body overrides —
    # schema defaults must not silently reset them.
    captured: list[dict] = []

    def fake_translate(*, settings, input_path, input_mime_type, request, checkpoint):
        del settings, input_path, input_mime_type, request, checkpoint
        return TranslateImageResult(image=PNG_BYTES, mime_type="image/png", segments=[], metadata={}, metrics={})

    def fake_retranslate(*, settings, input_path, source_grouping, request, checkpoint):
        del settings, input_path, source_grouping, checkpoint
        captured.append(request)
        return TranslateImageResult(image=PNG_BYTES, mime_type="image/png", segments=[], metadata={}, metrics={})

    monkeypatch.setattr("app.runtime.service.run_translate_image_pipeline", fake_translate)
    monkeypatch.setattr("app.runtime.service.run_retranslate_image_pipeline", fake_retranslate)
    app = create_app(_settings_path(tmp_path))
    with TestClient(app) as client:
        _submit(
            client,
            {"request_id": "src", "task": "translate_image", "source_lang_code": "en",
             "target_lang_code": "nl", "preserve_unchanged_text": True,
             "use_geometry_columns": False, "render_size_mode": "median"},
        )
        _wait_completed(client, "src")
        grouping_dir = tmp_path / "work" / "src"
        grouping_dir.mkdir(parents=True, exist_ok=True)
        (grouping_dir / "grouping.json").write_text("{}", encoding="utf-8")

        response = client.post("/v1/requests/src/retranslate", json={"request_id": "re1"})
        assert response.status_code == 202
        _wait_completed(client, "re1")
        inherited = captured[-1]
        assert inherited["preserve_unchanged_text"] is True
        assert inherited["use_geometry_columns"] is False
        assert inherited["render_size_mode"] == "median"

        response = client.post(
            "/v1/requests/src/retranslate", json={"request_id": "re2", "use_geometry_columns": True}
        )
        assert response.status_code == 202
        _wait_completed(client, "re2")
        assert captured[-1]["use_geometry_columns"] is True  # body override wins
        assert captured[-1]["preserve_unchanged_text"] is True  # untouched flags still inherit


def test_testset_name_cannot_escape_the_testset_root(tmp_path: Path) -> None:
    app = create_app(_settings_path(tmp_path))
    with TestClient(app) as client:
        response = client.post(
            "/v1/regression/testset",
            json={"request_id": "whatever", "name": "../../outside"},
        )
        assert response.status_code == 400
        assert response.json()["code"] == "REGRESSION_BAD_NAME"


def test_invalid_request_id_is_rejected_at_the_edge(tmp_path: Path) -> None:
    app = create_app(_settings_path(tmp_path))
    with TestClient(app) as client:
        response = client.post(
            "/v1/requests",
            files={
                "request_json": (None, json.dumps({
                    "request_id": "../evil id", "task": "translate_image", "source_lang_code": "en",
                }), "application/json"),
                "image_file": ("input.png", PNG_BYTES, "image/png"),
            },
        )
        assert response.status_code == 400
        assert response.json()["code"] == "REQUEST_INVALID"


def test_oversized_upload_is_rejected(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("app.main._MAX_UPLOAD_BYTES", 10)
    app = create_app(_settings_path(tmp_path))
    with TestClient(app) as client:
        response = client.post(
            "/v1/requests",
            files={
                "request_json": (None, json.dumps({"task": "translate_image", "source_lang_code": "en"}), "application/json"),
                "image_file": ("input.png", PNG_BYTES, "image/png"),
            },
        )
        assert response.status_code == 413
        assert response.json()["code"] == "REQUEST_INPUT_TOO_LARGE"


def test_conflicting_resubmission_does_not_replace_the_stored_input(tmp_path: Path, monkeypatch) -> None:
    from PIL import Image

    def fake_pipeline(*, settings, input_path, input_mime_type, request, checkpoint):
        del settings, input_path, input_mime_type, request, checkpoint
        return TranslateImageResult(image=PNG_BYTES, mime_type="image/png", segments=[], metadata={}, metrics={})

    monkeypatch.setattr("app.runtime.service.run_translate_image_pipeline", fake_pipeline)
    other = BytesIO()
    Image.new("RGB", (2, 2), (0, 0, 0)).save(other, format="PNG")

    app = create_app(_settings_path(tmp_path))
    with TestClient(app) as client:
        _submit(client, {"request_id": "req_dup", "task": "translate_image", "source_lang_code": "en"})
        _wait_completed(client, "req_dup")
        upload_dir = tmp_path / "work" / "_uploads" / "req_dup"
        first_files = sorted(p.name for p in upload_dir.iterdir())
        assert len(first_files) == 1

        response = client.post(
            "/v1/requests",
            files={
                "request_json": (None, json.dumps({
                    "request_id": "req_dup", "task": "translate_image", "source_lang_code": "nl",
                }), "application/json"),
                "image_file": ("other.png", other.getvalue(), "image/png"),
            },
        )
        assert response.status_code == 409
        # the rejected submission's file is cleaned up; the record's own input file survives
        assert sorted(p.name for p in upload_dir.iterdir()) == first_files

        completions = client.get("/v1/completions").json()
        assert completions["instance_id"]  # per-process id for cursor resets after a restart


def test_prompt_update_is_not_an_upsert_and_preserves_handwritten_meta(tmp_path: Path) -> None:
    from app.translation.prompts import PromptEntry
    from app.translation.prompts.store import PromptStore

    app = create_app(_settings_path(tmp_path))
    with TestClient(app) as client:
        response = client.put("/v1/prompts/does-not-exist", json={"system": "x"})
        assert response.status_code == 404

    root = tmp_path / "prompts"
    entry_dir = root / "my-prompt"
    entry_dir.mkdir(parents=True)
    (entry_dir / "system.md").write_text("sys", encoding="utf-8")
    (entry_dir / "user.md").write_text("user", encoding="utf-8")
    (entry_dir / "meta.toml").write_text('title = "Handgeschreven"\ntags = ["a"]\n', encoding="utf-8")
    store = PromptStore(root)
    store.update("my-prompt", PromptEntry(id="my-prompt", system="nieuw", user="user", tags=["b"]))
    meta = (entry_dir / "meta.toml").read_text(encoding="utf-8")
    assert 'title = "Handgeschreven"' in meta  # hand-written key survives the API rewrite
    assert '"b"' in meta


def test_startup_sweeps_orphaned_work_dirs(tmp_path: Path) -> None:
    stale = tmp_path / "work" / "req_old"
    stale.mkdir(parents=True)
    (stale / "output.png").write_bytes(PNG_BYTES)

    app = create_app(_settings_path(tmp_path))
    with TestClient(app) as client:
        client.get("/v1/status")
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and stale.exists():
            time.sleep(0.05)
        assert not stale.exists()  # no in-memory record can reference it after a restart


def test_framework_validation_speaks_the_service_error_dialect(tmp_path: Path) -> None:
    # since_seq=abc used to produce FastAPI's 422 {"detail": [...]} — a second error dialect.
    app = create_app(_settings_path(tmp_path))
    with TestClient(app) as client:
        response = client.get("/v1/completions", params={"since_seq": "abc"})
        assert response.status_code == 400
        body = response.json()
        assert body["code"] == "REQUEST_INVALID"
        assert body["retryable"] is False
        assert "since_seq" in body["message"]


def test_completion_events_are_notifications_without_the_full_response(tmp_path: Path, monkeypatch) -> None:
    # Events must not pin full responses (incl. llm_calls) in memory past the record TTL;
    # a consumer polls GET /v1/requests/{id} for the payload.
    def fake_pipeline(*, settings, input_path, input_mime_type, request, checkpoint):
        del settings, input_path, input_mime_type, request, checkpoint
        return TranslateImageResult(image=PNG_BYTES, mime_type="image/png", segments=[], metadata={}, metrics={})

    monkeypatch.setattr("app.runtime.service.run_translate_image_pipeline", fake_pipeline)
    app = create_app(_settings_path(tmp_path))
    with TestClient(app) as client:
        _submit(client, {"request_id": "req_evt", "task": "translate_image", "source_lang_code": "en"})
        _wait_completed(client, "req_evt")
        events = client.get("/v1/completions").json()["events"]
        assert events and events[-1]["request_id"] == "req_evt"
        assert events[-1]["response"] is None
