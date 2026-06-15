from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import json
import os
from pathlib import Path
from typing import Any


DEFAULT_SETTINGS_PATH = Path(__file__).resolve().parents[2] / "config" / "settings.json"
DEFAULT_LOCAL_SETTINGS_PATH = Path(__file__).resolve().parents[2] / "config" / "local.json"


@dataclass(frozen=True)
class ServiceSettings:
    host: str = "127.0.0.1"
    port: int = 8030
    log_level: str = "info"
    work_root: str = "data/requests"
    prompts_root: str = "data/prompts"


@dataclass(frozen=True)
class SchedulerSettings:
    runner_slots: int = 2
    queue_limit: int = 20
    records_max: int = 10000
    records_ttl_s: dict[str, int] = field(
        default_factory=lambda: {"completed": 900, "failed": 1800, "cancelled": 600}
    )


@dataclass(frozen=True)
class LlmPoolSettings:
    base_url: str = "http://127.0.0.1:8010"
    translator_model: str = ""
    translator_mode: str = "generic"
    grouping_model: str = ""
    request_timeout_s: float = 120.0


@dataclass(frozen=True)
class OcrSettings:
    backend: str = "paddleocr"
    # Empty = route per image on the VLM grouping hint (Han/Kana glyphs -> the
    # multilingual server models, everything else -> the en recognizer). Non-empty
    # pins one PaddleOCR language code; PaddleOCR's own lookup then picks the models.
    language: str = ""
    min_confidence: float = 0.35
    device: str = "cpu"
    ocr_version: str = "PP-OCRv5"
    text_det_limit_side_len: int = 2048
    text_det_limit_type: str = "max"
    # Explicit PaddleOCR model names; when set they override the lang-based model
    # selection (e.g. "PP-OCRv5_server_det" / "PP-OCRv5_server_rec" for the
    # multilingual server pair that also recognizes CJK).
    det_model: str = ""
    rec_model: str = ""


@dataclass(frozen=True)
class AppSettings:
    service: ServiceSettings = field(default_factory=ServiceSettings)
    scheduler: SchedulerSettings = field(default_factory=SchedulerSettings)
    llm_pool: LlmPoolSettings = field(default_factory=LlmPoolSettings)
    ocr: OcrSettings = field(default_factory=OcrSettings)


def load_settings(path: str | Path | None = None) -> AppSettings:
    settings_path = _resolve_settings_path(path)
    payload: dict[str, Any] = {}
    if settings_path.exists():
        loaded = json.loads(settings_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            payload = loaded

    local_settings_path = _resolve_local_settings_path(settings_path)
    if local_settings_path.exists():
        loaded = json.loads(local_settings_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            payload = _merge_dicts(payload, loaded)

    service_payload = _dict(payload.get("service"))
    scheduler_payload = _dict(payload.get("scheduler"))
    llm_pool_payload = _dict(payload.get("llm_pool"))
    ocr_payload = _dict(payload.get("ocr"))

    records_ttl_s = _int_dict(
        scheduler_payload.get("records_ttl_s"),
        default={"completed": 900, "failed": 1800, "cancelled": 600},
    )

    return AppSettings(
        service=ServiceSettings(
            host=str(service_payload.get("host", "127.0.0.1")),
            port=int(service_payload.get("port", 8030)),
            log_level=str(service_payload.get("log_level", "info")),
            work_root=str(service_payload.get("work_root", "data/requests") or "").strip() or "data/requests",
            prompts_root=str(service_payload.get("prompts_root", "data/prompts") or "").strip() or "data/prompts",
        ),
        scheduler=SchedulerSettings(
            runner_slots=max(1, int(scheduler_payload.get("runner_slots", 2))),
            queue_limit=max(1, int(scheduler_payload.get("queue_limit", 20))),
            records_max=max(100, int(scheduler_payload.get("records_max", 10000))),
            records_ttl_s={
                "completed": max(10, int(records_ttl_s.get("completed", 900))),
                "failed": max(10, int(records_ttl_s.get("failed", 1800))),
                "cancelled": max(10, int(records_ttl_s.get("cancelled", 600))),
            },
        ),
        llm_pool=LlmPoolSettings(
            base_url=str(llm_pool_payload.get("base_url", "http://127.0.0.1:8010") or "").rstrip("/")
            or "http://127.0.0.1:8010",
            translator_model=str(llm_pool_payload.get("translator_model", "") or "").strip(),
            translator_mode=_translator_mode(llm_pool_payload.get("translator_mode", "generic")),
            grouping_model=str(llm_pool_payload.get("grouping_model", "") or "").strip(),
            request_timeout_s=max(1.0, float(llm_pool_payload.get("request_timeout_s", 120.0))),
        ),
        ocr=OcrSettings(
            backend=str(ocr_payload.get("backend", "paddleocr") or "").strip().lower() or "paddleocr",
            language=str(ocr_payload.get("language", "") or "").strip(),
            min_confidence=min(1.0, max(0.0, float(ocr_payload.get("min_confidence", 0.35)))),
            device=str(ocr_payload.get("device", "cpu") or "").strip() or "cpu",
            ocr_version=str(ocr_payload.get("ocr_version", "PP-OCRv5") or "").strip() or "PP-OCRv5",
            text_det_limit_side_len=max(64, int(ocr_payload.get("text_det_limit_side_len", 2048))),
            text_det_limit_type=_text_det_limit_type(ocr_payload.get("text_det_limit_type", "max")),
            det_model=str(ocr_payload.get("det_model", "") or "").strip(),
            rec_model=str(ocr_payload.get("rec_model", "") or "").strip(),
        ),
    )


def _resolve_settings_path(path: str | Path | None) -> Path:
    if path is not None:
        return Path(path)
    env_value = os.environ.get("TRANSLATION_SERVICES_SETTINGS_PATH", "").strip()
    if env_value:
        return Path(env_value)
    return DEFAULT_SETTINGS_PATH


def _resolve_local_settings_path(settings_path: Path) -> Path:
    env_value = os.environ.get("TRANSLATION_SERVICES_LOCAL_SETTINGS_PATH", "").strip()
    if env_value:
        return Path(env_value)
    if settings_path == DEFAULT_SETTINGS_PATH:
        return DEFAULT_LOCAL_SETTINGS_PATH
    return settings_path.with_name("local.json")


def _merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _merge_dicts(existing, value)
        else:
            merged[key] = value
    return merged


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _text_det_limit_type(value: Any) -> str:
    parsed = str(value or "max").strip().lower()
    return parsed if parsed in {"max", "min"} else "max"


def _translator_mode(value: Any) -> str:
    parsed = str(value or "generic").strip().lower()
    if parsed not in {"translategemma", "generic"}:
        raise ValueError(f"unsupported llm_pool.translator_mode: {parsed}")
    return parsed


def _int_dict(value: Any, *, default: dict[str, int]) -> dict[str, int]:
    if not isinstance(value, dict):
        return dict(default)
    parsed = dict(default)
    for key, item in value.items():
        try:
            parsed[str(key)] = int(item)
        except (TypeError, ValueError):
            continue
    return parsed
