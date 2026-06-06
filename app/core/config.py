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
    translator_mode: str = "translategemma"
    literal_translator_model: str = ""
    literal_translator_mode: str = "generic"
    translation_routes: dict[str, "TranslationRouteSettings"] = field(default_factory=dict)
    request_timeout_s: float = 120.0


@dataclass(frozen=True)
class TranslationRouteSettings:
    translator_model: str = ""
    translator_mode: str = ""
    literal_translator_model: str = ""
    literal_translator_mode: str = ""


@dataclass(frozen=True)
class OcrSettings:
    backend: str = "paddleocr"
    language: str = "en"
    min_confidence: float = 0.35
    device: str = "cpu"
    ocr_version: str = "PP-OCRv5"
    use_doc_orientation_classify: bool = False
    use_doc_unwarping: bool = False
    use_textline_orientation: bool = False


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
    translation_routes = _translation_routes(llm_pool_payload.get("translation_routes"))

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
            translator_mode=_translator_mode(llm_pool_payload.get("translator_mode", "translategemma")),
            literal_translator_model=str(llm_pool_payload.get("literal_translator_model", "") or "").strip(),
            literal_translator_mode=_literal_translator_mode(llm_pool_payload.get("literal_translator_mode", "generic")),
            translation_routes=translation_routes,
            request_timeout_s=max(1.0, float(llm_pool_payload.get("request_timeout_s", 120.0))),
        ),
        ocr=OcrSettings(
            backend=str(ocr_payload.get("backend", "paddleocr") or "").strip().lower() or "paddleocr",
            language=str(ocr_payload.get("language", "en") or "").strip() or "en",
            min_confidence=min(1.0, max(0.0, float(ocr_payload.get("min_confidence", 0.35)))),
            device=str(ocr_payload.get("device", "cpu") or "").strip() or "cpu",
            ocr_version=str(ocr_payload.get("ocr_version", "PP-OCRv5") or "").strip() or "PP-OCRv5",
            use_doc_orientation_classify=bool(ocr_payload.get("use_doc_orientation_classify", False)),
            use_doc_unwarping=bool(ocr_payload.get("use_doc_unwarping", False)),
            use_textline_orientation=bool(ocr_payload.get("use_textline_orientation", False)),
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


def _translator_mode(value: Any) -> str:
    parsed = str(value or "translategemma").strip().lower()
    if parsed not in {"translategemma", "generic", "auto"}:
        raise ValueError(f"unsupported llm_pool.translator_mode: {parsed}")
    return parsed


def _literal_translator_mode(value: Any) -> str:
    parsed = str(value or "generic").strip().lower()
    if parsed != "generic":
        raise ValueError(f"unsupported llm_pool.literal_translator_mode: {parsed}")
    return parsed


def _translation_routes(value: Any) -> dict[str, TranslationRouteSettings]:
    routes_payload = _dict(value)
    routes: dict[str, TranslationRouteSettings] = {}
    for route_key, raw_route_payload in routes_payload.items():
        route_payload = _dict(raw_route_payload)
        parsed_key = _translation_route_key(route_key)
        routes[parsed_key] = TranslationRouteSettings(
            translator_model=str(
                route_payload.get("translator_model", route_payload.get("model", "")) or ""
            ).strip(),
            translator_mode=_optional_route_translator_mode(
                route_payload.get("translator_mode", route_payload.get("mode"))
            ),
            literal_translator_model=str(
                route_payload.get("literal_translator_model", route_payload.get("literal_model", "")) or ""
            ).strip(),
            literal_translator_mode=_optional_literal_translator_mode(
                route_payload.get("literal_translator_mode", route_payload.get("literal_mode"))
            ),
        )
    return routes


def _translation_route_key(value: Any) -> str:
    parsed = str(value or "").strip().lower()
    if parsed in {"", "*", "default"}:
        return "default"
    if ":" not in parsed:
        raise ValueError(f"unsupported llm_pool.translation_routes key: {parsed}")
    source, target = parsed.split(":", 1)
    source = source.strip() or "*"
    target = target.strip() or "*"
    return f"{source}:{target}"


def _optional_route_translator_mode(value: Any) -> str:
    if value is None:
        return ""
    parsed = str(value or "").strip().lower()
    if not parsed:
        return ""
    if parsed not in {"translategemma", "generic"}:
        raise ValueError(f"unsupported llm_pool.translation_routes translator_mode: {parsed}")
    return parsed


def _optional_literal_translator_mode(value: Any) -> str:
    if value is None:
        return ""
    parsed = str(value or "").strip().lower()
    if not parsed:
        return ""
    if parsed != "generic":
        raise ValueError(f"unsupported llm_pool.translation_routes literal_translator_mode: {parsed}")
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
