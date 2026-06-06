from __future__ import annotations

from dataclasses import dataclass
import re

from app.core.config import AppSettings
from app.core.config import TranslationRouteSettings


@dataclass(frozen=True)
class TranslationRouteDecision:
    translation_route: str
    translator_model: str
    translator_mode: str
    route_key: str = ""


def resolve_translation_route(
    *,
    settings: AppSettings,
    translator_model: str,
    translator_mode: str,
    source_lang_code: str,
    target_lang_code: str,
    source_text: str,
) -> TranslationRouteDecision:
    mode = str(translator_mode or "").strip().lower()
    if mode != "auto":
        return TranslationRouteDecision(
            translation_route=f"configured_{mode}",
            translator_model=translator_model,
            translator_mode=mode,
        )

    route_key, route_settings = _translation_route_settings(
        settings=settings,
        source_lang_code=source_lang_code,
        target_lang_code=target_lang_code,
    )
    if _should_use_literal_route(source_text):
        literal_model = (
            route_settings.literal_translator_model
            or settings.llm_pool.literal_translator_model
        ).strip()
        if not literal_model:
            raise RuntimeError("literal_translator_model is required when translator_mode=auto uses literal routing")
        return TranslationRouteDecision(
            translation_route="literal_generic",
            translator_model=literal_model,
            translator_mode=route_settings.literal_translator_mode
            or settings.llm_pool.literal_translator_mode,
            route_key=route_key,
        )

    default_model = (route_settings.translator_model or translator_model).strip()
    default_mode = route_settings.translator_mode or "translategemma"
    return TranslationRouteDecision(
        translation_route=f"default_{default_mode}",
        translator_model=default_model,
        translator_mode=default_mode,
        route_key=route_key,
    )


def _translation_route_settings(
    *,
    settings: AppSettings,
    source_lang_code: str,
    target_lang_code: str,
) -> tuple[str, TranslationRouteSettings]:
    source = _normalize_lang_code(source_lang_code)
    target = _normalize_lang_code(target_lang_code)
    for route_key in (f"{source}:{target}", f"{source}:*", f"*:{target}", "default"):
        route_settings = settings.llm_pool.translation_routes.get(route_key)
        if route_settings is not None:
            return route_key, route_settings
    return "", TranslationRouteSettings()


def _normalize_lang_code(value: str) -> str:
    return str(value or "").strip().lower() or "*"


def _should_use_literal_route(text: str) -> bool:
    words = [word for word in re.findall(r"[A-Za-z0-9]+", str(text or "")) if word]
    if not words:
        return False
    letters = [char for char in str(text or "") if char.isalpha()]
    uppercase_letters = [char for char in letters if char.isupper()]
    return bool(letters) and 3 <= len(words) <= 8 and (len(uppercase_letters) / len(letters)) >= 0.8
