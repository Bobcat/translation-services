"""Stage #6: pick the translator for a unit.

Single direct A->B path for now — it returns the configured model + mode. This
module is kept as the routing seam: richer routing (per language-pair or
content-based) will grow back here as testing shows the need. The previous
literal-vs-default heuristic was removed.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.core.config import AppSettings


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
    # Reserved inputs for when routing grows back (per-pair / content-based).
    del settings, source_lang_code, target_lang_code, source_text
    mode = str(translator_mode or "").strip().lower()
    return TranslationRouteDecision(
        translation_route=f"configured_{mode}",
        translator_model=translator_model,
        translator_mode=mode,
    )
