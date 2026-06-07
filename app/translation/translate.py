"""Stage #6 + #7: route each translation unit and translate it via llm-pool.

For every unit produced by grouping, ``resolve_translation_route`` picks the
translator model/mode for the language pair (#6), and the unit's ``source_text``
is translated through llm-pool's Responses API (#7): ``input`` = source text,
``output_text`` = the translation. The request shape depends on the routed mode:
``translategemma`` sends ``source_lang_code`` + ``target_lang_code`` and omits
``instructions`` (required by ``translategemma_template``); any other mode sends
a target-language ``instructions`` prompt instead.

One call per unit (routing is per-unit). Units with no translatable text (a bare
price/number field) are skipped. A failed call raises and fails the job — no
fallback.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from app.core.config import AppSettings
from app.grouping.units import TranslationUnit
from app.translation.routing import resolve_translation_route


_RESPONSES_PATH = "/v1/responses"

# Language names read better than codes in the translation prompt; the engine
# uses names too. Fall back to the raw code for anything not listed.
_LANG_NAMES = {
    "en": "English", "nl": "Dutch", "de": "German", "fr": "French", "es": "Spanish",
    "it": "Italian", "pt": "Portuguese", "is": "Icelandic", "da": "Danish",
    "sv": "Swedish", "no": "Norwegian", "fi": "Finnish", "pl": "Polish",
    "cs": "Czech", "tr": "Turkish", "ru": "Russian", "zh": "Chinese",
    "ja": "Japanese", "ko": "Korean", "ar": "Arabic",
}


@dataclass(frozen=True)
class TranslatedUnit:
    unit_id: int
    source_text: str
    translated_text: str
    translator_model: str
    translation_route: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "translated_text": self.translated_text,
            "translator_model": self.translator_model,
            "translation_route": self.translation_route,
        }


class TranslationError(RuntimeError):
    """Raised when a translation call fails."""


def translate_units(
    *,
    settings: AppSettings,
    units: list[TranslationUnit],
    source_lang_code: str,
    target_lang_code: str,
    translator_model: str,
    translator_mode: str,
) -> list[TranslatedUnit]:
    results: list[TranslatedUnit] = []
    for unit in units:
        source_text = str(unit.source_text or "").strip()
        if not source_text:
            results.append(
                TranslatedUnit(
                    unit_id=unit.id,
                    source_text=unit.source_text,
                    translated_text="",
                    translator_model="",
                    translation_route="skipped_empty",
                )
            )
            continue
        decision = resolve_translation_route(
            settings=settings,
            translator_model=translator_model,
            translator_mode=translator_mode,
            source_lang_code=source_lang_code,
            target_lang_code=target_lang_code,
            source_text=source_text,
        )
        translated = _translate_one(
            settings=settings,
            model=decision.translator_model,
            mode=decision.translator_mode,
            text=source_text,
            source_lang_code=source_lang_code,
            target_lang_code=target_lang_code,
        )
        results.append(
            TranslatedUnit(
                unit_id=unit.id,
                source_text=unit.source_text,
                translated_text=translated,
                translator_model=decision.translator_model,
                translation_route=decision.translation_route,
            )
        )
    return results


def _translate_one(
    *,
    settings: AppSettings,
    model: str,
    mode: str,
    text: str,
    source_lang_code: str,
    target_lang_code: str,
) -> str:
    if not str(model or "").strip():
        raise TranslationError("translator_model is required to translate a unit")
    payload: dict[str, Any] = {
        "model": model,
        "input": text,
        "stream": False,
        "decoding": {
            "top_k": 1,
            "top_p": 1,
            "temperature": 0.0,
            "repetition_penalty": 1.0,
            "max_tokens": 512,
        },
    }
    if str(mode).strip().lower() == "translategemma":
        # translategemma_template models require source/target language codes and
        # ignore (so omit) instructions — see llm-pool /v1/responses docs.
        payload["source_lang_code"] = source_lang_code
        payload["target_lang_code"] = target_lang_code
    else:
        payload["instructions"] = _system_prompt(target_lang_code)
    url = f"{settings.llm_pool.base_url}{_RESPONSES_PATH}"
    try:
        response = httpx.post(url, json=payload, timeout=settings.llm_pool.request_timeout_s)
        response.raise_for_status()
        data = response.json()
    except httpx.HTTPStatusError as exc:
        body = exc.response.text.strip()
        raise TranslationError(
            f"llm-pool /v1/responses HTTP {exc.response.status_code}: {body or exc}"
        ) from exc
    except httpx.HTTPError as exc:
        raise TranslationError(f"llm-pool /v1/responses unavailable: {exc}") from exc

    if not isinstance(data, dict):
        raise TranslationError("llm-pool /v1/responses returned a non-object response")
    return str(data.get("output_text") or "").strip()


def _system_prompt(target_lang_code: str) -> str:
    target = _lang_name(target_lang_code)
    return (
        f"You are a translation engine. Translate the user's text into {target}. "
        "Return only the translation, nothing else."
    )


def _lang_name(code: str) -> str:
    normalized = str(code or "").strip().lower()
    return _LANG_NAMES.get(normalized, normalized or "the target language")
