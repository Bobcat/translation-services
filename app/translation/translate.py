"""Stage #6 + #7: route the translation units and translate them via llm-pool.

``resolve_translation_route`` picks the translator model/mode (#6); the units'
``source_text`` is translated through llm-pool's Responses API (#7). The request
shape depends on the routed mode: ``translategemma`` sends ``source_lang_code`` +
``target_lang_code`` and omits ``instructions`` (required by
``translategemma_template``); any other (instruction-based) mode sends a
target-language ``instructions`` prompt that also carries the image **category**.

Instruction-based modes translate **all units in ONE batch call** — a numbered list
in, a numbered list out, mapped back by number — so the model sees the whole document
(better context/coherence, fewer round-trips). Any unit missing from the batch output
falls back to a single per-unit call. ``translategemma`` (single-text template) stays
one-call-per-unit. Units with no translatable text (a bare price/number) are skipped.
"""
from __future__ import annotations

import re
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
    category: str = "",
) -> list[TranslatedUnit]:
    decision = resolve_translation_route(
        settings=settings,
        translator_model=translator_model,
        translator_mode=translator_mode,
        source_lang_code=source_lang_code,
        target_lang_code=target_lang_code,
        source_text="",
    )
    translatable = [(unit.id, str(unit.source_text or "").strip()) for unit in units if str(unit.source_text or "").strip()]

    # One batch call for instruction-based modes; translategemma can't batch.
    batched = decision.translator_mode != "translategemma" and len(translatable) > 1
    batch: dict[int, str] = (
        _translate_batch(
            settings=settings,
            model=decision.translator_model,
            items=translatable,
            target_lang_code=target_lang_code,
            category=category,
        )
        if batched
        else {}
    )

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
        translated = batch.get(unit.id)
        route = f"{decision.translation_route}_batch"
        if translated is None:
            translated = _translate_one(
                settings=settings,
                model=decision.translator_model,
                mode=decision.translator_mode,
                text=source_text,
                source_lang_code=source_lang_code,
                target_lang_code=target_lang_code,
                category=category,
            )
            route = f"{decision.translation_route}_batch_fallback" if batched else decision.translation_route
        results.append(
            TranslatedUnit(
                unit_id=unit.id,
                source_text=unit.source_text,
                translated_text=translated,
                translator_model=decision.translator_model,
                translation_route=route,
            )
        )
    return results


def _translate_batch(
    *,
    settings: AppSettings,
    model: str,
    items: list[tuple[int, str]],
    target_lang_code: str,
    category: str,
) -> dict[int, str]:
    """Translate every item in one call; return ``{unit_id: translation}`` (may be partial)."""
    if not str(model or "").strip():
        raise TranslationError("translator_model is required to translate units")
    numbered = "\n".join(f"{i}. {text}" for i, (_uid, text) in enumerate(items, start=1))
    payload: dict[str, Any] = {
        "model": model,
        "input": numbered,
        "instructions": _batch_system_prompt(target_lang_code, category),
        "stream": False,
        "decoding": {"top_k": 1, "top_p": 1, "temperature": 0.0, "repetition_penalty": 1.0, "max_tokens": 4096},
    }
    url = f"{settings.llm_pool.base_url}{_RESPONSES_PATH}"
    try:
        response = httpx.post(url, json=payload, timeout=settings.llm_pool.request_timeout_s)
        response.raise_for_status()
        data = response.json()
    except httpx.HTTPStatusError as exc:
        body = exc.response.text.strip()
        raise TranslationError(f"llm-pool /v1/responses HTTP {exc.response.status_code}: {body or exc}") from exc
    except httpx.HTTPError as exc:
        raise TranslationError(f"llm-pool /v1/responses unavailable: {exc}") from exc
    if not isinstance(data, dict):
        raise TranslationError("llm-pool /v1/responses returned a non-object response")

    out: dict[int, str] = {}
    for number, translation in _parse_numbered(str(data.get("output_text") or "")).items():
        if 1 <= number <= len(items):
            out[items[number - 1][0]] = translation
    return out


def _parse_numbered(text: str) -> dict[int, str]:
    """``"1. foo\\n2. bar"`` -> ``{1: "foo", 2: "bar"}``; continuation lines append."""
    result: dict[int, str] = {}
    current: int | None = None
    buffer: list[str] = []

    def flush() -> None:
        if current is not None:
            result[current] = " ".join(buffer).strip()

    for raw in str(text or "").splitlines():
        match = re.match(r"^\s*(\d+)[.)]\s*(.*)$", raw)
        if match:
            flush()
            current = int(match.group(1))
            buffer = [match.group(2)]
        elif current is not None and raw.strip():
            buffer.append(raw.strip())
    flush()
    return result


def _translate_one(
    *,
    settings: AppSettings,
    model: str,
    mode: str,
    text: str,
    source_lang_code: str,
    target_lang_code: str,
    category: str = "",
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
        # ignore (so omit) instructions — see llm-pool /v1/responses docs. No place to
        # pass the image category here; it only helps the instruction-based modes.
        payload["source_lang_code"] = source_lang_code
        payload["target_lang_code"] = target_lang_code
    else:
        payload["instructions"] = _system_prompt(target_lang_code, category)
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


def _system_prompt(target_lang_code: str, category: str = "") -> str:
    target = _lang_name(target_lang_code)
    # The image category (from grouping) is cheap, high-value context: it stops
    # context-free mistranslations such as "hoofdgerechten" -> "fundamental rights"
    # (instead of "main courses") on a restaurant menu.
    context = f"The text is from this image: {category.strip()}. " if str(category or "").strip() else ""
    return (
        f"You are a translation engine. {context}"
        f"Translate the user's text into {target}. Return only the translation, nothing else."
    )


def _batch_system_prompt(target_lang_code: str, category: str = "") -> str:
    target = _lang_name(target_lang_code)
    context = f"The text is from this image: {category.strip()}. " if str(category or "").strip() else ""
    return (
        f"You are a translation engine. {context}"
        f"You receive a numbered list of text snippets. Translate EACH snippet into {target}. "
        "Return ONLY a numbered list of the translations, using the SAME numbers and the SAME "
        "count, one item per line. Do not merge, split, reorder, drop or add items, and write "
        "no other text."
    )


def _lang_name(code: str) -> str:
    normalized = str(code or "").strip().lower()
    return _LANG_NAMES.get(normalized, normalized or "the target language")
