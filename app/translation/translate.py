"""Stage #6 + #7: route the translation units and translate them via llm-pool.

``resolve_translation_route`` picks the translator model/mode (#6); the units'
``source_text`` is translated through llm-pool's Responses API (#7). The request
shape depends on the routed mode: ``translategemma`` sends ``source_lang_code`` +
``target_lang_code`` and omits ``instructions`` (required by
``translategemma_template``); any other (instruction-based) mode sends a
target-language ``instructions`` prompt that also carries the image **category**.

Instruction-based modes translate **all units in ONE call** so the model sees the whole
document (better context/coherence, fewer round-trips). Preferred path is **structured**:
re-translate the clean hint lines (grouped into their blocks, ``###``-separated in the
call, ``|`` fields kept) and map each translated line back onto its unit by
``hint_index`` — this keeps a multi-line unit's full-sentence context (e.g. "THE SHOE
WORKS IF YOU DO." is not split into "THE SHOE"). When no hint is available or the model
does not preserve the structure, it falls back to a **numbered list** (a numbered list
in/out, mapped back by number); any unit still missing falls back to a single per-unit
call. ``translategemma`` (single-text template) stays one-call-per-unit. Leftover units
(no hint line) ride along as a final extra block, so they are translated with document
context instead of in isolation. Units with no translatable text (a bare price/number)
or OCR noise (a pictogram read as "i") are skipped.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import httpx

from app.core.config import AppSettings
from app.grouping.align import _is_nontranslatable
from app.grouping.units import TranslationUnit
from app.translation.prompts import PromptEntry
from app.translation.prompts.templates import IMAGE_DEFAULT_ID
from app.translation.prompts import render_template
from app.translation.prompts.templates import BUILTIN_PROMPTS
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
    # For a table row (the hint line carried '|' fields): (source, translated) for each
    # translatable field, so the renderer can place each in its own cell — matching by the
    # source text, since the VLM does not always list fields in the cells' reading order.
    # None for normal (non-tabular) units.
    field_translations: list[tuple[str, str]] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "translated_text": self.translated_text,
            "translator_model": self.translator_model,
            "translation_route": self.translation_route,
            "field_translations": self.field_translations,
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
    hint_units: list[str] | None = None,
    hint_block_ids: list[int] | None = None,
    prompt: PromptEntry | None = None,
    call_log: list[dict[str, Any]] | None = None,
) -> list[TranslatedUnit]:
    decision = resolve_translation_route(
        settings=settings,
        translator_model=translator_model,
        translator_mode=translator_mode,
        source_lang_code=source_lang_code,
        target_lang_code=target_lang_code,
        source_text="",
    )
    translatable = [
        (unit.id, text)
        for unit in units
        if (text := str(unit.source_text or "").strip()) and not _is_noise(text)
    ]

    # Instruction-based modes translate in ONE call (translategemma can't batch). Preferred
    # path: STRUCTURED — re-translate the clean hint lines (blocks / newlines / | fields)
    # and map each translated line back to its unit by hint index, so a multi-line unit
    # keeps full-sentence context (e.g. "THE SHOE WORKS IF YOU DO." is not split). Falls
    # back to the numbered-list batch when no hint is available or the structure is not preserved.
    batched = decision.translator_mode != "translategemma" and len(translatable) > 1
    batch: dict[int, str] = {}
    batch_fields: dict[int, list[str]] = {}
    if batched:
        if hint_units:
            batch, batch_fields = _translate_structured(
                settings=settings,
                model=decision.translator_model,
                units=units,
                hint_units=hint_units,
                hint_block_ids=hint_block_ids,
                source_lang_code=source_lang_code,
                target_lang_code=target_lang_code,
                category=category,
                prompt=prompt or BUILTIN_PROMPTS[IMAGE_DEFAULT_ID],
                call_log=call_log,
            )
        if not batch:
            batch = _translate_batch(
                settings=settings,
                model=decision.translator_model,
                items=translatable,
                target_lang_code=target_lang_code,
                category=category,
                call_log=call_log,
            )

    results: list[TranslatedUnit] = []
    for unit in units:
        source_text = str(unit.source_text or "").strip()
        if not source_text or _is_noise(source_text):
            results.append(
                TranslatedUnit(
                    unit_id=unit.id,
                    source_text=unit.source_text,
                    translated_text="",
                    translator_model="",
                    translation_route="skipped_empty" if not source_text else "skipped_noise",
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
                call_log=call_log,
            )
            route = f"{decision.translation_route}_batch_fallback" if batched else decision.translation_route
        results.append(
            TranslatedUnit(
                unit_id=unit.id,
                source_text=unit.source_text,
                translated_text=translated,
                translator_model=decision.translator_model,
                translation_route=route,
                field_translations=batch_fields.get(unit.id),
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
    call_log: list[dict[str, Any]] | None = None,
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
    if call_log is not None:
        call_log.append({"role": "translation_main_numbered", "payload": payload, "response": data})

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
    call_log: list[dict[str, Any]] | None = None,
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
    if call_log is not None:
        call_log.append({"role": f"translation_fallback: {text[:40]!r}", "payload": payload, "response": data})
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


def _translate_structured(
    *,
    settings: AppSettings,
    model: str,
    units: list[TranslationUnit],
    hint_units: list[str],
    hint_block_ids: list[int] | None,
    source_lang_code: str,
    target_lang_code: str,
    category: str,
    prompt: PromptEntry,
    call_log: list[dict[str, Any]] | None = None,
) -> tuple[dict[int, str], dict[int, list[tuple[str, str]]]]:
    """Translate all clean hint lines in ONE call — blocks joined by ``###`` separator
    lines, newlines and ``|`` fields preserved — then map each translated line back onto
    its unit by ``hint_index``. Leftover units (no hint line) are appended as one extra
    block and mapped back by position, so they get document context instead of an
    isolated per-unit call. Returns ``{unit_id: translation}``; empty (so the caller
    falls back to the numbered-list batch) if the model did not preserve the structure."""
    if not str(model or "").strip():
        raise TranslationError("translator_model is required to translate units")
    source_blocks = _blocks_of(hint_units, hint_block_ids)
    leftovers = [
        (unit.id, text)
        for unit in units
        if unit.hint_index is None
        and (text := str(unit.source_text or "").strip())
        and not _is_noise(text)
    ]
    if leftovers:
        source_blocks = source_blocks + [[text for _unit_id, text in leftovers]]
    source_window = "\n###\n".join("\n".join(block) for block in source_blocks)
    variables = {
        "source_window": source_window,
        "source_lang": _lang_name(source_lang_code),
        "target_lang": _lang_name(target_lang_code),
        "category": str(category or "").strip() or "unknown",
    }
    payload: dict[str, Any] = {
        "model": model,
        "input": render_template(prompt.user, variables),
        "instructions": render_template(prompt.system, variables),
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
    if call_log is not None:
        call_log.append({"role": "translation_main", "payload": payload, "response": data})

    translated_blocks = _parse_blocks(str(data.get("output_text") or ""))
    if not source_blocks or len(source_blocks) != len(translated_blocks):
        return {}, {}  # block structure not preserved -> caller falls back to the numbered list
    # Align BLOCK BY BLOCK: a reflow inside one block can't shift the mapping of the others.
    # A block whose line count changed maps to None — those units fall back to a per-unit call.
    aligned: list[str | None] = []
    for src_block, dst_block in zip(source_blocks, translated_blocks):
        aligned.extend(dst_block if len(src_block) == len(dst_block) else [None] * len(src_block))
    if len(aligned) != len(hint_units) + len(leftovers):
        return {}, {}
    leftover_aligned = aligned[len(hint_units):]
    aligned = aligned[: len(hint_units)]
    out: dict[int, str] = {}
    fields: dict[int, list[tuple[str, str]]] = {}
    for unit in units:
        index = unit.hint_index
        if index is None or not (0 <= index < len(aligned)):
            continue
        line = aligned[index]
        if line is None:
            continue
        text = _translatable_segment(line)
        if text:
            out[unit.id] = text
            pairs = _field_pairs(hint_units[index], line)
            if pairs is not None:
                fields[unit.id] = pairs
    for (unit_id, _source), line in zip(leftovers, leftover_aligned):
        if line is None:
            continue
        text = _translatable_segment(line)
        if text:
            out[unit_id] = text
    return out, fields


def _blocks_of(hint_units: list[str], hint_block_ids: list[int] | None) -> list[list[str]]:
    """Group the hint lines into their blocks (consecutive equal block ids). Without
    (parallel) block ids every line is its own block — the strictest mapping."""
    if not hint_block_ids or len(hint_block_ids) != len(hint_units):
        return [[line] for line in hint_units]
    blocks: list[list[str]] = []
    previous: int | None = None
    for line, block_id in zip(hint_units, hint_block_ids):
        if not blocks or block_id != previous:
            blocks.append([])
            previous = block_id
        blocks[-1].append(line)
    return blocks


def _parse_blocks(text: str) -> list[list[str]]:
    """Parse the translated output into ``###``-separated blocks, each a list of its
    lines (same line cleaning as ``parse_grouping_output``; a leading ``CATEGORY:``
    line is dropped)."""
    blocks: list[list[str]] = []
    current: list[str] = []
    seen_category = False
    for raw in str(text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        if not seen_category and line.lower().startswith("category:"):
            seen_category = True
            continue
        compact = line.replace(" ", "")
        if len(compact) >= 3 and set(compact) <= {"#", "-", "=", "*", ":"}:  # separator -> block break
            if current:
                blocks.append(current)
                current = []
            continue
        cleaned = line.lstrip("-*#").strip().strip("*").strip()
        if cleaned:
            current.append(cleaned)
    if current:
        blocks.append(current)
    return blocks


def _is_noise(text: str) -> bool:
    """OCR junk (a pictogram read as "i", a stray "GM"/"s0") — never send it to the
    model: it wastes a call and the model may chat back ("Good morning", "I cannot
    translate…") instead of translating. Two ASCII chars cannot be meaningful standalone
    text; short CJK (危險) can be, and stays translatable."""
    stripped = str(text or "").strip()
    return len(stripped) <= 2 and stripped.isascii()


def _translatable_segment(line: str) -> str:
    """The translatable part of a translated hint line: drop the ``|`` price/number fields
    (kept as the original pixels in the render) and join the rest."""
    segments = [seg.strip() for seg in str(line or "").split("|")]
    return " ".join(seg for seg in segments if seg and not _is_nontranslatable(seg)).strip()


def _field_pairs(source_line: str, translated_line: str) -> list[tuple[str, str]] | None:
    """(source, translated) for each translatable ``|`` field of a table row, in order.

    Splits both the source hint line and its translation on ``|`` and pairs them field by
    field, keeping only the translatable fields (prices/numbers stay as original pixels).
    The source half lets the renderer match a field to its OWN cell by text — the VLM does
    not always list the fields in the cells' reading order. Returns ``None`` when the line
    carried no ``|`` fields or the source/translation field counts disagree (then the unit
    falls back to the reflow path)."""
    src = [seg.strip() for seg in str(source_line or "").split("|")]
    dst = [seg.strip() for seg in str(translated_line or "").split("|")]
    if len(src) <= 1 or len(src) != len(dst):
        return None
    pairs = [(s, t) for s, t in zip(src, dst) if t and not _is_nontranslatable(t)]
    return pairs or None


def _lang_name(code: str) -> str:
    normalized = str(code or "").strip().lower()
    return _LANG_NAMES.get(normalized, normalized or "the target language")
