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
in/out, mapped back by number); hinted units still missing after a batch are skipped
rather than translated from OCR garble. ``translategemma`` (single-text template)
stays one-call-per-unit. Leftover units
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
from app.grouping.preserve import _is_nontranslatable
from app.grouping.units import TranslationUnit
from app.translation.prompts import PromptEntry
from app.translation.prompts.templates import IMAGE_DEFAULT_ID
from app.translation.prompts import render_template
from app.translation.prompts.templates import BUILTIN_PROMPTS
from app.translation.routing import resolve_translation_route


_RESPONSES_PATH = "/v1/responses"
_URL_TOKEN = re.compile(
    r"(?i)\b(?:https?://|www\.)\S+|\b\S+\.(?:com|nl|org|net|io|de|fr|co|eu)\b"
)

# Language names read better than codes in the translation prompt; the engine
# uses names too. Fall back to the raw code for anything not listed.
_LANG_NAMES = {
    "af": "Afrikaans", "ar": "Arabic", "bg": "Bulgarian", "bn": "Bengali",
    "cs": "Czech", "da": "Danish", "de": "German", "el": "Greek",
    "en": "English", "es": "Spanish", "fa": "Persian", "fi": "Finnish",
    "fr": "French", "he": "Hebrew", "hi": "Hindi", "hr": "Croatian",
    "hu": "Hungarian", "id": "Indonesian", "is": "Icelandic",
    "it": "Italian", "ja": "Japanese", "ko": "Korean", "ms": "Malay",
    "nl": "Dutch", "no": "Norwegian", "pl": "Polish", "pt": "Portuguese",
    "ro": "Romanian", "ru": "Russian", "sk": "Slovak", "sv": "Swedish",
    "sw": "Swahili", "ta": "Tamil", "th": "Thai", "tl": "Tagalog",
    "tr": "Turkish", "uk": "Ukrainian", "ur": "Urdu", "vi": "Vietnamese",
    "zh": "Chinese",
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
    preserve_heuristic_text: bool = True,
    preserve_unchanged_text: bool = False,
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
    # translategemma can batch too (one ###-window call, structure preserved), so it no longer goes
    # per-unit — only the count gate applies.
    batched = len(translatable) > 1
    batch: dict[int, str] = {}
    batch_fields: dict[int, list[tuple[str, str]]] = {}
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
                translator_mode=decision.translator_mode,
                call_log=call_log,
                preserve_heuristic_text=preserve_heuristic_text,
                preserve_unchanged_text=preserve_unchanged_text,
            )
        # The numbered-list batch is the no-hint batch: one call for a document the VLM gave no
        # usable hint for (its items are OCR fragments — there is nothing better then). With
        # hints present, a failed structured call goes to the per-hint-line fallback in the
        # result loop below instead: clean full lines beat numbered fragments, and the numbered
        # batch succeeding would shadow that better path. Not for translategemma
        # (instruction-based).
        if not batch and not hint_units and decision.translator_mode != "translategemma":
            batch = _translate_batch(
                settings=settings,
                model=decision.translator_model,
                items=translatable,
                target_lang_code=target_lang_code,
                category=category,
                call_log=call_log,
            )

    # Per-hint-line fallback for hinted units the batch left untranslated (a wobbled block, a
    # rejected numbered reply, or a failed translategemma window): translate the clean VLM LINE —
    # full-sentence context, mapped exactly like the structured path — instead of leaving the
    # unit as original pixels. Never the unit's OCR text: an isolated cell fragment ("WORKS IF")
    # is untranslatable in any prompt. Cached per line: units sharing a line share the one call.
    hint_line_fallbacks: dict[int, tuple[str, list[tuple[str, str]] | None] | None] = {}

    def _translate_hint_line(index: int) -> tuple[str, list[tuple[str, str]] | None] | None:
        if index not in hint_line_fallbacks:
            try:
                line = _translate_one(
                    settings=settings,
                    model=decision.translator_model,
                    mode=decision.translator_mode,
                    text=hint_units[index],
                    source_lang_code=source_lang_code,
                    target_lang_code=target_lang_code,
                    category=category,
                    prompt=prompt or BUILTIN_PROMPTS[IMAGE_DEFAULT_ID],
                    call_log=call_log,
                ).strip()
            except TranslationError:
                line = ""  # degrade to skipped: original pixels stay, the failed call is logged
            hint_line_fallbacks[index] = (
                _mapped_hint_line(
                    hint_units[index],
                    line,
                    preserve_heuristic_text=preserve_heuristic_text,
                    preserve_unchanged_text=preserve_unchanged_text,
                )
                if line
                else None
            )
        return hint_line_fallbacks[index]

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
        # An island unit (⟦Mn⟧ tokens in its cell text) must translate from that CELL
        # text: the structured/hint paths translate the VLM's hint lines, which carry the
        # VLM's own reading of the math (TeX), never our tokens — the gate below would
        # reject every one of them. Measured on the live paper: all island units degraded
        # to preserve through the hint fallback. Per-unit translation carries the tokens.
        has_islands = bool(_ISLAND_TOKEN_RE.search(source_text))
        translated = None if has_islands else batch.get(unit.id)
        route = f"{decision.translation_route}_batch"
        if translated is not None and _batch_line_mismatch(source_text, translated):
            # Measured failure mode: on degenerate short input ('A " A- A-') the
            # batch answer carried ANOTHER line's translation, erasing the source
            # and printing a duplicated sentence in its place. Length is the tell —
            # a translation several times longer than its source is not a
            # translation of it. Reroute through the per-unit fallback below.
            translated = None
        if translated is not None and _island_token_mismatch(source_text, translated):
            translated = None  # islands lost/invented in the batch reply: retry per unit
        if translated is None:
            hint_index = unit.hint_index
            if batched and hint_index is not None and not has_islands:
                fallback = (
                    _translate_hint_line(hint_index)
                    if hint_units and 0 <= hint_index < len(hint_units)
                    else None
                )
                if fallback is None:
                    results.append(
                        TranslatedUnit(
                            unit_id=unit.id,
                            source_text=unit.source_text,
                            translated_text="",
                            translator_model="",
                            translation_route="skipped_hinted_missing",
                        )
                    )
                    continue
                translated, pairs = fallback
                if pairs:
                    batch_fields[unit.id] = pairs
                route = f"{decision.translation_route}_hint_line_fallback"
            else:
                try:
                    translated = _translate_one(
                        settings=settings,
                        model=decision.translator_model,
                        mode=decision.translator_mode,
                        text=source_text,
                        source_lang_code=source_lang_code,
                        target_lang_code=target_lang_code,
                        category=category,
                        prompt=prompt or BUILTIN_PROMPTS[IMAGE_DEFAULT_ID],
                        call_log=call_log,
                    )
                except TranslationError:
                    # One transient HTTP failure on a fallback unit must not discard the whole
                    # request (and all completed batch work) — for a BATCHED run, degrade this one
                    # unit to untranslated (original pixels stay) with a route that says why. A
                    # single-unit run has nothing else to save: stay loud.
                    if not batched:
                        raise
                    results.append(
                        TranslatedUnit(
                            unit_id=unit.id,
                            source_text=unit.source_text,
                            translated_text="",
                            translator_model="",
                            translation_route=f"{decision.translation_route}_batch_fallback_failed",
                        )
                    )
                    continue
                route = f"{decision.translation_route}_batch_fallback" if batched else decision.translation_route
        if translated and _emphasis_leak(source_text, translated):
            # Runs BEFORE the TeX gate and the unchanged-text check, so both judge the text
            # that will actually render. Field pairs are cleaned per pair (a table's "**Header**"
            # cell renders from field_translations, not from the unit text).
            cleaned = _strip_emphasis_markup(translated)
            if cleaned:
                translated = cleaned
                fields = batch_fields.get(unit.id)
                if fields:
                    batch_fields[unit.id] = [
                        (source, _strip_emphasis_markup(text) if _emphasis_leak(source, text) else text)
                        for source, text in fields
                    ]
                route = f"{route}_emphasis_stripped"
        if translated and _tex_leak_mismatch(source_text, translated):
            # The unit's own cells carry no math (formula lines drop before grouping), so a
            # TeX-bearing translation came from the hint's reading of dropped lines. Salvage
            # what is safely ours before falling back to preserve:
            # - a FIELD ROW (a complexity table): fields whose source is itself TeX describe
            #   preserved formula pixels, never our cells — drop those pairs, keep the labels;
            # - PROSE with only decoration TeX (a footnote marker): strip it when the removed
            #   share is tiny and real prose remains.
            # Everything else keeps the preserve: rendering a formula-heavy translation's
            # remainder would duplicate content whose dropped-line pixels remain.
            fields = batch_fields.get(unit.id)
            salvaged = ""
            if fields:
                clean = [(s, t) for s, t in fields if not (_tex_markup(s) or _tex_markup(t))]
                if clean and len(clean) < len(fields):
                    batch_fields[unit.id] = clean
                    salvaged = " ".join(t for _s, t in clean)
                    route = f"{route}_tex_fields_dropped"
            elif len(re.findall(r"\w+", source_text)) >= _TEX_STRIP_MIN_SOURCE_WORDS:
                stripped = _strip_tex_markup(translated)
                removed = max(0, len(translated) - len(stripped))
                if stripped and removed <= _TEX_STRIP_MAX_SHARE * len(translated):
                    salvaged = stripped
                    route = f"{route}_tex_stripped"
            if salvaged and not _tex_leak_mismatch(source_text, salvaged):
                translated = salvaged
            else:
                translated = ""
                route = f"{route}_tex_leak"
        if translated and _island_token_mismatch(source_text, translated):
            # The islands contract (islands design doc, phase 3): every ⟦Mn⟧ token of the
            # source must appear exactly once in the translation — reordering is fine (the
            # island follows its slot in the translated sentence), loss or invention is not:
            # a lost token would drop a formula, an invented one would paste the wrong crop.
            # Deterministic gate; on failure the unit stays untranslated (pixels stay).
            translated = ""
            route = f"{route}_island_mismatch"
        if (
            translated
            and preserve_unchanged_text
            and _is_effectively_unchanged(
                source_text,
                translated,
                preserve_heuristic_text=preserve_heuristic_text,
            )
        ):
            translated = ""
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


def _log_failed_call(
    call_log: list[dict[str, Any]] | None, *, role: str, payload: dict[str, Any], error: str
) -> None:
    """Record a FAILED llm-pool call before raising, so the persisted ``llm_calls`` artifact
    shows what died — not only the calls that succeeded."""
    if call_log is not None:
        call_log.append({"role": role, "payload": payload, "error": error})


def _post_responses(
    settings: AppSettings,
    payload: dict[str, Any],
    *,
    role: str,
    call_log: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """The one door to llm-pool ``/v1/responses`` for every translation call. Pins
    ``allow_remote`` off — document TEXT must never route to a remote backend, matching the
    grouping call's pin (the pool default is already False; this survives a default change).
    Logs the call (or its failure) into ``call_log`` and raises ``TranslationError`` on
    transport/HTTP errors. Kept on module-level ``httpx.post`` deliberately: it is the seam
    the tests fake."""
    payload = {**payload, "allow_remote": False}
    url = f"{settings.llm_pool.base_url}{_RESPONSES_PATH}"
    try:
        response = httpx.post(url, json=payload, timeout=settings.llm_pool.request_timeout_s)
        response.raise_for_status()
        data = response.json()
    except httpx.HTTPStatusError as exc:
        body = exc.response.text.strip()
        message = f"llm-pool /v1/responses HTTP {exc.response.status_code}: {body or exc}"
        _log_failed_call(call_log, role=role, payload=payload, error=message)
        raise TranslationError(message) from exc
    except httpx.HTTPError as exc:
        message = f"llm-pool /v1/responses unavailable: {exc}"
        _log_failed_call(call_log, role=role, payload=payload, error=message)
        raise TranslationError(message) from exc
    if not isinstance(data, dict):
        raise TranslationError("llm-pool /v1/responses returned a non-object response")
    if call_log is not None:
        call_log.append({"role": role, "payload": payload, "response": data})
    return data


def _reply_truncated(data: dict[str, Any], payload: dict[str, Any]) -> bool:
    """Whether the reply hit the decoding cap. The pool's envelope carries no finish_reason;
    ``metrics.engine_output_tokens`` reaching ``max_tokens`` is the truncation signal it does
    carry. A truncated structured/numbered reply has lost its tail blocks — resending it to a
    same-capped fallback truncates again, so the caller must reject it instead of mapping it."""
    metrics = data.get("metrics") or {}
    try:
        produced = int(metrics.get("engine_output_tokens"))
    except (TypeError, ValueError):
        return False
    cap = int((payload.get("decoding") or {}).get("max_tokens") or 0)
    return cap > 0 and produced >= cap


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
    data = _post_responses(settings, payload, role="translation_main_numbered", call_log=call_log)
    if _reply_truncated(data, payload):
        return {}  # tail items lost -> per-unit handling decides, never a partial mis-mapping

    parsed = _parse_numbered(str(data.get("output_text") or ""))
    # Accept the reply only when it numbers exactly 1..n: a renumbered (0-based), merged or
    # partial list means at least one translation would silently land on the WRONG unit —
    # wrong-text-in-place is worse than the per-unit fallback this rejection hands over to.
    if sorted(parsed) != list(range(1, len(items) + 1)):
        return {}
    return {items[number - 1][0]: translation for number, translation in parsed.items()}


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
    prompt: PromptEntry | None = None,
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
    elif prompt is not None:
        # The RESOLVED prompt — the same one the structured call uses. A single-unit image and
        # a batch-fallback unit must translate under the caller's prompt (custom or default,
        # with its every-occurrence rule), not under a divergent hardcoded one.
        variables = {
            "source_window": text,
            "source_lang": _lang_name(source_lang_code),
            "target_lang": _lang_name(target_lang_code),
            "category": str(category or "").strip() or "unknown",
            "category_instructions": _category_instructions(category),
        }
        payload["input"] = render_template(prompt.user, variables)
        payload["instructions"] = render_template(prompt.system, variables)
    else:
        payload["instructions"] = _system_prompt(target_lang_code, category)
    data = _post_responses(settings, payload, role=f"translation_fallback: {text[:40]!r}", call_log=call_log)
    if _reply_truncated(data, payload):
        return ""  # a cut-off line is not a translation; empty degrades to original pixels
    return str(data.get("output_text") or "").strip()


def _system_prompt(target_lang_code: str, category: str = "") -> str:
    target = _lang_name(target_lang_code)
    # The image category (from grouping) is cheap, high-value context: it stops
    # context-free mistranslations such as "hoofdgerechten" -> "fundamental rights"
    # (instead of "main courses") on a restaurant menu.
    context = f"The text is from this image: {category.strip()}. " if str(category or "").strip() else ""
    return (
        f"You are a translation engine. {context}"
        f"Translate the user's text into {target}. Return only the translation, nothing else. "
        "Copy any \u27e6...\u27e7 token unchanged, at the position where its content belongs."
    )


def _batch_system_prompt(target_lang_code: str, category: str = "") -> str:
    target = _lang_name(target_lang_code)
    context = f"The text is from this image: {category.strip()}. " if str(category or "").strip() else ""
    return (
        f"You are a translation engine. {context}"
        f"You receive a numbered list of text snippets. Translate EACH snippet into {target}. "
        "Return ONLY a numbered list of the translations, using the SAME numbers and the SAME "
        "count, one item per line. Copy any \u27e6...\u27e7 token unchanged, at the position "
        "where its content belongs. Do not merge, split, reorder, drop or add items, and write "
        "no other text."
    )


# Prepended to the ###-joined source when routing translategemma in the structured (batched) path.
# The template alone takes only source/target codes; this preamble gives the per-segment multilingual
# guidance, and was validated to translate the whole window cleanly while keeping the ### structure.
_TRANSLATEGEMMA_PREAMBLE = (
    "You must translate the text from an image of category: **{category}** The source language may "
    "vary throughout the text. Identify the source language of each sentence or segment and translate "
    "all translatable content into the target language. Keep proper names (places, brands). Output "
    "ONLY the translation."
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
    translator_mode: str = "",
    call_log: list[dict[str, Any]] | None = None,
    preserve_heuristic_text: bool = True,
    preserve_unchanged_text: bool = False,
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
    decoding = {"top_k": 1, "top_p": 1, "temperature": 0.0, "repetition_penalty": 1.0, "max_tokens": 4096}
    if str(translator_mode or "").strip().lower() == "translategemma":
        # translategemma_template: source text in `input` + source/target codes, NO `instructions`.
        # It CAN translate the whole ###-joined window in one call (keeps the ### structure), so the
        # batched/structured path applies here too — no per-unit calls. The preamble gives the
        # per-segment multilingual guidance the bare template lacks.
        preamble = _TRANSLATEGEMMA_PREAMBLE.format(category=str(category or "").strip() or "unknown")
        payload: dict[str, Any] = {
            "model": model,
            "input": f"{preamble}\n\n\n{source_window}",
            "source_lang_code": source_lang_code,
            "target_lang_code": target_lang_code,
            "stream": False,
            "decoding": decoding,
        }
    else:
        variables = {
            "source_window": source_window,
            "source_lang": _lang_name(source_lang_code),
            "target_lang": _lang_name(target_lang_code),
            "category": str(category or "").strip() or "unknown",
            "category_instructions": _category_instructions(category),
        }
        payload = {
            "model": model,
            "input": render_template(prompt.user, variables),
            "instructions": render_template(prompt.system, variables),
            "stream": False,
            "decoding": decoding,
        }
    data = _post_responses(settings, payload, role="translation_main", call_log=call_log)
    if _reply_truncated(data, payload):
        return {}, {}  # tail blocks lost -> the per-hint-line fallback covers the whole document

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
        mapped = _mapped_hint_line(
            hint_units[index],
            line,
            preserve_heuristic_text=preserve_heuristic_text,
            preserve_unchanged_text=preserve_unchanged_text,
        )
        if mapped is None:
            continue
        text, pairs = mapped
        out[unit.id] = text
        if pairs:
            fields[unit.id] = pairs
    for (unit_id, _source), line in zip(leftovers, leftover_aligned):
        if line is None:
            continue
        text = _translatable_segment(line, preserve_heuristic_text=preserve_heuristic_text)
        if (
            text
            and preserve_unchanged_text
            and _is_effectively_unchanged(
                _source,
                line,
                preserve_heuristic_text=preserve_heuristic_text,
            )
        ):
            out[unit_id] = ""
            continue
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
        if len(compact) >= 3 and set(compact) <= {"#", "-", "=", "*", ":"}:
            # Only a '#'-bearing rule is a block break — the prompt's separator is "###". A
            # ---/===/*** line is the model's decoration: breaking on it would shift the block
            # count and discard an otherwise good structured reply; dropping it keeps the counts.
            if "#" in compact:
                if current:
                    blocks.append(current)
                    current = []
            continue
        # Strip leading markdown only when whitespace follows: the "-" of "-25% KORTING" is the
        # amount's sign, not a bullet (same rule as the hint parser's _MARKDOWN_LEAD).
        cleaned = re.sub(r"^[-*#]+(?=\s|$)", "", line).strip().strip("*").strip()
        if cleaned:
            current.append(cleaned)
    if current:
        blocks.append(current)
    return blocks


def _mapped_hint_line(
    source_line: str,
    translated_line: str,
    *,
    preserve_heuristic_text: bool = True,
    preserve_unchanged_text: bool = False,
) -> tuple[str, list[tuple[str, str]] | None] | None:
    """The unit-facing ``(text, field_pairs)`` of ONE translated hint line — the single mapping
    both the structured path and the per-line fallback use, so a unit gets the same text for the
    same line whichever route delivered it. ``None`` when the line yields nothing usable."""
    pairs = _field_pairs(
        source_line,
        translated_line,
        preserve_heuristic_text=preserve_heuristic_text,
        preserve_unchanged_text=preserve_unchanged_text,
    )
    if pairs is not None:
        text = " ".join(translated for _source, translated in pairs).strip()
        return text, (pairs if pairs else None)
    text = _translatable_segment(translated_line, preserve_heuristic_text=preserve_heuristic_text)
    if (
        text
        and preserve_unchanged_text
        and _is_effectively_unchanged(
            source_line,
            translated_line,
            preserve_heuristic_text=preserve_heuristic_text,
        )
    ):
        return "", None
    if text:
        return text, None
    return None


_ISLAND_TOKEN_RE = re.compile(r"\u27e6M\d+\u27e7")  # ⟦Mn⟧ inline-island token

# TeX markup in a TRANSLATION whose source has none is a model artifact, never content:
# the VLM's grouping hint transcribes formula images as LaTeX, and a unit translated
# through a hint path prints that markup as literal running text (measured: a page whose
# formula-dominated lines were dropped rendered "$(x_1, \dots, x_n)$" mid-sentence).
# A command (\dots, \mathbb) is the strong signal; a $...$ pair counts only when its
# content looks like math (script/brace/backslash chars, or one short atom like "h_t"),
# so money amounts ("$5 en $10 samen") stay out of the net.
_TEX_COMMAND_RE = re.compile(r"\\[a-zA-Z]{2,}")
_TEX_DOLLAR_RE = re.compile(r"\$([^$]{1,40})\$")


def _tex_markup(text: str) -> bool:
    if _TEX_COMMAND_RE.search(text):
        return True
    return any(
        re.search(r"[_^{}\\]", content) or re.fullmatch(r"[A-Za-z0-9(),./]{1,6}", content.strip())
        for content in _TEX_DOLLAR_RE.findall(text)
    )


def _tex_leak_mismatch(source_text: str, translated: str) -> bool:
    return _tex_markup(str(translated)) and not _tex_markup(str(source_text))


# Salvage bounds for a TeX-bearing translation (islands design doc, tex gate). Decoration
# TeX (a footnote marker "$^5$") is a tiny share of the translation and strips cleanly; a
# formula-heavy translation covers dropped math lines whose pixels remain, and rendering
# its stripped remainder would duplicate that content — those keep the preserve.
_TEX_STRIP_MAX_SHARE = 0.12
_TEX_STRIP_MIN_SOURCE_WORDS = 3


def _strip_tex_markup(translated: str) -> str:
    """``translated`` without its TeX spans: math-like $...$ pairs and bare commands are
    removed, whitespace collapsed and space-before-punctuation tidied."""
    out = str(translated)

    def _drop_pair(match):
        content = match.group(1)
        if re.search(r"[_^{}\\]", content) or re.fullmatch(r"[A-Za-z0-9(),./]{1,6}", content.strip()):
            return " "
        return match.group(0)

    out = _TEX_DOLLAR_RE.sub(_drop_pair, out)
    out = _TEX_COMMAND_RE.sub(" ", out)
    out = re.sub(r"\s+([,.;:!?)\]])", r"\1", re.sub(r"\s+", " ", out)).strip()
    return out


# Markdown emphasis in a TRANSLATION whose source has none is the model's formatting, never
# content: it marks a run-in heading or a phrase it read as emphasised, and the renderer prints
# the asterisks as literal glyphs. Two measured forms: a PAIR around a phrase ("(*Bahdanau et
# al., 2015*)") and an orphaned CLOSER ("Residual Dropout** We apply ...") — the structured
# parser strips markup at the line EDGE, so the opening pair of a run-in heading disappears and
# its closer is left behind mid-line. Gated on the source carrying no asterisk at all, which is
# what keeps a real footnote marker ("Ashish Vaswani*") out of the net.
# Underscores are deliberately NOT emphasis here: a translation's "d_model"/"P_drop" are
# subscript identifiers (measured on a table header whose source had none), and folding them
# would corrupt the symbol, not clean it.
_EMPHASIS_PAIR_RE = re.compile(r"\*{1,2}(?=\S)(.+?)(?<=\S)\*{1,2}")
# A stray marker flanks a WORD on one side only: "Dropout**", "*Bahdanau". A "*" with word
# characters on BOTH sides is not markdown emphasis (an "a*b" product) and stays.
_EMPHASIS_STRAY_RE = re.compile(r"(?<=\w)\*{1,2}(?!\w)|(?<!\w)\*{1,2}(?=\w)")


def _emphasis_leak(source_text: str, translated: str) -> bool:
    return "*" in str(translated) and "*" not in str(source_text)


def _strip_emphasis_markup(translated: str) -> str:
    """``translated`` without its markdown emphasis markers, the emphasised words kept."""
    out = _EMPHASIS_PAIR_RE.sub(r"\1", str(translated))
    out = _EMPHASIS_STRAY_RE.sub("", out)
    return re.sub(r"\s{2,}", " ", out).strip()


def _island_token_mismatch(source_text: str, translated: str) -> bool:
    """The translation must carry exactly the source's ⟦Mn⟧ island tokens (any
    order). No tokens in the source = nothing to check."""
    source_tokens = sorted(_ISLAND_TOKEN_RE.findall(str(source_text)))
    if not source_tokens:
        return False
    return sorted(_ISLAND_TOKEN_RE.findall(str(translated))) != source_tokens


def _batch_line_mismatch(source_text: str, translated: str) -> bool:
    """A batch line whose translation is several times longer than its source is
    another line's answer (LLM cross-talk on degenerate input), not a translation.
    The floor keeps legitimate short-word expansion ("Nu" -> "Now", CJK -> Latin)
    out of the net; real expansion between languages stays well under 4x."""
    return len(str(translated).strip()) > max(20, 4 * len(str(source_text).strip()))


def _is_noise(text: str) -> bool:
    """OCR junk (a pictogram read as "i", a stray "GM"/"s0") — never send it to the
    model: it wastes a call and the model may chat back ("Good morning", "I cannot
    translate…") instead of translating. Two ASCII chars cannot be meaningful standalone
    text; short CJK (危險) can be, and stays translatable."""
    stripped = str(text or "").strip()
    return len(stripped) <= 2 and stripped.isascii()


def _translatable_segment(line: str, *, preserve_heuristic_text: bool = True) -> str:
    """The translatable part of a translated hint line: drop the ``|`` price/number fields
    (kept as the original pixels in the render) and join the rest."""
    segments = [seg.strip() for seg in str(line or "").split("|")]
    if not preserve_heuristic_text:
        return " ".join(seg for seg in segments if seg).strip()
    return " ".join(seg for seg in segments if _segment_has_translatable_text(seg)).strip()


def _segment_has_translatable_text(segment: str) -> bool:
    """True for ordinary text, including text that also contains a URL/number.

    A whole URL/code/price field stays non-translatable, but a sentence like
    "Visit www.example.nl" must keep its structured translation instead of falling
    through to OCR-source fallback.
    """
    text = str(segment or "").strip()
    if not text:
        return False
    without_urls = _URL_TOKEN.sub(" ", text).strip()
    return bool(without_urls) and not _is_nontranslatable(without_urls)


def _field_pairs(
    source_line: str,
    translated_line: str,
    *,
    preserve_heuristic_text: bool = True,
    preserve_unchanged_text: bool = False,
) -> list[tuple[str, str]] | None:
    """(source, translated) for each translatable ``|`` field of a table row, in order.

    Splits both the source hint line and its translation on ``|`` and pairs them field by
    field, keeping source-side translatable fields. Numeric/code fields stay out of the
    joined text flow; textual fields still render even when the model left them unchanged
    (brand/product names on dense receipts).
    The source half lets the renderer match a field to its OWN cell by text — the VLM does
    not always list the fields in the cells' reading order. Returns ``None`` when the line
    carried no ``|`` fields or the source/translation field counts disagree (then the unit
    falls back to the reflow path)."""
    src = [seg.strip() for seg in str(source_line or "").split("|")]
    dst = [seg.strip() for seg in str(translated_line or "").split("|")]
    if len(src) <= 1 or len(src) != len(dst):
        return None
    return [
        (s, t)
        for s, t in zip(src, dst)
        if t
        and (not preserve_heuristic_text or not _is_nontranslatable(s))
        and (not preserve_unchanged_text or _field_text_key(s) != _field_text_key(t))
    ]


def _is_effectively_unchanged(
    source: str,
    translated: str,
    *,
    preserve_heuristic_text: bool = True,
) -> bool:
    source_text = _translatable_segment(
        source,
        preserve_heuristic_text=preserve_heuristic_text,
    ) or str(source or "").strip()
    translated_text = _translatable_segment(
        translated,
        preserve_heuristic_text=preserve_heuristic_text,
    ) or str(translated or "").strip()
    return bool(source_text) and _field_text_key(source_text) == _field_text_key(translated_text)


def _field_text_key(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _lang_name(code: str) -> str:
    normalized = str(code or "").strip().lower()
    return _LANG_NAMES.get(normalized, normalized or "the target language")


# Extra, category-specific guidance injected into the prompt's ``{{category_instructions}}`` slot.
# The category is the VLM's free-text classification (e.g. "restaurant menu", "Restaurant menu page"),
# so match on a discriminating keyword, not an exact string. First matching rule wins; add rules here
# as categories need them.
_CATEGORY_RULES: tuple[tuple[str, str], ...] = ()  # empty for now — no category hint; add rules later


def _category_instructions(category: str) -> str:
    """An extra instruction line for the VLM's free-text ``category``, injected into the prompt's
    ``{{category_instructions}}`` slot, or ``""`` when no rule matches. Injected as a plain line in
    the prompt flow (no ``# Additional instructions`` heading — a markdown section measurably lowered
    translation reliability), so an empty result leaves the core prompt exactly as-is."""
    cat = str(category or "").strip().lower()
    body = next((text for keyword, text in _CATEGORY_RULES if keyword in cat), "")
    return f"{body}\n" if body else ""
