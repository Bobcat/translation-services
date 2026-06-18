from __future__ import annotations

from app.core.config import AppSettings
from app.grouping.units import TranslationUnit
from app.translation import translate as translate_module
from app.translation.translate import translate_units


def _unit(unit_id: int, source_text: str, hint_index: int | None = None) -> TranslationUnit:
    return TranslationUnit(
        id=unit_id, order=unit_id, members=[], bbox={}, source_text=source_text,
        hint_index=hint_index,
    )


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:  # noqa: D401 - mimic httpx.Response
        return None

    def json(self) -> dict:
        return self._payload


def test_translategemma_mode_uses_lang_codes_no_instructions(monkeypatch) -> None:
    calls: list[dict] = []

    def fake_post(url, json, timeout):  # noqa: A002 - httpx signature uses 'json'
        calls.append(json)
        return _FakeResponse({"output_text": "NL: " + json["input"]})

    monkeypatch.setattr(translate_module.httpx, "post", fake_post)

    units = [_unit(1, "Keep your distance."), _unit(2, "   ")]
    out = translate_units(
        settings=AppSettings(),
        units=units,
        source_lang_code="en",
        target_lang_code="nl",
        translator_model="translategemma-x",
        translator_mode="translategemma",
    )

    # translatable unit got translated by the routed model
    assert out[0].translated_text == "NL: Keep your distance."
    assert out[0].translator_model == "translategemma-x"
    assert out[0].translation_route == "configured_translategemma"
    # empty-source unit is skipped (no call)
    assert out[1].translated_text == ""
    assert out[1].translation_route == "skipped_empty"
    assert len(calls) == 1

    # translategemma: language codes, NO instructions prompt
    payload = calls[0]
    assert payload["source_lang_code"] == "en"
    assert payload["target_lang_code"] == "nl"
    assert "instructions" not in payload
    assert payload["input"] == "Keep your distance."


def test_noise_unit_is_skipped_without_a_call(monkeypatch) -> None:
    # OCR junk ("GM", a pictogram read as "i") must never reach the model — it may chat
    # back ("Good morning") instead of translating. Short CJK is real text and stays.
    calls: list[dict] = []

    def fake_post(url, json, timeout):  # noqa: A002
        calls.append(json)
        return _FakeResponse({"output_text": "danger"})

    monkeypatch.setattr(translate_module.httpx, "post", fake_post)

    out = translate_units(
        settings=AppSettings(),
        units=[_unit(1, "GM"), _unit(2, "危險")],
        source_lang_code="nl",
        target_lang_code="en",
        translator_model="gemma",
        translator_mode="generic",
    )

    assert out[0].translation_route == "skipped_noise"
    assert out[0].translated_text == ""
    assert out[1].translated_text == "danger"
    assert len(calls) == 1  # only the CJK unit was sent
    assert calls[0]["input"] == "危險"


def test_structured_translation_carries_leftovers_as_extra_block(monkeypatch) -> None:
    # A leftover unit (no hint line) rides along as a final ### block in the ONE
    # structured call — document context, no isolated per-unit fallback call.
    calls: list[dict] = []

    def fake_post(url, json, timeout):  # noqa: A002
        calls.append(json)
        return _FakeResponse({"output_text": "Cardholder\n###\nCard holder copy"})

    monkeypatch.setattr(translate_module.httpx, "post", fake_post)

    out = translate_units(
        settings=AppSettings(),
        units=[_unit(1, "Kaarthouder", hint_index=0), _unit(2, "Kaar thouder")],
        source_lang_code="nl",
        target_lang_code="en",
        translator_model="gemma",
        translator_mode="generic",
        hint_units=["Kaarthouder"],
        hint_block_ids=[0],
    )

    assert len(calls) == 1
    # The default prompt frames the units under a category header; the leftover unit
    # still rides along as the final ### block of the one call.
    assert calls[0]["input"].endswith("Kaarthouder\n###\nKaar thouder")
    assert out[0].translated_text == "Cardholder"
    assert out[1].translated_text == "Card holder copy"
    assert out[1].translation_route.endswith("_batch")


def test_generic_mode_uses_target_only_instructions(monkeypatch) -> None:
    captured: list[dict] = []

    def fake_post(url, json, timeout):  # noqa: A002
        captured.append(json)
        return _FakeResponse({"output_text": "x"})

    monkeypatch.setattr(translate_module.httpx, "post", fake_post)

    translate_units(
        settings=AppSettings(),
        units=[_unit(1, "The shoe works if you do.")],
        source_lang_code="en",
        target_lang_code="nl",
        translator_model="gemma",
        translator_mode="generic",
    )

    payload = captured[0]
    # generic: an instructions prompt naming the target only, no language-code fields
    assert "into Dutch" in payload["instructions"]
    assert "English" not in payload["instructions"]  # source must NOT be in the prompt
    assert "source_lang_code" not in payload
    assert payload["input"] == "The shoe works if you do."


def test_generic_mode_names_client_target_languages_in_instructions(monkeypatch) -> None:
    target_names = {
        "af": "Afrikaans",
        "ar": "Arabic",
        "bg": "Bulgarian",
        "bn": "Bengali",
        "cs": "Czech",
        "da": "Danish",
        "de": "German",
        "el": "Greek",
        "en": "English",
        "es": "Spanish",
        "fa": "Persian",
        "fi": "Finnish",
        "fr": "French",
        "he": "Hebrew",
        "hi": "Hindi",
        "hr": "Croatian",
        "hu": "Hungarian",
        "id": "Indonesian",
        "it": "Italian",
        "ja": "Japanese",
        "ko": "Korean",
        "ms": "Malay",
        "nl": "Dutch",
        "no": "Norwegian",
        "pl": "Polish",
        "pt": "Portuguese",
        "ro": "Romanian",
        "ru": "Russian",
        "sk": "Slovak",
        "sv": "Swedish",
        "sw": "Swahili",
        "ta": "Tamil",
        "th": "Thai",
        "tl": "Tagalog",
        "tr": "Turkish",
        "uk": "Ukrainian",
        "ur": "Urdu",
        "vi": "Vietnamese",
        "zh": "Chinese",
    }

    for target_code, target_name in target_names.items():
        payloads: list[dict] = []

        def fake_post_capture(url, json, timeout):  # noqa: A002
            payloads.append(json)
            return _FakeResponse({"output_text": "x"})

        monkeypatch.setattr(translate_module.httpx, "post", fake_post_capture)

        translate_units(
            settings=AppSettings(),
            units=[_unit(1, "The shoe works if you do.")],
            source_lang_code="en",
            target_lang_code=target_code,
            translator_model="gemma",
            translator_mode="generic",
        )

        assert f"into {target_name}" in payloads[0]["instructions"]
        assert f"into {target_code}" not in payloads[0]["instructions"]
