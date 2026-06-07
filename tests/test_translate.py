from __future__ import annotations

from app.core.config import AppSettings
from app.grouping.units import TranslationUnit
from app.translation import translate as translate_module
from app.translation.translate import translate_units


def _unit(unit_id: int, source_text: str) -> TranslationUnit:
    return TranslationUnit(
        id=unit_id, order=unit_id, kind="field", members=[], bbox={}, source_text=source_text
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
