from __future__ import annotations

from app.core.config import AppSettings
from app.grouping.units import TranslationUnit
from app.grouping.units import UnitMember
from app.tasks.translate_image import _units_for_preserve_heuristic_text
from app.translation import translate as translate_module
from app.translation.translate import translate_units


def _unit(unit_id: int, source_text: str, hint_index: int | None = None) -> TranslationUnit:
    return TranslationUnit(
        id=unit_id, order=unit_id, members=[], bbox={}, source_text=source_text,
        hint_index=hint_index,
    )


def test_preserve_heuristic_disabled_keeps_all_unit_members_in_source_text() -> None:
    unit = TranslationUnit(
        id=1,
        order=1,
        bbox={"left": 0, "top": 0, "width": 140, "height": 20},
        source_text="KARNEMELK",
        hint_index=0,
        members=[
            UnitMember(
                cell_id=1,
                text="1",
                translate=False,
                bbox={"left": 0, "top": 0, "width": 20, "height": 20},
                order=1,
            ),
            UnitMember(
                cell_id=2,
                text="KARNEMELK",
                translate=True,
                bbox={"left": 30, "top": 0, "width": 80, "height": 20},
                order=2,
            ),
            UnitMember(
                cell_id=3,
                text="1,69",
                translate=False,
                bbox={"left": 120, "top": 0, "width": 20, "height": 20},
                order=3,
            ),
        ],
    )

    out = _units_for_preserve_heuristic_text([unit], preserve_heuristic_text=False)

    assert out[0].source_text == "1 KARNEMELK 1,69"
    assert [member.translate for member in out[0].members] == [True, True, True]
    assert unit.source_text == "KARNEMELK"
    assert [member.translate for member in unit.members] == [False, True, False]


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


def test_structured_translation_keeps_mixed_url_line_without_fallback(monkeypatch) -> None:
    # A sentence containing a URL is still translatable as a sentence. The URL itself is preserved
    # inside the structured translation; falling back to OCR-source text is worse on tilted fine
    # print, where OCR order can be badly scrambled.
    calls: list[dict] = []

    def fake_post(url, json, timeout):  # noqa: A002
        calls.append(json)
        return _FakeResponse(
            {
                "output_text": (
                    "Smoking causes heart attacks\n###\n"
                    "Stop now! Visit www.ikstopnu.nl Or call the stop line 0800-1995 (free)"
                )
            }
        )

    monkeypatch.setattr(translate_module.httpx, "post", fake_post)

    out = translate_units(
        settings=AppSettings(),
        units=[
            _unit(1, "Roken veroorzaakt hartaanvallen", hint_index=0),
            _unit(2, "(grati op Kijk nu! stoplijn Stop de bel Of", hint_index=1),
        ],
        source_lang_code="nl",
        target_lang_code="en",
        translator_model="gemma",
        translator_mode="generic",
        hint_units=[
            "Roken veroorzaakt hartaanvallen",
            "Stop nu! Kijk op www.ikstopnu.nl Of bel de stoplijn 0800-1995 (gratis)",
        ],
        hint_block_ids=[0, 1],
    )

    assert len(calls) == 1
    assert out[1].translated_text == "Stop now! Visit www.ikstopnu.nl Or call the stop line 0800-1995 (free)"
    assert out[1].translation_route == "configured_generic_batch"


def test_translatable_segment_drops_price_tax_but_keeps_url_sentences() -> None:
    assert translate_module._translatable_segment("1,69 B") == ""
    assert translate_module._translatable_segment("4,65") == ""
    assert translate_module._translatable_segment("1,69 B", preserve_heuristic_text=False) == "1,69 B"
    assert translate_module._translatable_segment("Stop nu! Kijk op www.ikstopnu.nl") == "Stop nu! Kijk op www.ikstopnu.nl"


def test_hinted_unit_missing_from_batch_is_skipped_without_per_unit_fallback(monkeypatch) -> None:
    # If the structured output changes one block's line count, that hinted unit is missing. Do not
    # send its OCR-source text to a per-unit fallback prompt; for hinted image text the VLM line was
    # the cleaner source of truth.
    calls: list[dict] = []

    def fake_post(url, json, timeout):  # noqa: A002
        calls.append(json)
        return _FakeResponse({"output_text": "Cardholder\n###\nextra\nline"})

    monkeypatch.setattr(translate_module.httpx, "post", fake_post)

    out = translate_units(
        settings=AppSettings(),
        units=[_unit(1, "Kaarthouder", hint_index=0), _unit(2, "garbled ocr text", hint_index=1)],
        source_lang_code="nl",
        target_lang_code="en",
        translator_model="gemma",
        translator_mode="generic",
        hint_units=["Kaarthouder", "clean VLM text"],
        hint_block_ids=[0, 1],
    )

    assert len(calls) == 1
    assert out[0].translated_text == "Cardholder"
    assert out[1].translated_text == ""
    assert out[1].translation_route == "skipped_hinted_missing"


def test_structured_table_row_keeps_text_fields_and_drops_numeric_fields(monkeypatch) -> None:
    # Receipt rows arrive as structured fields. Numeric fields must not be joined
    # into the product description, but unchanged textual product fields still
    # need to render on dense receipts so neighbouring erases cannot damage them.
    calls: list[dict] = []

    def fake_post(url, json, timeout):  # noqa: A002
        calls.append(json)
        return _FakeResponse(
            {
                "output_text": (
                    "|1|SKIMMED MILK| |1.69 B|\n###\n"
                    "|1|AH YOGHURT| |2.09|"
                )
            }
        )

    monkeypatch.setattr(translate_module.httpx, "post", fake_post)

    out = translate_units(
        settings=AppSettings(),
        units=[
            _unit(1, "KARNEMELK", hint_index=0),
            _unit(2, "AH YOGHURT", hint_index=1),
        ],
        source_lang_code="nl",
        target_lang_code="en",
        translator_model="gemma",
        translator_mode="generic",
        hint_units=[
            "|1|KARNEMELK| |1,69 B|",
            "|1|AH YOGHURT| |2,09|",
        ],
        hint_block_ids=[0, 1],
    )

    assert len(calls) == 1
    assert out[0].translated_text == "SKIMMED MILK"
    assert out[0].field_translations == [("KARNEMELK", "SKIMMED MILK")]
    assert out[1].translated_text == "AH YOGHURT"
    assert out[1].field_translations == [("AH YOGHURT", "AH YOGHURT")]


def test_structured_table_row_can_skip_unchanged_text_fields(monkeypatch) -> None:
    calls: list[dict] = []

    def fake_post(url, json, timeout):  # noqa: A002
        calls.append(json)
        return _FakeResponse(
            {
                "output_text": (
                    "|1|SKIMMED MILK| |1.69 B|\n###\n"
                    "|1|AH YOGHURT| |2.09|"
                )
            }
        )

    monkeypatch.setattr(translate_module.httpx, "post", fake_post)

    out = translate_units(
        settings=AppSettings(),
        units=[
            _unit(1, "KARNEMELK", hint_index=0),
            _unit(2, "AH YOGHURT", hint_index=1),
        ],
        source_lang_code="nl",
        target_lang_code="en",
        translator_model="gemma",
        translator_mode="generic",
        hint_units=[
            "|1|KARNEMELK| |1,69 B|",
            "|1|AH YOGHURT| |2,09|",
        ],
        hint_block_ids=[0, 1],
        preserve_unchanged_text=True,
    )

    assert len(calls) == 1
    assert out[0].translated_text == "SKIMMED MILK"
    assert out[0].field_translations == [("KARNEMELK", "SKIMMED MILK")]
    assert out[1].translated_text == ""
    assert out[1].field_translations is None


def test_structured_table_row_can_skip_unchanged_text_when_model_drops_pipes(monkeypatch) -> None:
    def fake_post(url, json, timeout):  # noqa: A002
        return _FakeResponse({"output_text": "GORGONZOLA\n###\n|1|SKIMMED MILK| |1.69 B|"})

    monkeypatch.setattr(translate_module.httpx, "post", fake_post)

    out = translate_units(
        settings=AppSettings(),
        units=[
            _unit(1, "GORGONZOLA", hint_index=0),
            _unit(2, "KARNEMELK", hint_index=1),
        ],
        source_lang_code="nl",
        target_lang_code="en",
        translator_model="gemma",
        translator_mode="generic",
        hint_units=[
            "|1|GORGONZOLA| |4,65|",
            "|1|KARNEMELK| |1,69 B|",
        ],
        hint_block_ids=[0, 1],
        preserve_unchanged_text=True,
    )

    assert out[0].translated_text == ""
    assert out[0].field_translations is None
    assert out[1].translated_text == "SKIMMED MILK"
    assert out[1].field_translations == [("KARNEMELK", "SKIMMED MILK")]


def test_structured_translation_can_skip_unchanged_non_table_text(monkeypatch) -> None:
    def fake_post(url, json, timeout):  # noqa: A002
        return _FakeResponse({"output_text": "ACME Store\n###\nSKIMMED MILK"})

    monkeypatch.setattr(translate_module.httpx, "post", fake_post)

    out = translate_units(
        settings=AppSettings(),
        units=[
            _unit(1, "ACME Store", hint_index=0),
            _unit(2, "KARNEMELK", hint_index=1),
        ],
        source_lang_code="nl",
        target_lang_code="en",
        translator_model="gemma",
        translator_mode="generic",
        hint_units=["ACME Store", "KARNEMELK"],
        hint_block_ids=[0, 1],
        preserve_unchanged_text=True,
    )

    assert out[0].translated_text == ""
    assert out[1].translated_text == "SKIMMED MILK"


def test_structured_table_row_can_render_heuristic_fields_when_preserve_disabled(monkeypatch) -> None:
    def fake_post(url, json, timeout):  # noqa: A002
        return _FakeResponse({"output_text": "|1|SKIMMED MILK|1.69 B|\n###\ndummy"})

    monkeypatch.setattr(translate_module.httpx, "post", fake_post)

    out = translate_units(
        settings=AppSettings(),
        units=[_unit(1, "KARNEMELK", hint_index=0), _unit(2, "dummy", hint_index=1)],
        source_lang_code="nl",
        target_lang_code="en",
        translator_model="gemma",
        translator_mode="generic",
        hint_units=["|1|KARNEMELK|1,69 B|", "dummy"],
        hint_block_ids=[0, 1],
        preserve_heuristic_text=False,
    )

    assert out[0].translated_text == "1 SKIMMED MILK 1.69 B"
    assert out[0].field_translations == [
        ("1", "1"),
        ("KARNEMELK", "SKIMMED MILK"),
        ("1,69 B", "1.69 B"),
    ]


def test_structured_table_row_can_preserve_only_unchanged_with_heuristics_disabled(monkeypatch) -> None:
    def fake_post(url, json, timeout):  # noqa: A002
        return _FakeResponse({"output_text": "|1|GORGONZOLA|4.65|\n###\ndummy"})

    monkeypatch.setattr(translate_module.httpx, "post", fake_post)

    out = translate_units(
        settings=AppSettings(),
        units=[_unit(1, "GORGONZOLA", hint_index=0), _unit(2, "dummy", hint_index=1)],
        source_lang_code="nl",
        target_lang_code="en",
        translator_model="gemma",
        translator_mode="generic",
        hint_units=["|1|GORGONZOLA|4,65|", "dummy"],
        hint_block_ids=[0, 1],
        preserve_heuristic_text=False,
        preserve_unchanged_text=True,
    )

    assert out[0].translated_text == "4.65"
    assert out[0].field_translations == [("4,65", "4.65")]


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
