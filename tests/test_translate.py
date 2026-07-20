from __future__ import annotations

from app.core.config import AppSettings
from app.grouping.units import TranslationUnit
from app.grouping.units import UnitMember
from app.tasks.translate_image import _units_for_preserve_heuristic_text
from app.translation import translate as translate_module
from app.translation.prompts import PromptEntry
from app.translation.translate import _parse_blocks
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
    assert "危險" in calls[0]["input"]  # rendered into the resolved prompt's user template


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


def test_hinted_unit_missing_from_batch_falls_back_per_hint_line(monkeypatch) -> None:
    # If the structured output changes one block's line count, that hinted unit is missing from
    # the batch. It falls back to ONE call for its clean VLM hint line (never its OCR-source
    # fragment — the VLM line is the cleaner source of truth), mapped like the structured path.
    calls: list[dict] = []

    def fake_post(url, json, timeout):  # noqa: A002
        calls.append(json)
        if len(calls) == 1:  # structured: block 0 ok, block 1 wobbled (2 lines for 1)
            return _FakeResponse({"output_text": "Cardholder\n###\nextra\nline"})
        return _FakeResponse({"output_text": "clean translated line"})

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

    assert len(calls) == 2  # structured + one hint-line call
    assert "clean VLM text" in calls[1]["input"]  # the VLM line, not "garbled ocr text"
    assert out[0].translated_text == "Cardholder"
    assert out[1].translated_text == "clean translated line"
    assert out[1].translation_route.endswith("_hint_line_fallback")


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
    # generic: an instructions prompt naming the target only, no language-code fields.
    # A single unit translates under the RESOLVED prompt (the same one the structured path
    # uses), so its text arrives rendered into that prompt's user template.
    assert "into Dutch" in payload["instructions"]
    assert "English" not in payload["instructions"]  # source must NOT be in the prompt
    assert "source_lang_code" not in payload
    assert "The shoe works if you do." in payload["input"]


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


def test_custom_prompt_reaches_a_single_unit_translation(monkeypatch) -> None:
    # A one-unit image bypasses the structured path, but the caller's prompt (custom or
    # default) must still reach the model — a prompt A/B on a one-line sign was a silent no-op.
    captured: list[dict] = []

    def fake_post(url, json, timeout):  # noqa: A002
        captured.append(json)
        return _FakeResponse({"output_text": "x"})

    monkeypatch.setattr(translate_module.httpx, "post", fake_post)
    prompt = PromptEntry(
        id="custom",
        system="Fit the footprint. Translate into {{target_lang}}.",
        user="{{source_window}}",
    )
    translate_units(
        settings=AppSettings(),
        units=[_unit(1, "PUSH")],
        source_lang_code="en",
        target_lang_code="nl",
        translator_model="gemma",
        translator_mode="generic",
        prompt=prompt,
    )

    assert captured[0]["instructions"].startswith("Fit the footprint.")
    assert captured[0]["input"] == "PUSH"


def test_renumbered_batch_reply_is_rejected_not_misassigned(monkeypatch) -> None:
    # A 0-based numbered reply would map item "1." — the SECOND source's translation — onto
    # unit 1: wrong text in place. Reject the reply; the per-unit fallback then translates.
    calls: list[dict] = []

    def fake_post(url, json, timeout):  # noqa: A002
        calls.append(json)
        if len(calls) == 1:  # the numbered batch call: a renumbered (0-based) reply
            return _FakeResponse({"output_text": "0. EEN\n1. TWEE"})
        return _FakeResponse({"output_text": f"OK{len(calls)}"})

    monkeypatch.setattr(translate_module.httpx, "post", fake_post)
    out = translate_units(
        settings=AppSettings(),
        units=[_unit(1, "eerste regel"), _unit(2, "tweede regel")],
        source_lang_code="nl",
        target_lang_code="en",
        translator_model="gemma",
        translator_mode="generic",
    )

    assert [item.translated_text for item in out] == ["OK2", "OK3"]
    assert all(item.translation_route.endswith("_batch_fallback") for item in out)


def test_failed_fallback_call_degrades_one_unit_not_the_request(monkeypatch) -> None:
    # One transient HTTP failure on a per-unit fallback must not discard the whole request:
    # that unit degrades to untranslated (with a route saying why), the rest keeps its result,
    # and the failed call is recorded in the call log.
    calls: list[dict] = []

    def fake_post(url, json, timeout):  # noqa: A002
        calls.append(json)
        if len(calls) == 1:  # numbered batch: unusable reply -> rejected
            return _FakeResponse({"output_text": "no numbers here"})
        if len(calls) == 2:  # first per-unit fallback dies
            raise translate_module.httpx.ConnectError("boom")
        return _FakeResponse({"output_text": "OK"})

    monkeypatch.setattr(translate_module.httpx, "post", fake_post)
    call_log: list[dict] = []
    out = translate_units(
        settings=AppSettings(),
        units=[_unit(1, "eerste regel"), _unit(2, "tweede regel")],
        source_lang_code="nl",
        target_lang_code="en",
        translator_model="gemma",
        translator_mode="generic",
        call_log=call_log,
    )

    assert out[0].translated_text == ""
    assert out[0].translation_route.endswith("_batch_fallback_failed")
    assert out[1].translated_text == "OK"
    assert any("error" in entry for entry in call_log)  # the failed call is in llm_calls


def test_parse_blocks_keeps_amount_sign_and_ignores_rule_decoration() -> None:
    # "-25% KORTING": the '-' is the sign, not a bullet. A ---/=== line is decoration and must
    # not break a block (only the prompt's '###' separator does), or the block counts shift and
    # a good structured reply is discarded.
    out = _parse_blocks("-25% KORTING\n---\nvolgende regel\n###\nblok twee")
    assert out == [["-25% KORTING", "volgende regel"], ["blok twee"]]


def test_translategemma_structured_failure_falls_back_per_hint_line(monkeypatch) -> None:
    # translategemma has no numbered-list fallback; when its one structured window does not
    # preserve the block structure, every hinted unit must still translate — one call per hint
    # line (with the language codes, no instructions), not an empty document.
    calls: list[dict] = []

    def fake_post(url, json, timeout):  # noqa: A002
        calls.append(json)
        if len(calls) == 1:  # structured window: one block too many -> rejected
            return _FakeResponse({"output_text": "extra\n###\nA\n###\nB"})
        return _FakeResponse({"output_text": f"LINE{len(calls)}"})

    monkeypatch.setattr(translate_module.httpx, "post", fake_post)

    out = translate_units(
        settings=AppSettings(),
        units=[_unit(1, "eerste ocr", hint_index=0), _unit(2, "tweede ocr", hint_index=1)],
        source_lang_code="nl",
        target_lang_code="en",
        translator_model="translategemma",
        translator_mode="translategemma",
        hint_units=["eerste regel", "tweede regel"],
        hint_block_ids=[0, 1],
    )

    assert len(calls) == 3  # structured + one per hint line
    assert calls[1]["input"] == "eerste regel"
    assert "source_lang_code" in calls[1] and "instructions" not in calls[1]
    assert [item.translated_text for item in out] == ["LINE2", "LINE3"]
    assert all(item.translation_route.endswith("_hint_line_fallback") for item in out)


def test_units_sharing_a_hint_line_share_one_fallback_call(monkeypatch) -> None:
    # Two units bound to the same hint line (a kept '|' field split) must cost ONE line call,
    # and both map from the same translated line.
    calls: list[dict] = []

    def fake_post(url, json, timeout):  # noqa: A002
        calls.append(json)
        if len(calls) == 1:
            return _FakeResponse({"output_text": "wrong\n###\nblock\n###\ncount"})
        return _FakeResponse({"output_text": "vertaalde regel"})

    monkeypatch.setattr(translate_module.httpx, "post", fake_post)

    out = translate_units(
        settings=AppSettings(),
        units=[_unit(1, "veld a", hint_index=0), _unit(2, "veld b", hint_index=0)],
        source_lang_code="nl",
        target_lang_code="en",
        translator_model="translategemma",
        translator_mode="translategemma",
        hint_units=["veld a en veld b samen"],
        hint_block_ids=[0],
    )

    assert len(calls) == 2  # structured + ONE shared line call
    assert [item.translated_text for item in out] == ["vertaalde regel", "vertaalde regel"]


def test_failed_hint_line_call_degrades_to_skipped(monkeypatch) -> None:
    # A dead llm-pool during the line fallback keeps the old behaviour: the unit stays
    # untranslated (original pixels), the request does not fail, the failure is logged.
    calls: list[dict] = []

    def fake_post(url, json, timeout):  # noqa: A002
        calls.append(json)
        if len(calls) == 1:
            return _FakeResponse({"output_text": "wrong\n###\nblock\n###\ncount"})
        raise translate_module.httpx.ConnectError("boom")

    monkeypatch.setattr(translate_module.httpx, "post", fake_post)

    call_log: list[dict] = []
    out = translate_units(
        settings=AppSettings(),
        units=[_unit(1, "eerste ocr", hint_index=0), _unit(2, "tweede ocr", hint_index=0)],
        source_lang_code="nl",
        target_lang_code="en",
        translator_model="translategemma",
        translator_mode="translategemma",
        hint_units=["een gedeelde regel"],
        hint_block_ids=[0],
        call_log=call_log,
    )

    assert len(calls) == 2  # the failed line call is not retried for the second unit
    assert [item.translated_text for item in out] == ["", ""]
    assert all(item.translation_route == "skipped_hinted_missing" for item in out)
    assert any("error" in entry for entry in call_log)


def test_generic_structured_failure_prefers_hint_line_fallback_over_numbered(monkeypatch) -> None:
    # With hints present, a failed structured call must NOT detour through the numbered batch:
    # its items are OCR fragments, and it succeeding would shadow the better per-hint-line
    # fallback. The numbered batch stays reserved for documents with no usable hint at all.
    calls: list[dict] = []

    def fake_post(url, json, timeout):  # noqa: A002
        calls.append(json)
        if len(calls) == 1:  # structured: wrong block count -> rejected
            return _FakeResponse({"output_text": "wrong\n###\nblock\n###\ncount"})
        return _FakeResponse({"output_text": f"LINE{len(calls)}"})

    monkeypatch.setattr(translate_module.httpx, "post", fake_post)

    out = translate_units(
        settings=AppSettings(),
        units=[_unit(1, "een ocr fragment", hint_index=0), _unit(2, "twee ocr fragment", hint_index=1)],
        source_lang_code="nl",
        target_lang_code="en",
        translator_model="gemma",
        translator_mode="generic",
        hint_units=["schone regel een", "schone regel twee"],
        hint_block_ids=[0, 1],
    )

    assert len(calls) == 3  # structured + one per hint line; NO numbered batch
    assert "schone regel een" in calls[1]["input"]
    assert "1." not in calls[1]["input"]  # not a numbered list
    assert [item.translated_text for item in out] == ["LINE2", "LINE3"]
    assert all(item.translation_route.endswith("_hint_line_fallback") for item in out)


def test_translation_calls_pin_allow_remote_off(monkeypatch) -> None:
    # Document TEXT must never route to a remote backend — every translation payload pins
    # allow_remote off, matching the grouping VLM call (belt and braces over the pool default).
    calls: list[dict] = []

    def fake_post(url, json, timeout):  # noqa: A002
        calls.append(json)
        return _FakeResponse({"output_text": "x\n###\ny"})

    monkeypatch.setattr(translate_module.httpx, "post", fake_post)
    translate_units(
        settings=AppSettings(),
        units=[_unit(1, "een", hint_index=0), _unit(2, "twee", hint_index=1)],
        source_lang_code="nl",
        target_lang_code="en",
        translator_model="gemma",
        translator_mode="generic",
        hint_units=["een regel", "twee regel"],
        hint_block_ids=[0, 1],
    )
    assert calls and all(payload["allow_remote"] is False for payload in calls)


def test_truncated_structured_reply_is_rejected_not_mapped(monkeypatch) -> None:
    # The pool envelope has no finish_reason; output tokens hitting the decoding cap is the
    # truncation signal. A truncated window lost its tail — mapping it would leave tail units
    # silently untranslated, so it must go to the per-hint-line fallback instead.
    calls: list[dict] = []

    def fake_post(url, json, timeout):  # noqa: A002
        calls.append(json)
        if len(calls) == 1:  # structured reply: RIGHT shape, but flagged as capped
            return _FakeResponse({
                "output_text": "one\n###\ntwo",
                "metrics": {"engine_output_tokens": json["decoding"]["max_tokens"]},
            })
        return _FakeResponse({"output_text": f"LINE{len(calls)}"})

    monkeypatch.setattr(translate_module.httpx, "post", fake_post)
    out = translate_units(
        settings=AppSettings(),
        units=[_unit(1, "een", hint_index=0), _unit(2, "twee", hint_index=1)],
        source_lang_code="nl",
        target_lang_code="en",
        translator_model="translategemma",
        translator_mode="translategemma",
        hint_units=["een regel", "twee regel"],
        hint_block_ids=[0, 1],
    )
    assert [item.translated_text for item in out] == ["LINE2", "LINE3"]
    assert all(item.translation_route.endswith("_hint_line_fallback") for item in out)


def test_batch_line_mismatch_flags_crosstalk_not_expansion() -> None:
    from app.translation.translate import _batch_line_mismatch

    # The measured failure: a degenerate source whose batch answer was another
    # line's full sentence.
    assert _batch_line_mismatch('A " A- A-', "Succesvolle duurzaamheidsinitiatieven hangen af van de kracht van de financiële positie van een bedrijf.")
    # Legitimate short-word expansion and normal sentence growth stay clear.
    assert not _batch_line_mismatch("Nu", "Now")
    assert not _batch_line_mismatch("推", "Push to open")
    assert not _batch_line_mismatch(
        "There is a small increase in the risk.",
        "Er is een kleine toename van het risico op complicaties.",
    )


def test_island_token_mismatch_gate() -> None:
    from app.translation.translate import _island_token_mismatch

    src = "we employ ⟦M3⟧ heads and use ⟦M4⟧ per head"
    # exact set, any order: fine (the island follows its slot in the sentence)
    assert not _island_token_mismatch(src, "per kop gebruiken we ⟦M4⟧ en ⟦M3⟧ koppen")
    # lost token: a formula would silently vanish
    assert _island_token_mismatch(src, "we gebruiken ⟦M3⟧ koppen")
    # invented token: the wrong crop would be pasted
    assert _island_token_mismatch(src, "⟦M3⟧ ⟦M4⟧ ⟦M5⟧")
    # duplicated token is also a mismatch
    assert _island_token_mismatch(src, "⟦M3⟧ ⟦M3⟧ ⟦M4⟧")
    # no tokens in the source: nothing to check
    assert not _island_token_mismatch("plain prose", "gewoon proza")


def test_island_unit_translates_from_cell_text_not_hint_line(monkeypatch) -> None:
    # An island unit's tokens live in its CELL text; the hint line carries the VLM's own
    # (TeX) reading of the math and never the tokens. The unit must therefore skip the
    # structured/hint paths and translate per unit from its token-bearing source text.
    calls: list[dict] = []

    def fake_post(url, json, timeout):  # noqa: A002
        calls.append(json)
        if len(calls) == 1:  # structured batch over hint lines
            return _FakeResponse({"output_text": "Kaarthouder\n###\nwe gebruiken $h=8$ koppen"})
        return _FakeResponse({"output_text": "we gebruiken ⟦M3⟧ parallelle koppen"})

    monkeypatch.setattr(translate_module.httpx, "post", fake_post)

    out = translate_units(
        settings=AppSettings(),
        units=[
            _unit(1, "Kaarthouder", hint_index=0),
            _unit(2, "we employ ⟦M3⟧ parallel heads", hint_index=1),
        ],
        source_lang_code="en",
        target_lang_code="nl",
        translator_model="gemma",
        translator_mode="generic",
        hint_units=["Kaarthouder", "we employ $h = 8$ parallel heads"],
        hint_block_ids=[0, 1],
    )

    assert len(calls) == 2  # structured + the island unit's own per-unit call
    assert "⟦M3⟧" in calls[1]["input"]  # cell text with the token, not the hint line
    assert out[1].translated_text == "we gebruiken ⟦M3⟧ parallelle koppen"
    assert "island_mismatch" not in out[1].translation_route


def test_island_unit_with_lost_token_stays_untranslated(monkeypatch) -> None:
    calls: list[dict] = []

    def fake_post(url, json, timeout):  # noqa: A002
        calls.append(json)
        if len(calls) == 1:
            return _FakeResponse({"output_text": "Kaarthouder\n###\nirrelevant"})
        return _FakeResponse({"output_text": "vertaling zonder token"})

    monkeypatch.setattr(translate_module.httpx, "post", fake_post)

    out = translate_units(
        settings=AppSettings(),
        units=[
            _unit(1, "Kaarthouder", hint_index=0),
            _unit(2, "we employ ⟦M3⟧ parallel heads", hint_index=1),
        ],
        source_lang_code="en",
        target_lang_code="nl",
        translator_model="gemma",
        translator_mode="generic",
        hint_units=["Kaarthouder", "we employ $h = 8$ parallel heads"],
        hint_block_ids=[0, 1],
    )

    assert out[1].translated_text == ""  # gate: pixels stay
    assert out[1].translation_route.endswith("_island_mismatch")


def test_tex_leak_mismatch_gate() -> None:
    from app.translation.translate import _tex_leak_mismatch

    # The measured leak: a hint-path translation carrying the VLM's TeX reading of
    # dropped formula lines, while the unit's own cells have no math at all.
    assert _tex_leak_mismatch("plain prose", "序列 $(x_1, \\dots, x_n)$ 映射")
    assert _tex_leak_mismatch("plain prose", "de toestand $h_t$ en invoer $t$")
    assert _tex_leak_mismatch("plain prose", "waarbij \\mathbb{R}^d de ruimte is")
    # Money pairs and plain prose stay out of the net.
    assert not _tex_leak_mismatch("costs $5 and $10", "kost $5 en $10 samen")
    assert not _tex_leak_mismatch("plain", "gewone vertaling")
    # A source that itself carries the markup (a document ABOUT TeX) is exempt.
    assert not _tex_leak_mismatch("source with $(x_1, \\dots)$", "vertaling met $(x_1, \\dots)$")


def test_tex_bearing_batch_translation_degrades_to_preserve(monkeypatch) -> None:
    calls: list[dict] = []

    def fake_post(url, json, timeout):  # noqa: A002
        calls.append(json)
        return _FakeResponse({"output_text": "Kaarthouder\n###\n序列 $(x_1, \\dots, x_n)$ 映射到"})

    monkeypatch.setattr(translate_module.httpx, "post", fake_post)

    out = translate_units(
        settings=AppSettings(),
        units=[
            _unit(1, "Kaarthouder", hint_index=0),
            _unit(2, "sequence of symbol representations to another", hint_index=1),
        ],
        source_lang_code="en",
        target_lang_code="zh",
        translator_model="gemma",
        translator_mode="generic",
        hint_units=["Kaarthouder", "sequence of symbol representations $(x_1, ..., x_n)$ to another"],
        hint_block_ids=[0, 1],
    )

    assert out[1].translated_text == ""  # pixels stay: no literal TeX, no duplicate content
    assert out[1].translation_route.endswith("_tex_leak")


def test_tex_gate_drops_tex_fields_and_keeps_clean_labels(monkeypatch) -> None:
    # A complexity-table row: the label field is clean, the formula fields are TeX on BOTH
    # sides (they describe preserved math pixels, never our cells). The gate drops those
    # pairs and keeps the translated label instead of preserving the whole row.
    calls: list[dict] = []

    def fake_post(url, json, timeout):  # noqa: A002
        calls.append(json)
        return _FakeResponse({"output_text": "Kaarthouder\n###\n自注意力|$O(n^2 \\cdot d)$|$O(1)$"})

    monkeypatch.setattr(translate_module.httpx, "post", fake_post)

    out = translate_units(
        settings=AppSettings(),
        units=[
            _unit(1, "Kaarthouder", hint_index=0),
            _unit(2, "Self-Attention", hint_index=1),
        ],
        source_lang_code="en",
        target_lang_code="zh",
        translator_model="gemma",
        translator_mode="generic",
        hint_units=["Kaarthouder", "Self-Attention|$O(n^2 \\cdot d)$|$O(1)$"],
        hint_block_ids=[0, 1],
    )

    assert out[1].translated_text == "自注意力"
    assert out[1].translation_route.endswith("_tex_fields_dropped")
    assert out[1].field_translations == [("Self-Attention", "自注意力")]


def test_tex_gate_strips_decoration_but_preserves_formula_heavy_prose(monkeypatch) -> None:
    calls: list[dict] = []

    def fake_post(url, json, timeout):  # noqa: A002
        calls.append(json)
        return _FakeResponse({"output_text":
            "$^5$我们分别使用了 2.8、3.7、6.0 和 9.5 TFLOPS 的值。\n###\n"
            "序列 $(x_1, \\dots, x_n)$ 映射到 $(z_1, \\dots, z_n)$ 的隐藏层。"})

    monkeypatch.setattr(translate_module.httpx, "post", fake_post)

    out = translate_units(
        settings=AppSettings(),
        units=[
            _unit(1, "We used values of 2.8, 3.7, 6.0 and 9.5 TFLOPS.", hint_index=0),
            _unit(2, "mapping one sequence to another of equal length such as a hidden layer", hint_index=1),
        ],
        source_lang_code="en",
        target_lang_code="zh",
        translator_model="gemma",
        translator_mode="generic",
        hint_units=[
            "$^5$We used values of 2.8, 3.7, 6.0 and 9.5 TFLOPS.",
            "mapping one sequence $(x_1, ..., x_n)$ to another $(z_1, ..., z_n)$ such as a hidden layer",
        ],
        hint_block_ids=[0, 1],
    )

    # Decoration TeX (a footnote marker): stripped, the prose renders.
    assert out[0].translated_text == "我们分别使用了 2.8、3.7、6.0 和 9.5 TFLOPS 的值。"
    assert out[0].translation_route.endswith("_tex_stripped")
    # Formula-heavy: the stripped remainder would duplicate dropped-line content — preserve.
    assert out[1].translated_text == ""
    assert out[1].translation_route.endswith("_tex_leak")
