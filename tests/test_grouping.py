from __future__ import annotations

from app.grouping.align import _build_hint_index
from app.grouping.align import _candidate_hints
from app.grouping.align import _match_scores
from app.grouping.align import build_units_from_hint
from app.grouping.hint_parser import parse_grouping_output
from app.grouping.tokens import _tokens


def test_candidate_hints_prunes_to_exact_lines_and_falls_back_on_garble() -> None:
    hint_sets = [
        set(_tokens("Franse vissoep met venkel")),
        set(_tokens("Kaarthouder pas")),
        set(_tokens("1 KARNEMELK 1,69")),
    ]
    index = _build_hint_index(hint_sets)
    # a cleanly-read cell shares an exact token -> only the line(s) that hold it
    assert _candidate_hints({"text": "vissoep"}, index) == {0}
    # OCR garble shares no exact token -> None, so the caller full-scans and the fuzzy match in
    # _token_score ("Kaarthuder" ~ "Kaarthouder") can still bind it
    assert _candidate_hints({"text": "Kaarthuder"}, index) is None


def test_match_scores_indexed_equals_full_scan() -> None:
    hint_sets = [set(_tokens("alpha bravo")), set(_tokens("bravo charlie")), set(_tokens("delta"))]
    index = _build_hint_index(hint_sets)
    for text in ("bravo", "alpha bravo", "charlie delta", "Kaarthuder", "zzz"):
        cell = {"text": text}
        full = _match_scores(cell, hint_sets, None)
        indexed = _match_scores(cell, hint_sets, _candidate_hints(cell, index))
        assert (full.candidates, full.score, full.full) == (indexed.candidates, indexed.score, indexed.full), text


def _cells() -> list[dict]:
    return [
        {"id": 1, "text": "THE SHOE", "bbox": {"left": 10, "top": 10, "width": 200, "height": 40}},
        {"id": 2, "text": "WORKS IF", "bbox": {"left": 10, "top": 60, "width": 220, "height": 40}},
        {"id": 3, "text": "YOU DO.", "bbox": {"left": 10, "top": 110, "width": 180, "height": 40}},
        {"id": 4, "text": "nike.com", "bbox": {"left": 80, "top": 300, "width": 120, "height": 20}},
        {"id": 5, "text": "KARNEMELK", "bbox": {"left": 10, "top": 400, "width": 160, "height": 20}},
        {"id": 6, "text": "1,69", "bbox": {"left": 300, "top": 400, "width": 50, "height": 20}},
    ]


def _hint() -> list[str]:
    return ["THE SHOE WORKS IF YOU DO.", "nike.com", "KARNEMELK"]


def test_consecutive_match_becomes_one_unit() -> None:
    result = build_units_from_hint(cells=_cells(), hint_units=_hint(), model="qwen")
    flow = result.units[0]
    assert [m.cell_id for m in flow.members] == [1, 2, 3]
    assert flow.source_text == "THE SHOE WORKS IF YOU DO."
    # union bbox spans the three stacked lines
    assert flow.bbox == {"left": 10, "top": 10, "width": 220, "height": 140}


def test_single_cell_is_own_unit() -> None:
    result = build_units_from_hint(cells=_cells(), hint_units=_hint(), model="qwen")
    karnemelk = next(u for u in result.units if any(m.cell_id == 5 for m in u.members))
    assert karnemelk.source_text == "KARNEMELK"


def test_url_member_is_not_translatable() -> None:
    result = build_units_from_hint(cells=_cells(), hint_units=_hint(), model="qwen")
    nike = next(u for u in result.units if any(m.cell_id == 4 for m in u.members))
    member = nike.members[0]
    assert member.translate is False
    assert nike.source_text == ""  # nothing translatable -> empty


def test_unmatched_cell_becomes_own_unit() -> None:
    # "1,69" matches no hint block -> leftover -> its own unit, not dropped
    result = build_units_from_hint(cells=_cells(), hint_units=_hint(), model="qwen")
    price_unit = next(u for u in result.units if any(m.cell_id == 6 for m in u.members))
    assert price_unit.members[0].translate is False  # bare number


def test_every_cell_accounted_for_exactly_once() -> None:
    result = build_units_from_hint(cells=_cells(), hint_units=_hint(), model="qwen")
    assigned = [m.cell_id for unit in result.units for m in unit.members]
    assert sorted(assigned) == [1, 2, 3, 4, 5, 6]
    assert len(assigned) == len(set(assigned))
    assert result.ignored_cell_ids == []


def test_units_are_ordered_in_reading_order() -> None:
    result = build_units_from_hint(cells=_cells(), hint_units=_hint(), model="qwen")
    assert [u.order for u in result.units] == list(range(1, len(result.units) + 1))
    # first unit is the heading (top of image), price unit comes after KARNEMELK
    assert result.units[0].members[0].cell_id == 1


def test_empty_hint_makes_each_cell_its_own_unit() -> None:
    result = build_units_from_hint(cells=_cells(), hint_units=[], model="qwen")
    assert len(result.units) == len(_cells())


def test_accent_and_case_tolerant_matching() -> None:
    cells = [{"id": 1, "text": "HATTA", "bbox": {"left": 0, "top": 0, "width": 10, "height": 10}}]
    result = build_units_from_hint(cells=cells, hint_units=["hatta!"], model="qwen")
    assert [m.cell_id for m in result.units[0].members] == [1]


def test_garbled_cell_binds_its_clean_hint_line() -> None:
    # OCR adds a character ("AHNEDAARDBEI" vs hint "AHNEDAARBEI") -> the cell must still
    # bind the clean VLM line, so the structured translation lands on it instead of the
    # fallback translating the garble in isolation.
    cells = [{"id": 1, "text": "AHNEDAARDBEI", "bbox": {"left": 0, "top": 0, "width": 10, "height": 10}}]
    hints = ["BONUS | AHNEDAARBEI | | -2,00", "BONUS | ARLABIOLOGIS | | -0,44"]
    result = build_units_from_hint(cells=cells, hint_units=hints, model="qwen")
    assert result.units[0].hint_index == 0


def test_split_word_cells_bind_the_joined_hint_line() -> None:
    # OCR splits one word over two cells ("Kaar thouder" vs hint "Kaarthouder").
    cells = [
        {"id": 1, "text": "Kaar", "bbox": {"left": 0, "top": 0, "width": 10, "height": 10}},
        {"id": 2, "text": "thouder", "bbox": {"left": 12, "top": 0, "width": 10, "height": 10}},
    ]
    result = build_units_from_hint(cells=cells, hint_units=["Kopie", "Kaarthouder"], model="qwen")
    assert len(result.units) == 1
    assert result.units[0].hint_index == 1


def test_exact_match_beats_fuzzy_substring() -> None:
    # "Kaart" is a substring of "Kaarthouder": each cell must keep its own exact line.
    cells = [
        {"id": 1, "text": "Kaarthouder", "bbox": {"left": 0, "top": 0, "width": 10, "height": 10}},
        {"id": 2, "text": "Kaart", "bbox": {"left": 0, "top": 20, "width": 10, "height": 10}},
    ]
    hints = ["Kaarthouder", "Kaart | 123456xxx0000"]
    result = build_units_from_hint(cells=cells, hint_units=hints, model="qwen")
    assert [u.hint_index for u in result.units] == [0, 1]


def test_anchored_tie_break_binds_each_duplicate_line_to_its_own_hint() -> None:
    # Two dishes share the line "en frites"-style text. The cells around the ambiguous
    # one anchor the y -> hint-index map, so the tie breaks to the hint at the cell's
    # actual position even when line density skews the global linear estimate (the
    # large gap before the last cell pulls the linear estimate far off).
    def cell(cid: int, text: str, top: int) -> dict:
        return {"id": cid, "text": text, "bbox": {"left": 0, "top": top, "width": 100, "height": 8}}

    hints = [
        "Saté van kippendijen",   # 0
        "koolsla en frites",      # 1  <- the wrong candidate for the tied cell
        "HESP hotdog van Brandt", # 2
        "met kool en uitjes",     # 3
        "en frites",              # 4  <- the right one
        "Biefstuk van de grill",  # 5
    ]
    cells = [
        cell(1, "Saté van kippendijen", 0),
        cell(2, "koolsla en frites", 10),
        cell(3, "HESP hotdog van Brandt", 20),
        cell(4, "met kool en uitjes", 30),
        cell(5, "en frites", 40),          # ties hints 1 and 4
        cell(6, "Biefstuk van de grill", 500),
    ]
    result = build_units_from_hint(cells=cells, hint_units=hints, model="qwen")
    ambiguous = next(u for u in result.units if any(m.cell_id == 5 for m in u.members))
    assert ambiguous.hint_index == 4


def test_continuation_cell_sticks_to_its_element_on_a_tie() -> None:
    # "en frites" ties two dish elements. Tilt distorts axis-aligned tops (the next
    # dish's first line can sit at almost the same top), so position is a coin flip —
    # but the cell is left-aligned directly under its element's previous line, so it
    # sticks with that element.
    cells = [
        {"id": 1, "text": "AAA hotdog kool", "bbox": {"left": 100, "top": 0, "width": 280, "height": 20}},
        {"id": 2, "text": "en frites", "bbox": {"left": 100, "top": 40, "width": 90, "height": 20}},
        {"id": 3, "text": "BBB biefstuk wijn", "bbox": {"left": 110, "top": 50, "width": 290, "height": 20}},
    ]
    hints = ["AAA hotdog kool en frites", "BBB biefstuk wijn en frites"]
    result = build_units_from_hint(cells=cells, hint_units=hints, model="qwen")
    hotdog = next(u for u in result.units if any(m.cell_id == 2 for m in u.members))
    assert hotdog.hint_index == 0


def test_interleaved_leftover_does_not_split_a_unit() -> None:
    # A noise cell between two cells of one element: the element stays ONE unit (the
    # structured translation must land once), the noise is its own leftover unit.
    cells = [
        {"id": 1, "text": "Biefstuk van de grill", "bbox": {"left": 100, "top": 10, "width": 300, "height": 30}},
        {"id": 2, "text": "xqzj", "bbox": {"left": 0, "top": 30, "width": 20, "height": 20}},
        {"id": 3, "text": "rode wijn jus en frites", "bbox": {"left": 100, "top": 50, "width": 280, "height": 30}},
    ]
    hints = ["Biefstuk van de grill met rode wijn jus en frites"]
    result = build_units_from_hint(cells=cells, hint_units=hints, model="qwen")
    dish = next(u for u in result.units if u.hint_index == 0)
    assert [m.cell_id for m in dish.members] == [1, 3]
    leftover = next(u for u in result.units if u.hint_index is None)
    assert [m.cell_id for m in leftover.members] == [2]


def test_rogue_seed_is_dropped_by_chaining() -> None:
    from app.grouping.align import _chain

    # Seeds in reading order; the out-of-order hint index (a coincidental unique match)
    # must drop out, the monotone rest stays.
    seeds = [(0.0, 0), (10.0, 1), (20.0, 5), (30.0, 2), (40.0, 3)]
    anchors = _chain(seeds)
    assert anchors == [(0.0, 0), (10.0, 1), (30.0, 2), (40.0, 3)]


def test_short_noise_cell_stays_leftover() -> None:
    # A misread single-char qty cell ("T") must not fuzzy-bind a row: it stays a leftover
    # (render skips it) instead of dragging the unit's bbox into the qty column.
    cells = [{"id": 1, "text": "T", "bbox": {"left": 0, "top": 0, "width": 10, "height": 10}}]
    result = build_units_from_hint(cells=cells, hint_units=["1 | KARNEMELK | | 1,69"], model="qwen")
    assert result.units[0].hint_index is None


def test_parse_grouping_output_extracts_category_and_units() -> None:
    raw = (
        "CATEGORY: Restaurant Menu\n"
        "###\nFranse vissoep met venkel € 8,50\n"
        "###\nPâté de Campagne € 7,25\n###\n"
    )
    hint = parse_grouping_output(raw)
    assert hint.category == "Restaurant Menu"
    assert hint.units == ["Franse vissoep met venkel € 8,50", "Pâté de Campagne € 7,25"]


def test_parse_grouping_output_strips_bullets_and_separators() -> None:
    # No category line -> empty category; bullets, markdown and ### / ----- separators
    # are dropped; every other non-empty line is one unit.
    raw = "- Alpha\n* Beta\n-----\n**Voorgerechten:**\n###\n  Gamma  \n"
    hint = parse_grouping_output(raw)
    assert hint.category == ""
    assert hint.units == ["Alpha", "Beta", "Voorgerechten:", "Gamma"]


def test_parse_colon_less_bullet_item_keeps_text_and_flags_bullet() -> None:
    # The model emits a bullet item as "|@bullet|<item>" and sometimes jams it straight onto the
    # label with no colon ("b|Roboto|24pt|400|l|@bullet|Auto kapot"). The bare label run must stop
    # at "@" so the item survives in the text and the unit is flagged a bullet — not swallowed
    # whole and dropped as an empty-text standalone label.
    raw = (
        "b|Roboto|24pt|400|l|@bullet|Auto kapot, ziek\n"
        "b|Roboto|24pt|400|l|@bullet|Plots extra geld nodig\n"
    )
    hint = parse_grouping_output(raw)
    assert hint.units == ["Auto kapot, ziek", "Plots extra geld nodig"]
    assert hint.bullets == [True, True]
    assert hint.levels == ["body", "body"]


def test_parse_labeled_output_extracts_levels_and_blocks() -> None:
    # Every labeled line is its own element, so its own block.
    raw = (
        "**Image classification: Informational sign**\n"
        "\n"
        "t|DejaVu|28pt|700|c: 25 m\n"
        "\n"
        "h|DejaVu|20pt|500|l: U bent te gast.\n"
        "b|DejaVu|16pt|400|l: Zij kunnen u schade toebrengen.\n"
        "\n"
        "m|DejaVu|12pt|400|l: 050 - 313 59 01\n"
    )
    hint = parse_grouping_output(raw)
    assert hint.category == "Informational sign"
    assert hint.units == [
        "25 m", "U bent te gast.", "Zij kunnen u schade toebrengen.", "050 - 313 59 01",
    ]
    assert hint.levels == ["title", "header", "body", "footer"]
    assert hint.block_ids == [0, 1, 2, 3]


def test_parse_wrapped_label_with_text_inside_or_after() -> None:
    # The model wavers between text inside the wrapped label ("**...: text**") and after it
    # ("**...**: text") — both must parse to the same unit + level.
    raw = "**m|Mono|12pt|400|l: 23:53**\n**t|Mono|28pt|700|l**: 48 STUKS\n"
    hint = parse_grouping_output(raw)
    assert hint.units == ["23:53", "48 STUKS"]
    assert hint.levels == ["footer", "title"]


def test_parse_continuation_lines_inherit_block_level_and_block() -> None:
    # If the model wraps an element over output lines anyway, the unlabeled
    # continuation inherits level AND block; the next label starts the next element.
    raw = (
        "b|Serif|16pt|400|l: Biefstuk van de grill met rode wijn | € 18,75\n"
        "jus, sjalotten, spitskool en frites\n"
        "b|Serif|16pt|400|l: HESP classic burger met kaas, | € 14\n"
        "tomaat, augurk, uitjes en frites\n"
    )
    hint = parse_grouping_output(raw)
    assert hint.levels == ["body", "body", "body", "body"]
    assert hint.block_ids == [0, 0, 1, 1]


def test_parse_single_star_colon_inside_label_is_the_locked_format() -> None:
    # The locked-in prompt wraps the label in single stars with the ':' inside: "*label:* text"
    # (the model drops the template's surrounding quotes). Classification, element level/font,
    # a | field row kept verbatim, and a bullet must all parse cleanly.
    raw = (
        "*Image classification:* Weather application interface\n"
        "*t|Roboto|28pt|400|l:* Springfield\n"
        "*b|Roboto|16pt|400|l:* 8 jun | Vandaag | 11° / 21°\n"
        "*b|Roboto|24pt|400|l:* |@bullet| Plots extra geld nodig\n"
    )
    hint = parse_grouping_output(raw)
    assert hint.category == "Weather application interface"
    assert hint.units == ["Springfield", "8 jun | Vandaag | 11° / 21°", "Plots extra geld nodig"]
    assert hint.levels == ["title", "body", "body"]
    assert hint.bullets == [False, False, True]
    assert hint.font_families == ["Roboto", "Roboto", "Roboto"]


def test_parse_label_wrapped_in_markdown_bold() -> None:
    # The model often wraps the whole label in bold ("**t|Inter|28pt|700|c**: DINER"). It must
    # still parse, or every line falls into block 0 with no level and reflows as one group.
    raw = (
        "**t|Inter|28pt|700|c**: DINER\n"
        "**b|Inter|16pt|400|l**: Franse vissoep | € 8,50\n"
        "**h|Inter|20pt|600|l:** HOOFDGERECHTEN\n"
    )
    hint = parse_grouping_output(raw)
    assert hint.units == ["DINER", "Franse vissoep | € 8,50", "HOOFDGERECHTEN"]
    assert hint.levels == ["title", "body", "header"]
    assert hint.alignments == ["center", None, None]
    assert hint.block_ids == [0, 1, 2]


def test_parse_standalone_label_applies_to_following_lines() -> None:
    # Receipt style: the label on its own line, the text below it.
    raw = "h|Mono|14pt|600|l:\nBETAALD MET:\n\nb|Mono|12pt|400|l:\nPINNEN | 58,51\n"
    hint = parse_grouping_output(raw)
    assert hint.units == ["BETAALD MET:", "PINNEN | 58,51"]
    assert hint.levels == ["header", "body"]
    assert hint.block_ids == [0, 1]


def test_parse_alignment_suffix_in_label() -> None:
    # The trailing l/c/r field sets alignment: 'c' -> center; 'l'/'r' -> None (both anchor at
    # the line's own edge).
    raw = (
        "h|Sans|20pt|600|c: VOORGERECHTEN\n"
        "b|Sans|16pt|400|l: Franse vissoep met venkel | € 8,50\n"
        "t|Sans|28pt|700|c: DINER\n"
    )
    hint = parse_grouping_output(raw)
    assert hint.alignments == ["center", None, "center"]
    assert hint.levels == ["header", "body", "title"]
    assert hint.units[0] == "VOORGERECHTEN"


def test_parse_typography_label_without_level_code_is_stripped() -> None:
    # Some grouping models drop the leading t/h/b/m importance code and emit only the typography
    # fields ("|Roboto|16pt|400|l: ..."). The label must still be stripped (else it leaks into the
    # translated text), the font is still read, and the level falls back to None for that element.
    raw = (
        "h|Roboto|16pt|400|l: Meerdaagse vooruitzichten\n"
        "|Roboto|16pt|400|l: 8 jun | Vandaag | 11° / 21°\n"
        "|Roboto|16pt|400|l: 9 jun | Morgen | 9° / 16°\n"
    )
    hint = parse_grouping_output(raw)
    assert hint.units == [
        "Meerdaagse vooruitzichten",
        "8 jun | Vandaag | 11° / 21°",
        "9 jun | Morgen | 9° / 16°",
    ]
    assert hint.levels == ["header", None, None]
    assert hint.font_families == ["Roboto", "Roboto", "Roboto"]
    # The no-level rows are their own elements, not continuations of the header block.
    assert hint.block_ids == [0, 1, 2]


def test_parse_field_row_without_typography_label_is_kept_verbatim() -> None:
    # A real | field row carries no "<digits>pt" field, so it must not be mistaken for a label.
    raw = "b|Roboto|16pt|400|l: 8 jun | Vandaag | 11° / 21°\n"
    hint = parse_grouping_output(raw)
    assert hint.units == ["8 jun | Vandaag | 11° / 21°"]
    assert hint.levels == ["body"]


def test_parse_labeled_separator_content_is_dropped() -> None:
    # A label wrapping a ruled line ("b|...: ------") is table decoration, not a unit; the
    # receipt rows around it stay separate elements.
    raw = (
        "b|Mono|14pt|400|l: 1 | KARNEMELK | | 1,69 B\n"
        "b|Mono|14pt|400|l: ------------------------\n"
        "b|Mono|14pt|400|l: 20 | SUBTOTAAL | | 60,95\n"
    )
    hint = parse_grouping_output(raw)
    assert hint.units == ["1 | KARNEMELK | | 1,69 B", "20 | SUBTOTAAL | | 60,95"]
    assert hint.block_ids == [0, 1]


def test_units_carry_hint_level_and_block_id() -> None:
    cells = [
        {"id": 1, "text": "KARNEMELK", "bbox": {"left": 0, "top": 0, "width": 90, "height": 10}},
        {"id": 2, "text": "xyzzy", "bbox": {"left": 0, "top": 30, "width": 50, "height": 10}},
    ]
    result = build_units_from_hint(
        cells=cells,
        hint_units=["KARNEMELK", "iets anders"],
        model="qwen",
        hint_levels=["body", "footer"],
        hint_block_ids=[0, 1],
    )
    matched = next(u for u in result.units if u.hint_index == 0)
    leftover = next(u for u in result.units if u.hint_index is None)
    assert matched.level == "body" and matched.block_id == 0
    assert leftover.level is None and leftover.block_id is None
