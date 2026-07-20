from __future__ import annotations

from app.grouping.align.matching import _build_hint_index
from app.grouping.align.matching import _candidate_hints
from app.grouping.align.positions import _line_anchor
from app.grouping.align.matching import _match_scores
from app.grouping.align.claims import _resolve_claim_clusters
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
        assert (full.candidates, full.score, full.full, full.full_alpha) == (
            indexed.candidates,
            indexed.score,
            indexed.full,
            indexed.full_alpha,
        ), text


def test_mixed_garble_cell_falls_back_to_the_fuzzy_full_scan() -> None:
    # One clean token ("pas") sits in the WRONG line, the rest is OCR garble of the right line.
    # The exact index alone would score only line 0 (1/3 < threshold) and drop the cell as a
    # leftover; the below-threshold full-scan fallback lets the fuzzy match bind line 1.
    hint_sets = [set(_tokens("totaal pas")), set(_tokens("Kaarthouder betaling"))]
    index = _build_hint_index(hint_sets)
    cell = {"text": "Kaarthuder betallng pas"}
    assert _candidate_hints(cell, index) == {0}
    match = _match_scores(cell, hint_sets, _candidate_hints(cell, index))
    assert match.candidates == [1]
    assert match.score >= 0.4


def test_garbled_continuation_claim_merges_instead_of_dropping() -> None:
    # Claim [1] bound its line purely by fuzzy garble ("kortlng" ~ "korting"): exact-token dedup
    # would see it as token-free and drop it — its original pixels would then double next to the
    # translated line. Counting fuzzy-covered line tokens lets it merge as new-content continuation.
    cells = [
        {"id": 1, "text": "AHNEDAARBEI extra", "bbox": {"left": 0, "top": 0, "width": 200, "height": 20}},
        {"id": 2, "text": "kortlng", "bbox": {"left": 205, "top": 0, "width": 70, "height": 20}},
    ]
    kept, demoted, dropped = _resolve_claim_clusters(0, [[0], [1]], cells, ["AHNEDAARBEI extra korting"])
    assert dropped == [] and demoted == []
    assert sorted(kept[0]) == [0, 1]


def test_redundant_garbled_stray_still_drops() -> None:
    # A second garbled read of an ALREADY covered word adds nothing — it stays dropped
    # (the dedup's whole point); fuzzy coverage must not turn strays into content.
    cells = [
        {"id": 1, "text": "AHNEDAARBEI extra korting", "bbox": {"left": 0, "top": 0, "width": 260, "height": 20}},
        {"id": 2, "text": "AHNEDAARBEl", "bbox": {"left": 0, "top": 25, "width": 120, "height": 20}},
    ]
    kept, demoted, dropped = _resolve_claim_clusters(0, [[0], [1]], cells, ["AHNEDAARBEI extra korting"])
    assert kept == [[0]] and demoted == []
    assert dropped == [1]


def test_repeated_tokens_do_not_inflate_a_full_match() -> None:
    # Per-character CJK tokens repeat: 小心小心 has token mass 4 but covers only 小+心 of a
    # 4-token line — that is NOT a full match. A cell that truly carries every token of the
    # line (in any order, duplicates or not) still is.
    hint_sets = [set(_tokens("小心地滑")), set(_tokens("小心"))]
    half_covering = _match_scores({"text": "小心小心"}, hint_sets, None)
    assert half_covering.full == (1,)
    truly_full = _match_scores({"text": "地滑小心"}, hint_sets, None)
    assert 0 in truly_full.full


def test_ligature_word_binds_its_line_exactly_but_a_misread_stays_leftover() -> None:
    # A line whose first word carries a ligature (æ) the VLM reads correctly while OCR spells
    # it out: the spelled-out cell must bind that line EXACTLY via the folded token — as a
    # leftover it would get its own translation rendered a second time over the same line. A
    # misread that DROPS the ligature's vowel must NOT reach the folded token through the fuzzy
    # ratio: it stays a leftover by design (the fuzzy universe keeps the unfolded fragments,
    # which — with the ligature early in the word — are too short to match). Synthetic words.
    hint = ["TITELWOORD", "SÆLDA TWEEDE", "DERDE VIERDE"]

    def cells(first_word: str) -> list[dict]:
        return [
            {"id": 1, "text": "TITELWOORD", "bbox": {"left": 60, "top": 10, "width": 300, "height": 40}},
            {"id": 2, "text": first_word, "bbox": {"left": 10, "top": 60, "width": 180, "height": 40}},
            {"id": 3, "text": "TWEEDE", "bbox": {"left": 210, "top": 60, "width": 180, "height": 40}},
            {"id": 4, "text": "DERDE", "bbox": {"left": 10, "top": 110, "width": 180, "height": 40}},
        ]

    bound = build_units_from_hint(cells=cells("SAELDA"), hint_units=hint, model="qwen")
    by_hint = {u.hint_index: [m.text for m in u.members] for u in bound.units}
    assert by_hint.get(1) == ["SAELDA", "TWEEDE"]
    assert all(u.hint_index is not None for u in bound.units)

    misread = build_units_from_hint(cells=cells("SALDA"), hint_units=hint, model="qwen")
    leftovers = [m.text for u in misread.units if u.hint_index is None for m in u.members]
    assert leftovers == ["SALDA"]


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


def test_line_neighbour_binds_shared_word_to_its_printed_line() -> None:
    # A word shared by two lines ("core") sits on the FIRST line, flanked by that line's words, but
    # its axis-aligned top dips low enough that the position estimate (~0.62) alone binds it to the
    # vertically-nearer second line. Its confident left/right line-neighbours win: it stays on its
    # own line. Mirrors a "dieren" shared by several sign sentences flipping to the wrong one.
    def cell(cid: int, text: str, top: int, left: int, width: int, height: int = 20) -> dict:
        return {"id": cid, "text": text, "bbox": {"left": left, "top": top, "width": width, "height": height}}

    hints = ["left core right", "down here core extra"]
    cells = [
        cell(1, "left", 0, 0, 50),
        cell(2, "core", 10, 58, 44),     # ambiguous, dips toward the line below
        cell(3, "right", 0, 110, 50),
        cell(4, "down here", 16, 0, 90),
        cell(5, "extra", 16, 110, 50),
    ]
    result = build_units_from_hint(cells=cells, hint_units=hints, model="qwen")
    shared = next(u for u in result.units if any(m.cell_id == 2 for m in u.members))
    assert shared.hint_index == 0  # without the line-neighbour rule the position tie-break picks 1


def test_sibling_list_items_do_not_collapse_onto_one_hint() -> None:
    # Two itemize items differ by a single word ("first"/"second") and stack directly under each
    # other, so the second reads as a continuation of the first. But it FULLY matches its OWN hint
    # line, so it must bind there — not stick to the first item's hint, which would lose the second
    # item entirely and reflow the first over both lines.
    def cell(cid: int, text: str, top: int) -> dict:
        return {"id": cid, "text": text, "bbox": {"left": 100, "top": top, "width": 200, "height": 14}}

    hints = ["Second level itemize first item", "Second level itemize second item"]
    cells = [
        cell(1, "Second level itemize first item", 0),
        cell(2, "Second level itemize second item", 18),
    ]
    result = build_units_from_hint(cells=cells, hint_units=hints, model="qwen")
    first = next(u for u in result.units if any(m.cell_id == 1 for m in u.members))
    second = next(u for u in result.units if any(m.cell_id == 2 for m in u.members))
    assert first.hint_index == 0
    assert second.hint_index == 1  # its own line, not stuck to item 1


def test_line_anchor_ignores_far_column_neighbour() -> None:
    # The rule only links words a WORD-gap apart, so a receipt's far-left label and far-right value
    # never pull each other — a column-gap neighbour yields no anchor (the cell keeps its own logic).
    def cell(text: str, top: int, left: int, width: int, height: int = 40) -> dict:
        return {"id": 0, "text": text, "bbox": {"left": left, "top": top, "width": width, "height": height}}

    cells = [
        cell("Kaartnr", 100, 600, 160),    # confident, but a column away
        cell("BETALING", 100, 1400, 120),  # the ambiguous cell
    ]
    assert _line_anchor(1, cells, [39, None]) is None


def test_line_anchor_needs_agreeing_sides() -> None:
    def cell(text: str, top: int, left: int, width: int, height: int = 40) -> dict:
        return {"id": 0, "text": text, "bbox": {"left": left, "top": top, "width": width, "height": height}}

    cells = [
        cell("a", 100, 0, 100),     # left line-neighbour
        cell("x", 100, 108, 60),    # the ambiguous cell
        cell("b", 100, 176, 100),   # right line-neighbour
    ]
    assert _line_anchor(1, cells, [7, None, 8]) is None  # sides disagree -> no anchor
    assert _line_anchor(1, cells, [7, None, 7]) == 7     # sides agree -> bind that line


def test_position_guard_filters_far_full_numeric_hint_before_tie_break() -> None:
    # A forecast row's day number ("13") also fully accounts for an earlier standalone
    # temperature hint ("13°"). The far full match must not beat the nearby row hint, or
    # the number becomes a leftover and the renderer draws "Jun 13" beside the original "13".
    def cell(cid: int, text: str, top: int, left: int = 0) -> dict:
        return {"id": cid, "text": text, "bbox": {"left": left, "top": top, "width": 40, "height": 20}}

    hints = [
        "13°",
        "anchor one",
        "anchor two",
        "anchor three",
        "12 jun | Vr | 13° / 18°",
        "13 jun | Za | 11° / 19°",
    ]
    cells = [
        cell(1, "anchor one", 0),
        cell(2, "12", 100, 0),
        cell(3, "jun", 100, 50),
        cell(4, "Vr", 100, 100),
        cell(5, "13", 200, 0),
        cell(6, "jun", 200, 50),
        cell(7, "Za", 200, 100),
    ]
    result = build_units_from_hint(cells=cells, hint_units=hints, model="qwen")
    day_number = next(u for u in result.units if any(m.cell_id == 5 for m in u.members))
    assert day_number.hint_index == 5


def test_position_guard_keeps_unique_alpha_full_match_over_near_partial_match() -> None:
    # Receipt payment metadata can be visually interleaved in two columns: the standalone
    # "BETALING" OCR cell sits near "Contactloze betaling" by y-position, but it fully matches
    # its own later hint line. Do not drop it as a duplicate of the earlier longer line.
    def cell(cid: int, text: str, top: int, left: int = 0) -> dict:
        return {"id": cid, "text": text, "bbox": {"left": left, "top": top, "width": 120, "height": 20}}

    hints = [
        "Token |1234567890123456789",
        "Contactloze betaling |MAESTRO <A0000000043060>",
        "Kaart |123456xxxxxxxxxxx000",
        "Kaartnr |00",
        "Datum |01/01/2020 00:00",
        "BETALING",
        "Auth. code |X00000",
    ]
    cells = [
        cell(1, "Token", 0),
        cell(2, "Contactloze betaling", 100, 0),
        cell(3, "BETALING", 110, 300),
        cell(4, "Kaart", 200),
        cell(5, "Kaartnr", 300),
        cell(6, "Datum", 400),
        cell(7, "Auth. code", 500, 300),
    ]

    result = build_units_from_hint(cells=cells, hint_units=hints, model="qwen")
    payment = next(u for u in result.units if any(m.cell_id == 3 for m in u.members))
    assert payment.hint_index == 5
    assert payment.source_text == "BETALING"
    assert result.ignored_cell_ids == []


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
    from app.grouping.align.positions import _chain

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


def test_parse_grouping_output_strips_bullet_sentinel_and_captures_marker() -> None:
    # New format "|@blt|@<bullet>|<item>": the VLM substitutes the glyph it SAW into @<bullet>; we
    # strip the sentinel, flag the unit, capture the marker, and keep clean item text. Numbered and
    # lettered markers ride the same channel. Older forms (a fixed "@bullet", a bare glyph, a missing
    # glyph field) parse with the default marker; a plain line is no bullet.
    cases = [
        ("*b|CM|12pt|400|l:*|@blt|@•|First item", "First item", True, "•"),
        ("*b|CM|12pt|400|l:*|@blt|@-|Dash item", "Dash item", True, "-"),
        ("*b|CM|12pt|400|l:*|@blt|*|Asterisk", "Asterisk", True, "*"),
        # no separate marker field -> the marker stays in the text, none is invented
        ("*b|CM|12pt|400|l:*|@blt|1. First item", "1. First item", True, None),
        ("*b|CM|12pt|400|l:*|@blt|(a) Second item", "(a) Second item", True, None),
        ("*b|CM|12pt|400|l:*Plain paragraph", "Plain paragraph", False, None),
    ]
    for line, expected_text, expected_bullet, expected_marker in cases:
        hint = parse_grouping_output(line)
        assert hint.units[0] == expected_text, line
        assert hint.bullets[0] is expected_bullet, line
        assert hint.bullet_markers[0] == expected_marker, line


def test_parse_dangling_star_bullet_label_strips_and_flags() -> None:
    # On a bullet line g4 opens the star wrapper but writes the "|@blt" element where the closing
    # star (and often the colon) belong, so si never closes: "*b|CM|12pt|400|l:|@blt|@•|item" (colon,
    # no closing star) and "*b|CM|12pt|400|l|@blt|@•|item" (no colon either — the residual variant).
    # The closed form "*...l:*|@blt|..." is covered above. Without eating that leading "*" the whole
    # line — label + sentinel + text — leaked into the unit and rode into the translation. Both must
    # strip to a clean bullet with marker, level and font parsed exactly as the closed form.
    cases = [
        ("*b|CM|12pt|400|l:|@blt|@•|First item", "First item", True, "•"),
        ("*b|CM|12pt|400|l|@blt|@•|First item", "First item", True, "•"),
        ("*b|CM|12pt|400|l|@blt|■|Second item", "Second item", True, "■"),
        ("*b|CM|12pt|400|l|@blt|1. First item", "1. First item", True, None),
    ]
    for line, expected_text, expected_bullet, expected_marker in cases:
        hint = parse_grouping_output(line)
        assert hint.units[0] == expected_text, line
        assert hint.bullets[0] is expected_bullet, line
        assert hint.bullet_markers[0] == expected_marker, line
        assert hint.levels[0] == "body", line
        assert hint.font_families[0] == "CM", line


def test_two_column_wrapped_tail_rescued_onto_its_paragraph() -> None:
    # Two-column page: the left column's cells interleave with the right column in reading
    # order, so the anchored positions drag the right paragraph's trailing line toward the
    # left column's hint indices — the position guard then rejects every text candidate
    # (including the true one) and the tail orphans: its words render twice (the hint-fed
    # translation of its paragraph already carries them). The rescue binds a leftover cell
    # to the label of the cell geometrically directly above it in its own column when its
    # tokens clear the bind threshold on that line.
    hints = [
        "Kolomtitel alfa bravo charlie delta",   # 0: right-column paragraph, wraps over 3 lines
        "Linksregel een", "Linksregel twee", "Linksregel drie", "Linksregel vier",
        "Linksregel vijf", "Linksregel zes", "Linksregel zeven", "Linksregel acht",
        "Slotwoord charlie delta elders",        # 9: far line that also carries the tail's tokens
    ]
    cells = [
        {"id": 1, "text": "Kolomtitel alfa", "bbox": {"left": 300, "top": 100, "width": 200, "height": 20}},
        {"id": 2, "text": "Linksregel een", "bbox": {"left": 40, "top": 110, "width": 150, "height": 20}},
        {"id": 3, "text": "bravo", "bbox": {"left": 300, "top": 130, "width": 90, "height": 20}},
        {"id": 4, "text": "Linksregel twee", "bbox": {"left": 40, "top": 132, "width": 150, "height": 20}},
        {"id": 5, "text": "Linksregel drie", "bbox": {"left": 40, "top": 150, "width": 150, "height": 20}},
        {"id": 6, "text": "charlie delta", "bbox": {"left": 300, "top": 160, "width": 150, "height": 20}},
        {"id": 7, "text": "Linksregel vier", "bbox": {"left": 40, "top": 170, "width": 150, "height": 20}},
        {"id": 8, "text": "Linksregel vijf", "bbox": {"left": 40, "top": 190, "width": 150, "height": 20}},
        {"id": 9, "text": "Linksregel zes", "bbox": {"left": 40, "top": 210, "width": 150, "height": 20}},
        {"id": 10, "text": "Linksregel zeven", "bbox": {"left": 40, "top": 230, "width": 150, "height": 20}},
        {"id": 11, "text": "Linksregel acht", "bbox": {"left": 40, "top": 250, "width": 150, "height": 20}},
        # a REPEATED print of the tail, stacked right below it: contributes no uncovered token,
        # so the rescue must leave it alone — its own leftover unit, translated at its own spot
        # (gluing it on would get it consolidation-dropped and leave the second print original)
        {"id": 13, "text": "charlie delta", "bbox": {"left": 300, "top": 188, "width": 150, "height": 20}},
        {"id": 12, "text": "Slotwoord charlie delta elders", "bbox": {"left": 40, "top": 400, "width": 260, "height": 20}},
    ]
    result = build_units_from_hint(cells=cells, hint_units=hints, model="qwen")
    by_hint = {u.hint_index: [m.text for m in u.members] for u in result.units}
    assert by_hint.get(0) == ["Kolomtitel alfa", "bravo", "charlie delta"]
    # the far line that shares the tail's tokens keeps only its own cell
    assert by_hint.get(9) == ["Slotwoord charlie delta elders"]
    leftovers = [[m.text for m in u.members] for u in result.units if u.hint_index is None]
    assert leftovers == [["charlie delta"]]  # the repeat: own unit, not glued, not dropped


def test_split_logo_fragment_prefers_its_contiguous_line_over_position() -> None:
    # Letterhead archetype: a two-line logo lockup left, a contact block right, interleaved in
    # reading order. The lockup's second line shares its tokens with a right-column line that
    # happens to sit at the same y, so the interpolated position prefers THAT line by a hair —
    # the fragment then gets icon-dropped there (redundant + detached) and its original pixels
    # survive untranslated. The phrase tie-break must bind it to the line that contains it as a
    # contiguous run (its own lockup) instead of the scattered same-token line.
    hints = [
        "Merknaam van Kwaliteit",                    # 0: the logo lockup, wrapped over two lines
        "De Weledele",
        "Naam Persoon",
        "Directeur & Hoofd van Afdeling Kwaliteit",  # 3: scatter-carries "van" + "Kwaliteit"
    ]
    cells = [
        {"id": 1, "text": "De Weledele", "bbox": {"left": 1220, "top": 147, "width": 300, "height": 40}},
        {"id": 2, "text": "Merknaam", "bbox": {"left": 133, "top": 183, "width": 190, "height": 65}},
        {"id": 3, "text": "Naam Persoon", "bbox": {"left": 1220, "top": 200, "width": 300, "height": 40}},
        {"id": 4, "text": "van Kwaliteit", "bbox": {"left": 132, "top": 238, "width": 230, "height": 58}},
        {"id": 5, "text": "Directeur & Hoofd", "bbox": {"left": 1220, "top": 241, "width": 300, "height": 40}},
        {"id": 6, "text": "van Afdeling Kwaliteit", "bbox": {"left": 1222, "top": 285, "width": 309, "height": 42}},
    ]
    result = build_units_from_hint(cells=cells, hint_units=hints, model="qwen")
    by_hint = {u.hint_index: [m.text for m in u.members] for u in result.units}
    assert by_hint.get(0) == ["Merknaam", "van Kwaliteit"]
    assert by_hint.get(3) == ["Directeur & Hoofd", "van Afdeling Kwaliteit"]
    assert result.ignored_cell_ids == []


def test_layout_columns_bind_the_two_column_tail_directly() -> None:
    # The same two-column shape, now WITH layout evidence: the columns get their own anchor
    # chains, the tail's expected position lands on its paragraph, and it binds through the
    # ordinary matcher — no rescue needed, no leftovers at all.
    hints = [
        "Kolomtitel alfa bravo charlie delta",
        "Linksregel een", "Linksregel twee", "Linksregel drie", "Linksregel vier",
        "Linksregel vijf", "Linksregel zes", "Linksregel zeven", "Linksregel acht",
        "Slotwoord charlie delta elders",
    ]
    cells = [
        {"id": 1, "text": "Kolomtitel alfa", "bbox": {"left": 300, "top": 100, "width": 200, "height": 20}},
        {"id": 2, "text": "Linksregel een", "bbox": {"left": 40, "top": 110, "width": 150, "height": 20}},
        {"id": 3, "text": "bravo", "bbox": {"left": 300, "top": 130, "width": 90, "height": 20}},
        {"id": 4, "text": "Linksregel twee", "bbox": {"left": 40, "top": 132, "width": 150, "height": 20}},
        {"id": 5, "text": "Linksregel drie", "bbox": {"left": 40, "top": 150, "width": 150, "height": 20}},
        {"id": 6, "text": "charlie delta", "bbox": {"left": 300, "top": 160, "width": 150, "height": 20}},
        {"id": 7, "text": "Linksregel vier", "bbox": {"left": 40, "top": 170, "width": 150, "height": 20}},
        {"id": 8, "text": "Linksregel vijf", "bbox": {"left": 40, "top": 190, "width": 150, "height": 20}},
        {"id": 9, "text": "Linksregel zes", "bbox": {"left": 40, "top": 210, "width": 150, "height": 20}},
        {"id": 10, "text": "Linksregel zeven", "bbox": {"left": 40, "top": 230, "width": 150, "height": 20}},
        {"id": 11, "text": "Linksregel acht", "bbox": {"left": 40, "top": 250, "width": 150, "height": 20}},
        {"id": 12, "text": "Slotwoord charlie delta elders", "bbox": {"left": 40, "top": 400, "width": 200, "height": 20}},
    ]
    regions = [
        {"label": "text", "score": 0.95, "coordinate": [290, 90, 520, 200]},    # right column
        {"label": "text", "score": 0.93, "coordinate": [30, 100, 250, 280]},    # left column
        {"label": "text", "score": 0.91, "coordinate": [30, 390, 260, 430]},    # left column, bottom
    ]
    result = build_units_from_hint(cells=cells, hint_units=hints, model="qwen",
                                   layout_regions=regions)
    by_hint = {u.hint_index: [m.text for m in u.members] for u in result.units}
    assert by_hint.get(0) == ["Kolomtitel alfa", "bravo", "charlie delta"]
    assert by_hint.get(9) == ["Slotwoord charlie delta elders"]
    assert all(u.hint_index is not None for u in result.units)


def test_layout_preserve_routes_chart_cells_to_ignored() -> None:
    # A document page with a chart: the chart's inner labels keep their original pixels —
    # align routes them to ignored_cell_ids (not erased, not translated, not a leftover
    # unit) — while the flow text around it binds normally.
    hints = ["Rapportkop over resultaten", "Toelichting bij de grafiek onderaan"]
    cells = [
        {"id": 1, "text": "Rapportkop over resultaten", "bbox": {"left": 40, "top": 30, "width": 400, "height": 24}},
        {"id": 2, "text": "Toelichting bij de grafiek onderaan", "bbox": {"left": 40, "top": 700, "width": 420, "height": 22}},
        {"id": 3, "text": "42%", "bbox": {"left": 200, "top": 300, "width": 50, "height": 16}},
        {"id": 4, "text": "Legenda-item", "bbox": {"left": 220, "top": 400, "width": 90, "height": 14}},
    ]
    regions = [
        {"label": "paragraph_title", "score": 0.95, "coordinate": [30, 20, 460, 70]},
        {"label": "text", "score": 0.93, "coordinate": [30, 680, 480, 740]},
        {"label": "figure_title", "score": 0.90, "coordinate": [30, 750, 480, 790]},
        {"label": "chart", "score": 0.92, "coordinate": [100, 100, 500, 600]},
    ]
    result = build_units_from_hint(cells=cells, hint_units=hints, model="qwen",
                                   layout_regions=regions)
    assert sorted(result.ignored_cell_ids) == [3, 4]
    texts = [m.text for u in result.units for m in u.members]
    assert "42%" not in texts and "Legenda-item" not in texts
    assert result.metrics["layout_gate_open"] is True
    assert result.metrics["layout_preserved_cell_count"] == 2

    # gate closed (no regions): the same cells become ordinary leftovers — nothing preserved
    plain = build_units_from_hint(cells=cells, hint_units=hints, model="qwen")
    assert plain.ignored_cell_ids == []
    assert plain.metrics["layout_gate_open"] is False


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


def test_parse_embedded_degraded_label_splits_two_elements() -> None:
    # Two elements fused onto one line, the second's label mid-line and degraded (pipes ->
    # spaces, weight dropped) or fully piped: the fragment must not leak into the text and
    # opens its own element, with its typography recovered.
    raw = (
        "*b|Sans|16pt|400|l:* Alpha Corp *b Sans 18pt r:* $23.5 billion spend in 2025\n"
        "*b|Sans|16pt|400|l:* Beta Ltd *b|Sans|18pt|400|r:* €14.1 billion spend in 2025\n"
    )
    hint = parse_grouping_output(raw)
    assert hint.units == [
        "Alpha Corp", "$23.5 billion spend in 2025", "Beta Ltd", "€14.1 billion spend in 2025",
    ]
    assert hint.levels == ["body", "body", "body", "body"]
    assert hint.block_ids == [0, 1, 2, 3]
    assert hint.font_sizes == [16, 18, 16, 18]


def test_parse_label_broken_across_newline_leaks_neither_half() -> None:
    # The fused second label can even break at the newline: the stranded opener ("*b") at the
    # first line's end and the code-less tail ("Sans 18pt r:*") opening the next line both strip.
    raw = (
        "*b|Sans|16pt|400|l:* Alpha Corp *b\n"
        "Sans 18pt r:* $23.5 billion spend in 2025\n"
    )
    hint = parse_grouping_output(raw)
    assert hint.units == ["Alpha Corp", "$23.5 billion spend in 2025"]
    assert hint.font_families == ["Sans", "Sans"]
    assert hint.font_sizes == [16, 18]


def test_parse_line_start_spaced_label_recovers_typography() -> None:
    # A line-start label with the pipes degraded to spaces was already stripped (si run); the
    # pipe-normalization now also recovers its level and typography.
    hint = parse_grouping_output("*b Sans 18pt r:* $9.2 billion spend FY2025\n")
    assert hint.units == ["$9.2 billion spend FY2025"]
    assert hint.levels == ["body"]
    assert hint.font_families == ["Sans"]
    assert hint.font_sizes == [18]


def test_parse_pt_mention_in_content_stays_text() -> None:
    # Ordinary content mentioning pt sizes (no star-opener before, no star after the colon)
    # must never be taken for a label fragment.
    raw = (
        "b|Serif|16pt|400|l: Beschikbaar in 10pt, 12pt en 14pt: alle kleuren\n"
        "Vraag naar 12pt formaat: gratis proefdruk\n"
    )
    hint = parse_grouping_output(raw)
    assert hint.units == [
        "Beschikbaar in 10pt, 12pt en 14pt: alle kleuren",
        "Vraag naar 12pt formaat: gratis proefdruk",
    ]


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


def test_bold_text_word_is_not_taken_for_a_label() -> None:
    # "**Menu**" is image text the model bolded, not a label: 'm' happens to be a level code,
    # but a real label carries |-fields. The heading must survive as a unit instead of becoming
    # a standalone footer label that deletes itself and relabels the lines below.
    raw = "**Menu**\nb|Serif|16pt|400|l: Franse vissoep | 8,50\n"
    hint = parse_grouping_output(raw)
    assert hint.units == ["Menu", "Franse vissoep | 8,50"]
    assert hint.levels == [None, "body"]


def test_text_row_starting_with_a_level_word_is_not_a_standalone_label() -> None:
    # A receipt tax row "B | 1,69" (or a table row "Title | Mr") starts with a level code/word
    # but is content: a genuine standalone label shows >= 2 pipes, a "<n>pt" field, or a
    # trailing ':'. The row must stay text; a real standalone label above it still labels it.
    raw = "b|Mono|12pt|400|l:\nB | 1,69\nTitle | Mr\n"
    hint = parse_grouping_output(raw)
    assert hint.units == ["B | 1,69", "Title | Mr"]
    assert hint.levels == ["body", "body"]


def test_footer_word_label_maps_to_footer_level() -> None:
    # The spelled-out "footer|..." first field parses as a label AND keeps its level ('f' is
    # not a code letter, so the first-letter fallback cannot map it).
    raw = "footer|Mono|10pt|400|l: Alle bedragen in euro\n"
    hint = parse_grouping_output(raw)
    assert hint.units == ["Alle bedragen in euro"]
    assert hint.levels == ["footer"]


def test_leading_minus_of_an_amount_survives() -> None:
    # "-2,00 korting": the '-' is the amount's sign, not a markdown bullet — only a marker
    # followed by whitespace ("- Alpha") is decoration.
    raw = "b|Mono|12pt|400|l: -2,00 korting\n- Alpha\n"
    hint = parse_grouping_output(raw)
    assert hint.units == ["-2,00 korting", "Alpha"]


def test_bullet_marker_does_not_eat_a_long_first_field() -> None:
    # In "|@blt|Prijs | 12,50" the field after the sentinel is row content, not a substituted
    # glyph: only a short field (or an explicit "@glyph") counts as the marker.
    raw = "b|CM|12pt|400|l:|@blt|Prijs | 12,50\n"
    hint = parse_grouping_output(raw)
    assert hint.units == ["Prijs | 12,50"]
    assert hint.bullets == [True]
    assert hint.bullet_markers == [None]


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


def _member(cid: int, text: str, left: int, width: int, top: int = 100, height: int = 40, translate: bool = True):
    from app.grouping.units import UnitMember
    return UnitMember(cell_id=cid, text=text, translate=translate, order=cid,
                      bbox={"left": left, "top": top, "width": width, "height": height})


def _col_unit(uid: int, members, hint_index: int = 0):
    from app.grouping.units import TranslationUnit
    return TranslationUnit(id=uid, order=uid, members=members, bbox={}, source_text="", hint_index=hint_index)


def test_geometry_injects_missed_column_pipe_but_not_a_wide_font_word_space() -> None:
    # JOUW VOORDEEL 2,44 in a wide header font: the VLM dropped the rule-4 `|`. Geometry adds it
    # before the far-right amount — and must NOT split JOUW|VOORDEEL (a word space in a wide font,
    # ~0.9 char-widths, while the amount gap is ~2x the measured word space).
    from app.grouping.field_geometry import geometry_adjusted_hints
    unit = _col_unit(1, [
        _member(1, "JOUW", 663, 282, height=85),
        _member(2, "VOORDEEL", 1009, 550, height=85),
        _member(3, "2.44", 1697, 298, height=85, translate=False),
    ])
    adjusted, changes = geometry_adjusted_hints([unit], ["JOUW VOORDEEL. 2,44"])
    assert adjusted == ["JOUW VOORDEEL. | 2,44"]
    assert changes[0]["mapped_into_vlm_line"] is True


def test_geometry_injects_pipe_for_two_translatable_columns() -> None:
    # Two translatable columns OCR read with a clear gap (no measured word space, 2 cells -> the
    # char-width reference). The VLM omitted the `|`; geometry restores it.
    from app.grouping.field_geometry import geometry_adjusted_hints
    unit = _col_unit(1, [
        _member(1, "Met consumentenapparaat", 592, 646, height=40),
        _member(2, "gevalideerd", 1386, 326, height=40),
    ])
    adjusted, _ = geometry_adjusted_hints([unit], ["Met consumentenapparaat gevalideerd"])
    assert adjusted == ["Met consumentenapparaat | gevalideerd"]


def test_geometry_leaves_normal_word_spacing_and_existing_pipes_alone() -> None:
    from app.grouping.field_geometry import geometry_adjusted_hints
    # A normal prose line (uniform word spacing) is not a column -> untouched.
    prose = _col_unit(1, [
        _member(1, "They", 0, 120), _member(2, "can", 130, 90),
        _member(3, "cause", 230, 140), _member(4, "harm", 380, 120),
    ])
    # A line the VLM already marked with `|` is left as-is.
    marked = _col_unit(2, [_member(1, "TOTAAL", 0, 200), _member(2, "58,51", 900, 200, translate=False)], hint_index=1)
    adjusted, changes = geometry_adjusted_hints([prose, marked], ["They can cause harm", "TOTAAL | 58,51"])
    assert adjusted == ["They can cause harm", "TOTAAL | 58,51"]
    assert changes == []


def test_geometry_skips_an_all_numeric_row_with_no_translatable_field() -> None:
    # Two side-by-side temperatures ("15°" and "15") OCR read as separate cells both bound to one
    # hint line. They are non-translatable, so a column `|` changes nothing — do not flag a
    # meaningless "15° | 15".
    from app.grouping.field_geometry import geometry_adjusted_hints
    unit = _col_unit(1, [
        _member(1, "15°", 410, 83, top=734, height=56, translate=False),
        _member(2, "15", 573, 71, top=736, height=54, translate=False),
    ])
    adjusted, changes = geometry_adjusted_hints([unit], ["15°"])
    assert adjusted == ["15°"]
    assert changes == []


def test_symbol_leftover_absorbed_into_its_line_and_translated() -> None:
    # OCR splits a styled '&' off a byline into its own cell; no tokens -> it can never bind by
    # matching, orphans as a leftover, keeps its pixels, and the line's translation renders the
    # '&' AGAIN next to them. It must fold back into the element whose printed line it sits on
    # (word-gap adjacency + the hint line carries the symbol) with translate=True so it is
    # erased. A symbol the hint does NOT carry (a decorative dingbat) stays its own leftover.
    hints = ["by Alpha Beta, & Gamma Delta"]
    cells = [
        {"id": 1, "text": "by Alpha Beta,", "bbox": {"left": 100, "top": 100, "width": 300, "height": 30}},
        {"id": 2, "text": "&", "bbox": {"left": 100, "top": 140, "width": 30, "height": 30}},
        {"id": 3, "text": "Gamma Delta", "bbox": {"left": 136, "top": 140, "width": 260, "height": 30}},
        {"id": 4, "text": "◆", "bbox": {"left": 100, "top": 400, "width": 30, "height": 30}},
    ]
    result = build_units_from_hint(cells=cells, hint_units=hints, model="qwen")
    by_hint = {u.hint_index: u for u in result.units}
    members = {m.cell_id: m for m in by_hint[0].members}
    assert set(members) == {1, 2, 3}
    assert members[2].translate is True  # erased; the hint translation carries the '&'
    leftovers = [[m.text for m in u.members] for u in result.units if u.hint_index is None]
    assert leftovers == [["◆"]]  # not in the hint text -> keeps its own pixels, as before


def test_exact_second_print_demotes_to_leftover_instead_of_ignored() -> None:
    # A redundant claim that is EXACT-clean and spatially apart from the kept claim is a
    # genuine OTHER print of duplicated text (a checkbox label repeated per column, a wrapped
    # tail a short header line full-match-stole): it must become its own leftover unit —
    # translated and rendered at its own spot — not ignored original pixels. A garbled
    # double-read (fuzzy-bound) still drops (see the stray test above).
    cells = [
        {"id": 1, "text": "Part D", "bbox": {"left": 100, "top": 100, "width": 90, "height": 24}},
        {"id": 2, "text": "Andere regel ertussen", "bbox": {"left": 100, "top": 400, "width": 300, "height": 24}},
        {"id": 3, "text": "Part D", "bbox": {"left": 900, "top": 700, "width": 90, "height": 24}},
    ]
    kept, demoted, dropped = _resolve_claim_clusters(
        0, [[0], [2]], cells, ["Part D"]
    )
    assert kept == [[0]]
    assert demoted == [[2]]
    assert dropped == []
    # end-to-end: the demoted print surfaces as a leftover unit, not in ignored
    result = build_units_from_hint(
        cells=cells, hint_units=["Part D", "Andere regel ertussen"], model="qwen"
    )
    assert result.ignored_cell_ids == []
    leftovers = [[m.cell_id for m in u.members] for u in result.units if u.hint_index is None]
    assert leftovers == [[3]]


def test_detached_second_print_via_merge_refusal_demotes_not_drops() -> None:
    # Display-title archetype: a TITLE whose words also occur in body prose. The title cells
    # claim the body's hint line, look like new content at first (merge candidates), the
    # paragraph then merges fully (their tokens are covered after all) but they sit far away
    # (merge refused). That refusal path must apply the same second-print verdict: clean +
    # spatially apart -> demote to a translated leftover, not ignored original pixels.
    cells = [
        # the display title, far above the paragraph
        {"id": 1, "text": "Shared Record", "bbox": {"left": 60, "top": 70, "width": 400, "height": 60}},
        # the body paragraph, wrapped over two adjacent printed lines
        {"id": 2, "text": "navigate the Shared", "bbox": {"left": 100, "top": 600, "width": 300, "height": 24}},
        {"id": 3, "text": "Record easily", "bbox": {"left": 100, "top": 628, "width": 280, "height": 24}},
    ]
    kept, demoted, dropped = _resolve_claim_clusters(
        0, [[1], [2], [0]], cells, ["navigate the Shared Record easily"]
    )
    assert kept == [[1, 2]]      # paragraph line + its wrapped continuation merged
    assert demoted == [[0]]      # the detached title print, rescued
    assert dropped == []


def test_parse_latex_arrow_token_normalises_to_glyph() -> None:
    # The VLM reads an arrow glyph as LaTeX; the notation must never reach translator/render.
    parsed = parse_grouping_output("*h|Roboto|18pt|700|l:* 4 $\\rightarrow$ Go To Arrow")
    assert parsed.units == ["4 → Go To Arrow"]


def test_parse_pipe_opened_embedded_label_splits_table_columns() -> None:
    # A two-column table row where the model writes the second column's typography label
    # straight after the row's field separator: the "|" opens the label (no star). The label
    # must strip, the second column becomes its own element, and the separator itself must
    # not leak into the first column's text.
    raw = (
        "*h|Serif|14pt|700|l:* Covered | h|Serif|14pt|700|r:* Not Covered\n"
        "*b|Serif|11pt|400|l:* You may be covered if X. | b|Serif|11pt|400|r:* ...but not if Y.\n"
    )
    hint = parse_grouping_output(raw)
    assert hint.units == [
        "Covered", "Not Covered", "You may be covered if X.", "...but not if Y.",
    ]
    assert hint.levels == ["header", "header", "body", "body"]
    assert hint.font_sizes == [14, 14, 11, 11]


def test_symbolic_label_classifier() -> None:
    # Rating codes and symbol runs with a stray capital are labels, not language.
    from app.grouping.preserve import _is_symbolic_label

    assert _is_symbolic_label('A-')
    assert _is_symbolic_label('A "')
    assert _is_symbolic_label('A+ B')
    # Two adjacent letters = a word candidate; short real words are out.
    assert not _is_symbolic_label('Nu')
    assert not _is_symbolic_label('be attentive to signs')
    # Any lowercase letter is out: measurements ("25 m", "1,5 m") and stray
    # lowercase fragments follow the existing _PRICE_TAX convention.
    assert not _is_symbolic_label('1,5 m')
    assert not _is_symbolic_label('i')
    # A single CJK char is a complete word.
    assert not _is_symbolic_label('推')
    assert not _is_symbolic_label('危險')


def test_all_symbolic_unit_is_preserved_whole() -> None:
    # A unit that is ENTIRELY rating-code tokens (three cells on one geometric
    # line) never reaches the translator — the measured batch hallucination
    # ('A " A- A-' answered with another line's sentence) starts here. A single
    # capital inside a prose unit keeps its normal flag.
    from app.grouping.align.build import _build_unit

    def cell(cid, text, left):
        return {"id": cid, "text": text, "bbox": {"left": left, "top": 100, "width": 90, "height": 40}}

    label_row = _build_unit(
        cells=[cell(1, 'A "', 100), cell(2, "A-", 300), cell(3, "A-", 500)],
        indices=[0, 1, 2], unit_id=1,
    )
    assert [m.translate for m in label_row.members] == [False, False, False]
    assert label_row.source_text == ""

    prose = _build_unit(
        cells=[cell(1, "maat", 100), cell(2, "M", 300)],
        indices=[0, 1], unit_id=2,
    )
    assert [m.translate for m in prose.members] == [True, True]

    # A SINGLE symbolic cell (an icon OCR'd as one capital, a receipt code)
    # keeps today's behaviour — the rule needs a row of them.
    single = _build_unit(cells=[cell(1, "Q", 100)], indices=[0], unit_id=3)
    assert [m.translate for m in single.members] == [True]


def test_cell_marker_makes_the_unit_a_bullet() -> None:
    # A cell that carries a stripped inline marker (text-layer "•") is ground
    # truth for the bullet flag — no hint label needed. OCR cells never carry
    # the field, so the image flow is untouched by construction.
    from app.grouping.align.build import _build_unit

    cells = [
        {"id": 1, "text": "report suspected reactions", "marker": "•",
         "bbox": {"left": 178, "top": 100, "width": 500, "height": 30}},
        {"id": 2, "text": "Yellow Card",
         "bbox": {"left": 178, "top": 132, "width": 140, "height": 30}},
    ]
    unit = _build_unit(cells=cells, indices=[0, 1], unit_id=1)
    assert unit.bullet is True
    assert unit.bullet_marker == "•"

    plain = _build_unit(cells=[dict(cells[1])], indices=[0], unit_id=2)
    assert plain.bullet is False


def test_unit_serde_keeps_a_text_layer_box_at_full_precision() -> None:
    # A PDF text layer's boxes are fractional pixels. Truncating them on the way back from
    # JSON re-renders a cached run on boxes up to a pixel smaller than the run that produced
    # them, so every re-entry (re-render, re-translate) drifts from the render it reproduces.
    from app.grouping.units import TranslationUnit
    from app.grouping.units import UnitMember

    member = UnitMember(
        cell_id=1, text="Attention", translate=True, order=1,
        bbox={"left": 160.0, "top": 138.69, "width": 484.92, "height": 32.34},
        polygon=[{"x": 160.0, "y": 138.69}],
    )
    unit = TranslationUnit(
        id=1, order=1, members=[member], source_text="Attention",
        bbox={"left": 160.0, "top": 138.69, "width": 484.92, "height": 32.34},
    )
    restored = TranslationUnit.from_dict(unit.to_dict())
    assert restored.bbox == {"left": 160, "top": 138.69, "width": 484.92, "height": 32.34}
    assert restored.members[0].bbox == restored.bbox
    assert restored.members[0].polygon == [{"x": 160, "y": 138.69}]
    # An OCR run's whole pixels stay int, so its artifacts keep their exact shape.
    ocr = UnitMember(
        cell_id=2, text="HELLO", translate=True, order=1,
        bbox={"left": 40, "top": 20, "width": 160, "height": 20},
    )
    assert UnitMember.from_dict(ocr.to_dict()).bbox == {"left": 40, "top": 20, "width": 160, "height": 20}
    assert all(isinstance(v, int) for v in UnitMember.from_dict(ocr.to_dict()).bbox.values())


def test_overclaimed_hint_suffix_is_trimmed_but_an_honest_repeat_is_not() -> None:
    # The grouping VLM reads a page as running prose, so a sentence that breaks across a
    # column can be written out in full on the line where it STARTS and again as the
    # continuation's own line. Both match cells, so the tail translates and renders twice.
    from app.grouping.align.overclaim import trim_overclaimed_hint_lines
    from app.grouping.units import TranslationUnit
    from app.grouping.units import UnitMember

    def unit(uid: int, hint_index: int, *texts: str) -> TranslationUnit:
        members = [
            UnitMember(cell_id=uid * 10 + n, text=text, translate=True, order=n,
                       bbox={"left": 0, "top": n * 20, "width": 100, "height": 18})
            for n, text in enumerate(texts, start=1)
        ]
        return TranslationUnit(
            id=uid, order=uid, members=members, source_text=" ".join(texts),
            bbox={"left": 0, "top": 0, "width": 100, "height": 18}, hint_index=hint_index,
        )

    tail = "high-quality dataset. An example of these differences is in Figure 2."
    hints = [f"2022) advances methods that are also suitable for {tail}", tail]
    units = [unit(1, 0, "2022) advances methods that are also suitable for"),
             unit(2, 1, "high-quality dataset. An example of these", "differences is in Figure 2.")]
    trimmed = trim_overclaimed_hint_lines(units, hints)
    assert trimmed[0] == "2022) advances methods that are also suitable for"
    assert trimmed[1] == tail          # the line that legitimately owns the text is untouched

    # An equal-length repeat is two honest lines (a warning printed twice): both keep their
    # text, or the second occurrence would render untranslated.
    repeat = "Danger! Do not enter this area without protective equipment"
    units = [unit(1, 0, repeat), unit(2, 1, repeat)]
    assert trim_overclaimed_hint_lines(units, [repeat, repeat]) == [repeat, repeat]

    # A line whose OWN cells print the tail keeps it: that is a real second print on the page,
    # not another line's text.
    own = [f"Intro line {tail}", tail]
    units = [unit(1, 0, "Intro line", "high-quality dataset. An example of these",
                  "differences is in Figure 2."),
             unit(2, 1, tail)]
    assert trim_overclaimed_hint_lines(units, own) == own


def test_a_heading_number_is_not_a_column_but_a_quantity_still_is() -> None:
    # A section number set in its own cell a wide space before its title lands on a knife edge in
    # the 2-cell gap rule: the threshold scales with average glyph width, so a LONGER title lowers
    # it and the same 27px gap splits one heading while leaving its neighbour whole. Number and
    # title are one heading, so the split must never happen there — while a body row's leading
    # quantity column (a receipt line) still splits on the same geometry.
    from app.grouping.field_geometry import geometry_adjusted_hints

    def heading(uid, title, width, hint_index):
        return _col_unit(uid, [
            _member(1, str(uid), 300, 17, top=1000, height=33),
            _member(2, title, 350, width, top=1000, height=33),
        ], hint_index=hint_index)

    units = [heading(2, "Background", 175, 0), heading(3, "Model Architecture", 278, 1)]
    for unit in units:
        object.__setattr__(unit, "level", "header")
    adjusted, _changes = geometry_adjusted_hints(units, ["2 Background", "3 Model Architecture"])
    assert adjusted == ["2 Background", "3 Model Architecture"]

    # Body text with a column-wide gap: still a column (a quantity beside a product name). The
    # heading rule must not have disabled the mechanism itself.
    body = _col_unit(9, [
        _member(1, "2", 300, 17, top=1000, height=33),
        _member(2, "Cola zero", 460, 175, top=1000, height=33),
    ])
    adjusted, _changes = geometry_adjusted_hints([body], ["2 Cola zero"])
    assert adjusted == ["2 | Cola zero"]
    # ...and the SAME row labelled a heading keeps its number attached.
    object.__setattr__(body, "level", "title")
    adjusted, _changes = geometry_adjusted_hints([body], ["2 Cola zero"])
    assert adjusted == ["2 Cola zero"]
