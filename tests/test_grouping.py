from __future__ import annotations

from app.grouping.align import build_units_from_hint
from app.grouping.vlm import parse_grouping_output


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


def test_consecutive_match_becomes_one_flow_unit() -> None:
    result = build_units_from_hint(cells=_cells(), hint_units=_hint(), model="qwen")
    flow = result.units[0]
    assert flow.kind == "flow"
    assert [m.cell_id for m in flow.members] == [1, 2, 3]
    assert flow.source_text == "THE SHOE WORKS IF YOU DO."
    # union bbox spans the three stacked lines
    assert flow.bbox == {"left": 10, "top": 10, "width": 220, "height": 140}


def test_single_matched_cell_is_field_unit() -> None:
    cells = [{"id": 1, "text": "KARNEMELK", "bbox": {"left": 10, "top": 10, "width": 160, "height": 20}}]
    result = build_units_from_hint(cells=cells, hint_units=["KARNEMELK"], model="qwen")
    assert len(result.units) == 1
    assert result.units[0].kind == "field"
    assert result.units[0].source_text == "KARNEMELK"


def test_url_member_is_not_translatable() -> None:
    result = build_units_from_hint(cells=_cells(), hint_units=_hint(), model="qwen")
    nike = next(u for u in result.units if any(m.cell_id == 4 for m in u.members))
    member = nike.members[0]
    assert member.translate is False
    assert nike.source_text == ""  # nothing translatable -> empty


def test_isolated_number_is_its_own_unit() -> None:
    # "1,69" matches no hint and sits in its own column (a gap from KARNEMELK), so it forms
    # its own non-translatable field unit instead of dragging the product's anchor right.
    result = build_units_from_hint(cells=_cells(), hint_units=_hint(), model="qwen")
    price_unit = next(u for u in result.units if any(m.cell_id == 6 for m in u.members))
    assert [m.cell_id for m in price_unit.members] == [6]
    assert price_unit.kind == "field"
    assert price_unit.members[0].translate is False
    karnemelk = next(u for u in result.units if any(m.cell_id == 5 for m in u.members))
    assert [m.cell_id for m in karnemelk.members] == [5]  # not dragged by the distant number


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


def test_empty_hint_makes_each_cell_its_own_field_unit() -> None:
    result = build_units_from_hint(cells=_cells(), hint_units=[], model="qwen")
    assert len(result.units) == len(_cells())
    assert all(u.kind == "field" for u in result.units)


def test_accent_and_case_tolerant_matching() -> None:
    cells = [{"id": 1, "text": "HATTA", "bbox": {"left": 0, "top": 0, "width": 10, "height": 10}}]
    result = build_units_from_hint(cells=cells, hint_units=["hatta!"], model="qwen")
    assert [m.cell_id for m in result.units[0].members] == [1]


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
    # No category line -> empty category; bullets/markdown stripped; ### / ----- split
    # blocks. Within a block, non-sentence lines join (a wrapped line continues).
    raw = "- Alpha\n* Beta\n-----\n**Voorgerechten:**\n###\n  Gamma  \n"
    hint = parse_grouping_output(raw)
    assert hint.category == ""
    assert hint.units == ["Alpha Beta", "Voorgerechten:", "Gamma"]


def test_parse_grouping_output_splits_block_on_sentences() -> None:
    # A heading and its explanation, stacked in one block, split at the sentence end; a
    # sentence wrapped across two lines (ending mid-sentence) stays one unit.
    raw = (
        "###\n"
        "Drijf de dieren niet op.\nGeef ze de tijd en de ruimte.\n"
        "###\n"
        "Ze worden hier opdringerig van en gaan bijten,\ndit dwingt ons ze te slachten.\n"
    )
    hint = parse_grouping_output(raw)
    assert hint.units == [
        "Drijf de dieren niet op.",
        "Geef ze de tijd en de ruimte.",
        "Ze worden hier opdringerig van en gaan bijten, dit dwingt ons ze te slachten.",
    ]


def test_parse_grouping_output_splits_table_row_on_pipe() -> None:
    # A "|" table row splits into one unit per field (each maps to its own OCR box).
    raw = "###\n1 | Karnemelk | 1,69\n###\n1 | AH Yogurt | 2,09\n"
    hint = parse_grouping_output(raw)
    assert hint.units == ["1", "Karnemelk", "1,69", "1", "AH Yogurt", "2,09"]
