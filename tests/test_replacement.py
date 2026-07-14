from __future__ import annotations

from io import BytesIO

import cv2
import numpy as np
import pytest
from PIL import Image
from PIL import ImageDraw

import math

from app.replacement.ground.color import sample_region_colors
from app.replacement.layout.markers import _strip_unprinted_lead
from app.replacement.ground.inpaint import border_pads
from app.replacement.ground.inpaint import budget_scale
from app.replacement.ground.inpaint import context_window
from app.replacement.ground.inpaint import work_scale
from app.replacement.layout.planning import _justified_text_image
from app.replacement.layout.planning import _justify_feasible
from app.replacement.layout.planning import _justify_overhang_width
from app.replacement.layout.planning import _planes_justified
from app.replacement.text.fit import _dominant_script
from app.replacement.text.fit import fit_text
from app.replacement.text.fit import load_font
from app.replacement.text.fit import fold_lone_fullwidth_punctuation
from app.replacement.text.fit import is_cjk_text
from app.replacement.text.angle import _baseline_angle
from app.replacement.text.size import _group_size
from app.replacement.render import _plan_group
from app.replacement.layout.tables import _reproduced_in
from app.replacement.layout.tables import _split_table_row
from app.replacement.render import render_translated_image


def test_dominant_script_routes_by_majority_not_shared_danda() -> None:
    # The Devanagari danda U+0964 ("।") ends Bengali / Tamil / Devanagari sentences alike; the face
    # must follow the line's own (majority) script, not the first range that the danda falls in
    # (which would send a whole Bengali line to the Devanagari face -> tofu).
    assert _dominant_script("আপনি প্রবেশ করছেন।") == "NotoSansBengali[wght].ttf"
    assert _dominant_script("வணக்கம் உலகம்।") == "NotoSansTamil[wght].ttf"
    assert _dominant_script("नमस्ते दुनिया।") == "NotoSansDevanagari[wght].ttf"
    assert _dominant_script("مرحبا بالعالم") == "NotoSansArabic[wght].ttf"
    assert _dominant_script("Hello, world!") is None  # Latin -> family / DejaVu chain


def test_reproduced_in_erases_inline_token_the_translation_repeats() -> None:
    # OCR split the inline "1, 2, 3, 4?" off the question into its own non-translatable member;
    # the structured translation re-emits it ("...after 1, 2, 3, 4?"), so the original must be
    # erased (pulled into the unit's erase) — not kept on top, which doubles it.
    assert _reproduced_in({"text": "1,2,3,4?"}, "Ruben, what comes after 1, 2, 3, 4?") is True


def test_reproduced_in_keeps_a_token_that_is_the_whole_translation() -> None:
    # A standalone non-translatable token translating to itself carries no other content, so it
    # stays preserved in place (nothing to double it against).
    assert _reproduced_in({"text": "58,41"}, "58,41") is False
    assert _reproduced_in({"text": "€ 58,41"}, "€ 58,41") is False


def test_split_table_row_groups_a_wrapped_column_into_one_cell() -> None:
    # The spend column wrapped across 3 OCR lines (one field -> several members at the same x);
    # the row must still split into 2 column-cells (company | spend), not bail because members
    # outnumber fields. Each fragment binds to the spend field by containment, not full-line ratio.
    unit = {
        "field_translations": [
            ("Amazon", "Amazon"),
            ("$23,5 miljard aan advertising & promotional costs in 2025",
             "$23.5 billion in advertising & promotional costs in 2025"),
        ],
        "members": [
            {"text": "Amazon", "translate": True, "bbox": {"left": 50, "top": 388, "width": 160, "height": 58}},
            {"text": "$23,5 miljard a", "translate": True, "bbox": {"left": 656, "top": 385, "width": 300, "height": 58}},
            {"text": "advertising &", "translate": True, "bbox": {"left": 655, "top": 470, "width": 300, "height": 56}},
            {"text": "promotional co", "translate": True, "bbox": {"left": 656, "top": 554, "width": 300, "height": 51}},
        ],
    }
    cells = _split_table_row(unit)
    assert cells is not None and len(cells) == 2
    by_left = {c["members"][0]["bbox"]["left"]: c for c in cells}
    assert set(by_left) == {50, 656}  # a company column and a spend column, at distinct x
    assert by_left[50]["translated_text"] == "Amazon"
    assert by_left[50]["members"] == [unit["members"][0]]  # company cell = its one member
    assert len(by_left[656]["members"]) == 3  # spend cell = all three wrapped fragments
    assert "advertising" in by_left[656]["translated_text"]


def test_split_table_row_pulls_a_reproduced_number_line_into_its_column() -> None:
    # The spend column's last line "2025" is a pure number -> non-translatable, but the spend
    # translation re-emits it. It must join the spend cell so the column erase covers it, instead
    # of the original "2025" showing through under the smaller re-rendered text.
    unit = {
        "field_translations": [
            ("Amazon", "Amazon"),
            ("$23,5 miljard aan advertising in 2025", "$23.5 billion in advertising in 2025"),
        ],
        "members": [
            {"text": "Amazon", "translate": True, "bbox": {"left": 50, "top": 388, "width": 160, "height": 58}},
            {"text": "$23,5 miljard aan advertising", "translate": True,
             "bbox": {"left": 656, "top": 385, "width": 300, "height": 58}},
            {"text": "2025", "translate": False, "bbox": {"left": 656, "top": 470, "width": 90, "height": 50}},
        ],
    }
    cells = _split_table_row(unit)
    assert cells is not None and len(cells) == 2
    spend = next(c for c in cells if c["members"][0]["bbox"]["left"] == 656)
    assert "2025" in [m["text"] for m in spend["members"]]  # reproduced number pulled in to be erased


def test_split_table_row_shares_one_box_between_two_fields() -> None:
    # The VLM split 'PRIJS | BEDRAG' but OCR read it as a single box, alongside a second column.
    # The lone box carries both field translations in field order; the row still yields 2 cells.
    unit = {
        "field_translations": [("PRIJS", "PRICE"), ("BEDRAG", "AMOUNT"), ("TOTAAL", "TOTAL")],
        "members": [
            {"text": "PRIJS BEDRAG", "translate": True, "bbox": {"left": 40, "top": 10, "width": 200, "height": 30}},
            {"text": "TOTAAL", "translate": True, "bbox": {"left": 400, "top": 10, "width": 120, "height": 30}},
        ],
    }
    cells = _split_table_row(unit)
    assert cells is not None and len(cells) == 2
    shared = next(c for c in cells if c["members"][0]["bbox"]["left"] == 40)
    assert shared["translated_text"] == "PRICE AMOUNT"  # both fields, kept in field order


def test_split_table_row_merges_touching_date_and_weekday_fields() -> None:
    # Weather forecast rows may be hinted as "13 jun | Za"; the translated date ("Jun 13")
    # needs to render together with the weekday, not as two touching pseudo-columns ("Jun 13Sat").
    unit = {
        "field_translations": [("13 jun", "Jun 13"), ("Za", "Sat")],
        "members": [
            {"text": "13", "translate": False, "bbox": {"left": 90, "top": 1598, "width": 65, "height": 62}},
            {"text": "jun", "translate": True, "bbox": {"left": 150, "top": 1598, "width": 89, "height": 67}},
            {"text": "Za", "translate": True, "bbox": {"left": 249, "top": 1596, "width": 75, "height": 56}},
        ],
    }
    cells = _split_table_row(unit)
    assert cells is not None and len(cells) == 1
    assert cells[0]["translated_text"] == "Jun 13 Sat"
    assert [member["text"] for member in cells[0]["members"]] == ["13", "jun", "Za"]


def test_split_table_row_keeps_explicit_quantity_and_price_columns() -> None:
    # In "translate everything" mode, field_translations contains fields that the aligner marked
    # non-translatable (quantity/price). They still need their own cells; otherwise the renderer
    # falls back to reflowing the whole receipt row and collapses the table columns.
    unit = {
        "field_translations": [
            ("1", "1"),
            ("KARNEMELK", "SKIMMED MILK"),
            ("1,69 B", "1.69 B"),
        ],
        "members": [
            {"text": "1", "translate": False, "bbox": {"left": 10, "top": 20, "width": 20, "height": 20}},
            {"text": "KARNEMELK", "translate": True, "bbox": {"left": 80, "top": 20, "width": 140, "height": 20}},
            {"text": "1,69 B", "translate": False, "bbox": {"left": 300, "top": 20, "width": 80, "height": 20}},
        ],
    }
    cells = _split_table_row(unit)
    assert cells is not None and len(cells) == 3
    assert [cell["translated_text"] for cell in cells] == ["1", "SKIMMED MILK", "1.69 B"]


def test_split_table_row_single_changed_field_leaves_preserved_neighbor_outside_cell() -> None:
    # With preserve_unchanged_text enabled, field_translations only contains changed fields.
    # The renderer must still split the row so the erase for "Contactless payment" cannot wipe
    # the unchanged MAESTRO field beside it.
    unit = {
        "field_translations": [("Contactloze betaling", "Contactless payment")],
        "members": [
            {
                "text": "Contactloze betaling",
                "translate": True,
                "bbox": {"left": 611, "top": 3377, "width": 560, "height": 90},
            },
            {
                "text": "MAESTRO <A0000000043060>",
                "translate": True,
                "bbox": {"left": 1400, "top": 3398, "width": 676, "height": 122},
            },
        ],
    }

    cells = _split_table_row(unit)

    assert cells is not None and len(cells) == 1
    assert cells[0]["translated_text"] == "Contactless payment"
    assert [member["text"] for member in cells[0]["members"]] == ["Contactloze betaling"]


def test_split_table_row_fires_for_cyrillic_fields() -> None:
    # A non-Latin row: an ASCII-only field key is empty for Cyrillic, no member ever placed, and
    # the row silently reflowed as one joined line — the price rendered behind the item name.
    unit = {
        "field_translations": [("МОЛОКО", "MILK"), ("1,69", "1,69")],
        "members": [
            {"text": "МОЛОКО", "translate": True, "bbox": {"left": 40, "top": 10, "width": 160, "height": 30}},
            {"text": "1,69", "translate": True, "bbox": {"left": 500, "top": 10, "width": 80, "height": 30}},
        ],
    }
    cells = _split_table_row(unit)
    assert cells is not None and len(cells) == 2
    assert {c["translated_text"] for c in cells} == {"MILK", "1,69"}


def test_slot_sweep_erases_ink_the_ocr_box_undershot(tmp_path) -> None:
    # The OCR box can undershoot the glyphs by a few px; the digit bottoms then survive the
    # member erase as dash-like remnants. A superseded (erase-only) line owns its whole slot,
    # so the sweep must leave it fully background.
    input_path = tmp_path / "in.png"
    img = Image.new("RGB", (300, 120), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    for x in range(44, 196, 14):    # glyph-like strokes, line 1: ink matches its box
        draw.rectangle((x, 24, x + 6, 38), fill=(0, 0, 0))
    for x in range(44, 136, 14):    # line 2: strokes stick 6px below the box (y 62..86)
        draw.rectangle((x, 62, x + 6, 86), fill=(0, 0, 0))
    img.save(input_path)
    units = [{
        "translated_text": "Hi",
        # hint-side source carries a token no member accounts for -> the sweep gate opens
        "field_translations": [("HELLO WORLD in", "Hi")],
        "members": [
            {"cell_id": 1, "text": "HELLO", "translate": True,
             "bbox": {"left": 40, "top": 20, "width": 160, "height": 20}},
            {"cell_id": 2, "text": "WORLD", "translate": True,
             "bbox": {"left": 40, "top": 60, "width": 100, "height": 22}},  # ink ends at y=86
        ],
    }]
    png = render_translated_image(input_path, units)
    out = np.asarray(Image.open(BytesIO(png)).convert("RGB"))
    line2_band = out[56:92, 30:220]
    assert line2_band.min() > 200  # no ink remnants below the undershot box


def test_sweep_erases_unclaimed_ink_but_keeps_other_units_pixels(tmp_path) -> None:
    # Screenshot anatomy: OCR detected '2025' but missed the 'in' before it. The unclaimed ink
    # inside the group's own line band is leftover source text -> swept. The same ink, when it
    # belongs to ANOTHER unit's member (a preserved/skipped cell), is protected ground.
    def make_input(path):
        img = Image.new("RGB", (300, 120), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        for x in range(44, 236, 14):   # line 1 glyph strokes (wide -> group span)
            draw.rectangle((x, 24, x + 6, 38), fill=(0, 0, 0))
        draw.rectangle((62, 64, 68, 78), fill=(0, 0, 0))     # the OCR-missed "in" ink
        draw.rectangle((74, 64, 80, 78), fill=(0, 0, 0))
        for x in range(102, 156, 14):  # the detected '2025' strokes
            draw.rectangle((x, 64, x + 6, 78), fill=(0, 0, 0))
        img.save(path)

    base_unit = {
        "translated_text": "Hi",
        # The hint-side source carries "in", which no member accounts for — the gate that
        # allows sweeping: the translation covers the undetected word.
        "field_translations": [("HELLO WORLD AGAIN in 2025", "Hi")],
        "members": [
            {"cell_id": 1, "text": "HELLO WORLD AGAIN", "translate": True,
             "bbox": {"left": 40, "top": 20, "width": 200, "height": 20}},
            {"cell_id": 2, "text": "2025", "translate": True,
             "bbox": {"left": 100, "top": 60, "width": 60, "height": 22}},
        ],
    }

    unclaimed = tmp_path / "unclaimed.png"
    make_input(unclaimed)
    out = np.asarray(Image.open(BytesIO(render_translated_image(unclaimed, [base_unit]))).convert("RGB"))
    assert out[64:78, 60:84].min() > 200  # unclaimed "in" ink swept with the band

    protected = tmp_path / "protected.png"
    make_input(protected)
    other_unit = {  # same ink, but now claimed by a skipped unit (empty translation)
        "translated_text": "",
        "members": [{"cell_id": 3, "text": "in", "translate": False,
                     "bbox": {"left": 60, "top": 62, "width": 24, "height": 18}}],
    }
    out = np.asarray(Image.open(BytesIO(render_translated_image(protected, [base_unit, other_unit]))).convert("RGB"))
    assert out[66:76, 62:82].min() < 100  # protected member ink survives


def test_flat_erase_swallows_residue_of_the_erased_text_but_not_neighbours(tmp_path) -> None:
    # A descender poking below the tight erase quad belongs to the text being erased: its ink
    # component lies mostly INSIDE the quads and is painted over with the background colour
    # too. A neighbouring object grazing the quad edge (an icon) lies mostly OUTSIDE and
    # must survive untouched.
    input_path = tmp_path / "in.png"
    img = Image.new("RGB", (300, 160), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    for x in range(64, 148, 14):  # glyph strokes inside the quad
        draw.rectangle((x, 62, x + 6, 88), fill=(0, 0, 0))
    draw.rectangle((156, 62, 162, 100), fill=(0, 0, 0))  # stroke with a descender past y~94
    draw.rectangle((196, 60, 226, 90), fill=(0, 0, 0))   # icon: overlaps the quad edge only
    img.save(input_path)
    unit = {"translated_text": "aa",  # narrow, anchored left: x>150 keeps no tile
            "members": [{"cell_id": 1, "text": "WARNING", "translate": True,
                         "bbox": {"left": 40, "top": 58, "width": 150, "height": 34}}]}
    out = np.asarray(Image.open(BytesIO(
        render_translated_image(input_path, [unit])
    )).convert("RGB"))
    assert out[95:100, 156:163].min() > 200  # descender residue below the quad: swallowed
    assert out[62:88, 212:226].min() < 100   # the icon's outside part survives


def test_bullet_geometry_accepts_a_wide_flat_dash_but_not_line_tall_edge_ink() -> None:
    # A bullet is small in ink HEIGHT, not necessarily narrow: a dash is wide but flat and
    # must be kept (its width sat right on the old 0.4x-line-height cap, so quad-height
    # wobble flipped it per line). A line-TALL run (a coloured panel edge) is still no bullet.
    from app.replacement.layout.markers import _bullet_geometry

    def strokes(draw):  # glyph-like text ink: thin strokes, not a solid slab
        for x in range(40, 120, 12):
            draw.rectangle((x, 15, x + 5, 35), fill=(0, 0, 0))

    img = Image.new("RGB", (200, 60), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle((14, 28, 26, 31), fill=(0, 0, 0))  # dash: 13px wide, 4px tall on a 20px line
    strokes(draw)
    frame = (None, None, 12.0, 130.0, 14.0, 34.0)
    geometry = _bullet_geometry(img, frame, 0.0)
    assert geometry is not None
    assert abs(geometry[0] - 40) <= 2  # text starts past the dash
    assert abs(geometry[1] - 29.5) <= 2  # centred on the dash

    tall = Image.new("RGB", (200, 60), (255, 255, 255))
    draw = ImageDraw.Draw(tall)
    draw.rectangle((14, 12, 26, 36), fill=(200, 30, 30))  # line-tall edge ink at the dash's spot
    strokes(draw)
    edge_case = _bullet_geometry(tall, frame, 0.0)
    # The guard under test: line-tall ink is NOT the bullet, so the text anchor must not sit
    # right after it (the pre-existing dot rule may still anchor elsewhere in the strokes).
    assert edge_case is None or edge_case[0] > 41


def test_bullet_geometry_rejects_a_match_deep_inside_the_text() -> None:
    # The VLM flags ToC/numbered rows as bullets with the section number as marker, but the line
    # carries NO glyph: the scan then lands on a narrow letter followed by a word gap INSIDE the
    # text. Anchoring there erased only the line's tail — the words left of it stayed standing at
    # full size with the translation squeezed in behind. A candidate deeper than ~1.5x line height
    # past the box left is no bullet: give up instead.
    from app.replacement.layout.markers import _bullet_geometry

    img = Image.new("RGB", (400, 60), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    # a word of tightly-packed strokes at the left edge (intra-word gaps too small to qualify),
    # then a bullet-SHAPED blob deep in the line followed by a word gap and more text: the blob
    # passes the shape rules, so only the depth guard can reject it
    for x in range(14, 100, 6):
        draw.rectangle((x, 15, x + 4, 35), fill=(0, 0, 0))
    draw.rectangle((150, 21, 157, 28), fill=(0, 0, 0))  # 8x8 blob: compact, but deep in the text
    for x in range(190, 300, 6):
        draw.rectangle((x, 15, x + 4, 38), fill=(0, 0, 0))
    frame = (None, None, 12.0, 320.0, 14.0, 34.0)
    assert _bullet_geometry(img, frame, 0.0) is None

    # control: the same line WITH a real dash bullet at the edge still anchors right after it
    with_bullet = Image.new("RGB", (400, 60), (255, 255, 255))
    draw = ImageDraw.Draw(with_bullet)
    draw.rectangle((14, 28, 26, 31), fill=(0, 0, 0))  # the dash bullet
    for x in range(40, 130, 9):
        draw.rectangle((x, 15, x + 4, 35), fill=(0, 0, 0))
    geometry = _bullet_geometry(with_bullet, frame, 0.0)
    assert geometry is not None
    assert abs(geometry[0] - 40) <= 2


def test_bullet_geometry_accepts_a_square_glyph_but_not_a_letter_tall_block() -> None:
    # A square bullet ("■", ~half the line height in BOTH dimensions) is neither narrow (dot)
    # nor flat (dash): it needs the compact rule. A letter-sized block (one dimension ~0.6x+)
    # must still not pass, or any bold capital would read as a bullet.
    from app.replacement.layout.markers import _bullet_geometry

    img = Image.new("RGB", (300, 60), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle((14, 20, 26, 32), fill=(0, 30, 90))  # square: 13x13 on a 27px line
    for x in range(40, 130, 9):
        draw.rectangle((x, 12, x + 4, 38), fill=(0, 0, 0))  # letter strokes, line-tall
    frame = (None, None, 16.0, 280.0, 12.0, 39.0)  # box left overlaps the square (the clip case)
    geometry = _bullet_geometry(img, frame, 0.0)
    assert geometry is not None
    assert abs(geometry[0] - 40) <= 2  # text starts past the square, which stays untouched

    letter = Image.new("RGB", (300, 60), (255, 255, 255))
    draw = ImageDraw.Draw(letter)
    draw.rectangle((14, 14, 30, 36), fill=(0, 0, 0))  # letter-tall block: 17x23 on a 27px line
    for x in range(44, 130, 6):  # tightly packed: intra-word gaps must not qualify as bullet gaps
        draw.rectangle((x, 12, x + 4, 38), fill=(0, 0, 0))
    assert _bullet_geometry(letter, frame, 0.0) is None


def test_bullet_geometry_rejects_a_cap_height_first_letter() -> None:
    # A narrow CAP-HEIGHT first letter ("I"/"l"/"1") is not a bullet: it is ~0.1x wide but
    # ~0.8x tall. The old dot rule checked width only, took the "I" of a title for the glyph,
    # anchored the erase after it, and the letter survived as a stray "|" before the
    # re-rendered translation.
    from app.replacement.layout.markers import _bullet_geometry

    img = Image.new("RGB", (300, 60), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle((18, 15, 20, 33), fill=(0, 0, 0))  # "I": 3px wide, 19px tall on a 24px line
    for x in range(26, 130, 6):  # the rest of the word, tightly packed
        draw.rectangle((x, 15, x + 4, 33), fill=(0, 0, 0))
    frame = (None, None, 16.0, 280.0, 13.0, 37.0)
    assert _bullet_geometry(img, frame, 0.0) is None

    # control: a real dot bullet on the same line geometry still anchors
    dot = Image.new("RGB", (300, 60), (255, 255, 255))
    draw = ImageDraw.Draw(dot)
    draw.rectangle((16, 22, 23, 29), fill=(0, 0, 0))  # 8x8 dot
    for x in range(34, 130, 6):
        draw.rectangle((x, 15, x + 4, 33), fill=(0, 0, 0))
    geometry = _bullet_geometry(dot, frame, 0.0)
    assert geometry is not None
    assert abs(geometry[0] - 34) <= 2


def test_cell_marker_accepts_dotted_section_numbers_but_not_decimals() -> None:
    # OCR merges a ToC/outline row's section number into the title's cell ("3.4.1 Title"); the
    # number must take the redraw path (erase all, re-prepend) or the erase swallows it while the
    # translation drops it. A single-dot decimal is a price and must stay unmatched; so does a
    # plain two-level "2.3" without a trailing dot (same shape as a decimal — the named limit).
    from app.replacement.layout.markers import _cell_marker

    def unit(source):
        return {"source_text": source, "bullet_marker": ""}

    assert _cell_marker(unit("3.4.1 Gegevensgevoeligheid en privacy")) == "3.4.1"
    assert _cell_marker(unit("3.4. Regels voor het delen")) == "3.4."
    assert _cell_marker(unit("A.1.2 Bijlagesectie")) == "A.1.2"
    assert _cell_marker(unit("1. Inleiding")) == "1."
    assert _cell_marker(unit("1.69 korting")) is None
    assert _cell_marker(unit("2.3 Titel zonder punt")) is None
    assert _cell_marker(unit("3.4.1nogeenwoord")) is None  # no whitespace after -> not a marker


def test_clean_right_extension_stops_at_ink_protected_cells_and_the_cap() -> None:
    from app.replacement.layout.planning import _clean_right_extension

    base = np.full((60, 400, 3), 250, dtype=np.uint8)
    plane = {"frame": (None, None, 20.0, 100.0, 20.0, 40.0), "pad": 4.0}
    # Clean page to the right: capped at the plane's own width (80px), minus nothing else.
    assert _clean_right_extension(base, plane, (250, 250, 250), []) == pytest.approx(76.0)
    # Ink at x=140 stops the run there.
    inked = base.copy()
    inked[22:38, 140:150] = (30, 30, 30)
    assert 0 < _clean_right_extension(inked, plane, (250, 250, 250), []) <= 140 - 104
    # A protected cell (another unit's text) blocks even without visible ink.
    boxes = [{"left": 130, "top": 18, "width": 40, "height": 24}]
    assert _clean_right_extension(base, plane, (250, 250, 250), boxes) <= 130 - 104
    # A surface change (panel edge) reads as ink against the plane bg.
    panel = base.copy()
    panel[:, 150:] = (180, 200, 220)
    assert _clean_right_extension(panel, plane, (250, 250, 250), []) <= 150 - 104
    # Near the image border a ~1-em margin stays clear: a line ending at x=340 on a
    # 400px-wide page may extend at most to 400 - line_height(20) = 380.
    edge_plane = {"frame": (None, None, 290.0, 340.0, 20.0, 40.0), "pad": 4.0}
    assert _clean_right_extension(base, edge_plane, (250, 250, 250), []) <= 380 - 344


def test_width_fit_extend_keeps_size_on_free_ground_and_footprint_when_blocked(tmp_path) -> None:
    # A short list item whose translation is much longer: "footprint" condenses/shrinks it
    # into the original width; "extend" verifies the page right of it is clean background
    # and keeps the size, rendering wider. With an obstacle directly right of the line the
    # guards fail and "extend" must render exactly like "footprint".
    def make(path, with_obstacle):
        img = Image.new("RGB", (400, 80), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        for x in range(20, 66, 12):  # the original short "word": glyph-like strokes
            draw.rectangle((x, 30, x + 5, 50), fill=(0, 0, 0))
        if with_obstacle:
            draw.rectangle((74, 28, 96, 52), fill=(0, 0, 0))  # directly right of the line
        img.save(path)

    unit = {"translated_text": "langere vertaling",
            "members": [{"cell_id": 1, "text": "kort", "translate": True,
                         "bbox": {"left": 20, "top": 30, "width": 50, "height": 20}}]}

    free = tmp_path / "free.png"
    make(free, with_obstacle=False)
    footprint = render_translated_image(free, [dict(unit)], width_fit_mode="footprint")
    extend = render_translated_image(free, [dict(unit)], width_fit_mode="extend")
    assert footprint != extend
    ink_cols = lambda png: np.nonzero(  # noqa: E731
        (np.asarray(Image.open(BytesIO(png)).convert("L")) < 128).any(axis=0)
    )[0]
    # The extended render's ink reaches clearly past the original cell's right edge.
    assert ink_cols(extend).max() > ink_cols(footprint).max() + 10

    blocked = tmp_path / "blocked.png"
    make(blocked, with_obstacle=True)
    assert render_translated_image(
        blocked, [dict(unit)], width_fit_mode="extend"
    ) == render_translated_image(blocked, [dict(unit)], width_fit_mode="footprint")


def test_centered_group_lines_snap_to_one_axis_when_the_spread_is_noise(tmp_path) -> None:
    # Two centered lines of one element whose plane centres differ by a few px (quad
    # noise) must render on ONE axis; a genuinely offset second line (spread far past
    # the noise gate) must keep its own centre.
    def render_centers(offset_px):
        img = Image.new("RGB", (400, 160), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        for x in range(100, 296, 14):  # line 1: centre 200
            draw.rectangle((x, 30, x + 6, 50), fill=(0, 0, 0))
        for x in range(100 + offset_px, 296 + offset_px, 14):  # line 2: centre 200+offset
            draw.rectangle((x, 70, x + 6, 90), fill=(0, 0, 0))
        path = tmp_path / f"c{offset_px}.png"
        img.save(path)
        unit = {"translated_text": "eerste regel tweede regel", "alignment": "center",
                "block_id": 1, "level": "body",
                "members": [
                    {"cell_id": 1, "text": "eerste regel", "translate": True,
                     "bbox": {"left": 100, "top": 30, "width": 202, "height": 20}},
                    {"cell_id": 2, "text": "tweede regel", "translate": True,
                     "bbox": {"left": 100 + offset_px, "top": 70, "width": 202, "height": 20}},
                ]}
        from app.replacement.layout.groups import _groups
        from app.replacement.render import _plan_group
        from app.replacement.text.angle import _image_is_flat
        group = _groups([unit])[0]
        jobs = _plan_group(Image.open(path).convert("RGB"), group,
                           snap_horizontal=_image_is_flat([unit]), render_size_mode="median")
        centers = [
            (min(p[0] for p in j.dst_quad) + max(p[0] for p in j.dst_quad)) / 2
            for j in jobs if j.dst_quad
        ]
        assert len(centers) == 2
        return centers

    noisy = render_centers(3)  # 3px << 0.12 * target: noise -> one axis
    assert abs(noisy[0] - noisy[1]) < 0.6
    designed = render_centers(40)  # 40px >> gate: genuinely offset -> kept apart
    assert abs(designed[0] - designed[1]) > 20


def test_band_metric_shrinks_a_parenthesis_inflated_line_and_leaves_siblings(tmp_path) -> None:
    # Four sibling lines share one text-band height; one of them carries sparse tall ink
    # (parenthesis-like tips above and below) and an OCR box stretched to match. Under
    # "extent" that line renders visibly taller than its siblings; under "band" it sinks
    # to the document norm while an untouched sibling renders byte-identically to extent.
    img = Image.new("RGB", (420, 320), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    units = []
    for row, (top, inflated) in enumerate([(30, False), (100, False), (170, True), (240, False)]):
        for x in range(40, 200, 14):  # glyph strokes: band 24px tall
            draw.rectangle((x, top, x + 6, top + 24), fill=(0, 0, 0))
        bbox = {"left": 40, "top": top, "width": 166, "height": 25}
        if inflated:
            draw.rectangle((40, top - 12, 42, top + 36), fill=(0, 0, 0))   # sparse tall tip left
            draw.rectangle((204, top - 12, 206, top + 36), fill=(0, 0, 0))  # and right
            bbox = {"left": 38, "top": top - 12, "width": 172, "height": 49}
        units.append({"translated_text": f"regel {row} vertaald", "block_id": row, "level": "body",
                      "members": [{"cell_id": row, "text": f"regel {row}", "translate": True,
                                   "bbox": bbox}]})
    path = tmp_path / "band.png"
    img.save(path)

    extent = render_translated_image(path, [dict(u) for u in units], size_metric_mode="extent")
    band = render_translated_image(path, [dict(u) for u in units], size_metric_mode="band")
    assert extent != band

    def ink_height(png, y0, y1):
        arr = np.asarray(Image.open(BytesIO(png)).convert("L"))[y0:y1]
        rows = np.nonzero((arr < 128).any(axis=1))[0]
        return (rows.max() - rows.min() + 1) if len(rows) else 0

    # The inflated line (row band y 158..220) shrinks under "band"; a sibling stays put.
    assert ink_height(band, 150, 225) < ink_height(extent, 150, 225)
    assert ink_height(band, 20, 70) == ink_height(extent, 20, 70)


def _tilted_line_unit(uid, y, deg, w=400.0, h=30.0, cx=500.0):
    # One single-cell text line: a w x h quad centred at (cx, y), rotated by deg.
    ux, uy = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    vx, vy = -uy, ux

    def pt(dx, dy):
        return {"x": cx + dx * ux + dy * vx, "y": y + dx * uy + dy * vy}

    poly = [pt(-w / 2, -h / 2), pt(w / 2, -h / 2), pt(w / 2, h / 2), pt(-w / 2, h / 2)]
    xs = [p["x"] for p in poly]
    ys = [p["y"] for p in poly]
    bbox = {"left": min(xs), "top": min(ys), "width": max(xs) - min(xs), "height": max(ys) - min(ys)}
    return {"translated_text": "vertaalde regel", "block_id": uid, "level": "body",
            "members": [{"cell_id": uid, "text": "een regel", "translate": True,
                         "bbox": bbox, "polygon": poly}]}


def test_document_angle_field_recovers_a_linear_gradient_and_gates_junk() -> None:
    from app.replacement.text.angle import _document_angle_field

    # A tilted sign: line angles follow angle(y) = 2 + 6 * y/1000 (the measured shape).
    gradient = [_tilted_line_unit(i, y, 2.0 + 6.0 * y / 1000.0)
                for i, y in enumerate(range(100, 1600, 200))]
    field = _document_angle_field(gradient)
    assert field is not None
    slope, intercept = field
    assert abs(slope - 0.006) < 0.0005
    assert abs(intercept - 2.0) < 0.5

    # Scattered angles (product-photo archetype: each line its own direction) must be
    # rejected by the residual gate, not fitted.
    scattered = [_tilted_line_unit(i, y, 8.0 if i % 2 else -8.0)
                 for i, y in enumerate(range(100, 1600, 200))]
    assert _document_angle_field(scattered) is None

    # Too little y-range to define a slope.
    narrow = [_tilted_line_unit(i, 400.0 + i * 20.0, 6.0) for i in range(8)]
    assert _document_angle_field(narrow) is None

    # A flat image never gets a field (its per-line angles are OCR noise).
    flat = [_tilted_line_unit(i, y, 0.5) for i, y in enumerate(range(100, 1600, 200))]
    assert _document_angle_field(flat) is None


def _sized_unit(uid, pt, height, width=300):
    top = uid * 100
    poly = [{"x": 0, "y": top}, {"x": width, "y": top},
            {"x": width, "y": top + height}, {"x": 0, "y": top + height}]
    return {"font_size": pt, "block_id": uid, "level": "body", "translated_text": "x",
            "members": [{"cell_id": uid, "text": "x", "translate": True,
                         "bbox": {"left": 0, "top": top, "width": width, "height": height},
                         "polygon": poly}]}


def test_size_cohorts_snap_on_agreement_split_by_pt_and_skip_on_disagreement() -> None:
    from app.replacement.text.size import _document_size_cohorts

    # One pt the VLM gave many elements, OCR heights tight -> snap to the cohort median.
    tight = [_sized_unit(i, 24, h) for i, h in enumerate([55, 58, 60, 61, 62])]
    cohorts = _document_size_cohorts(tight)
    assert 24 in cohorts and 58 <= cohorts[24] <= 62

    # Two pt cohorts (a real size difference the VLM saw): each tight -> both kept, distinct.
    mixed = ([_sized_unit(i, 16, h) for i, h in enumerate([50, 52, 51, 49])] +
             [_sized_unit(10 + i, 24, h) for i, h in enumerate([66, 64, 68])])
    cm = _document_size_cohorts(mixed)
    assert set(cm) == {16, 24} and cm[16] < cm[24]

    # VLM claims one pt but OCR strongly disagrees -> not snapped (the equal claim is wrong).
    noisy = [_sized_unit(i, 12, h) for i, h in enumerate([30, 45, 60, 70])]
    assert 12 not in _document_size_cohorts(noisy)

    # Too few members to trust a cohort -> omitted.
    assert _document_size_cohorts([_sized_unit(i, 18, h) for i, h in enumerate([40, 41])]) == {}


def test_size_cohort_mode_sizes_a_short_sibling_up_to_the_cohort(tmp_path) -> None:
    # Three list items the VLM labelled one pt; OCR measured the third shorter. Under "vlm" the
    # short item renders at the cohort size (taller) instead of its own small measurement.
    img = Image.new("RGB", (400, 360), (255, 255, 255))
    img.save(tmp_path / "list.png")
    units = [_sized_unit(i, 24, h) for i, h in enumerate([60, 58, 44])]
    for u in units:
        u["translated_text"] = "vertaalde regel"

    def ink_h(png, y0, y1):
        arr = np.asarray(Image.open(BytesIO(png)).convert("L"))[y0:y1]
        rows = np.nonzero((arr < 128).any(axis=1))[0]
        return (rows.max() - rows.min() + 1) if len(rows) else 0

    off = render_translated_image(tmp_path / "list.png", [dict(u) for u in units], size_cohort_mode="off")
    vlm = render_translated_image(tmp_path / "list.png", [dict(u) for u in units], size_cohort_mode="vlm")
    assert off != vlm
    # the third item sits at y ~200..260; it renders taller under "vlm"
    assert ink_h(vlm, 195, 300) > ink_h(off, 195, 300)


def test_plan_group_reads_its_angle_from_the_document_field(tmp_path) -> None:
    # A one-line group whose own quad reads +2deg on a document whose field says +8deg
    # at that y must render at the field angle; without a field it trusts its own quad.
    img = Image.new("RGB", (1000, 800), (255, 255, 255))
    path = tmp_path / "field.png"
    img.save(path)
    unit = _tilted_line_unit(1, 400.0, 2.0)

    def top_edge_angle(angle_field):
        jobs = _plan_group(Image.open(path).convert("RGB"), [dict(unit)],
                           snap_horizontal=False, angle_field=angle_field)
        (job,) = [j for j in jobs if j.dst_quad]
        (x0, y0), (x1, y1) = job.dst_quad[0], job.dst_quad[1]
        return math.degrees(math.atan2(y1 - y0, x1 - x0))

    assert abs(top_edge_angle((0.0, 8.0)) - 8.0) < 0.3
    assert abs(top_edge_angle(None) - 2.0) < 0.3


def test_inpaint_budget_scale_caps_the_model_input_area() -> None:
    assert budget_scale(1000, 1000, 1_500_000) == 1.0
    scale = budget_scale(4096, 3072, 1_500_000)
    assert 0.0 < scale < 1.0
    assert (4096 * scale) * (3072 * scale) <= 1_500_000 + 1  # float slack


def test_inpaint_work_scale_upscales_tiny_crops_but_caps_at_2x_and_budget() -> None:
    assert work_scale(2048, 1536, 1_500_000) < 1.0  # over budget: downscale wins
    assert work_scale(800, 600, 1_500_000) == 1.0  # comfortable size: untouched
    assert work_scale(307, 164, 1_500_000) == pytest.approx(320 / 164)  # tiny thumbnail: ~2x
    assert work_scale(400, 100, 1_500_000) == 2.0  # thin strip: the 2x cap binds
    assert work_scale(400, 300, 1_500_000) == pytest.approx(320 / 300)  # just short: exact ratio
    assert work_scale(300, 200, 90_000) == pytest.approx(math.sqrt(90_000 / 60_000))  # budget caps the upscale


def test_inpaint_ground_router_keeps_flat_on_designed_ground_and_models_gradients() -> None:
    # Designed ground: a solid band with the line fully inside it — every side band is
    # constant along the line, so the flat paint is right and the model must stay out.
    # Textured ground: shading drifting along the line scars under a flat rectangle and
    # must go to the model. The job's bg is the sampled ground (mid-gradient for the
    # graded case), as the pipeline's colour sampling would produce.
    from app.replacement.ground.erase import _ellipse
    from app.replacement.ground.erase import _GROUND_RING_INNER_PX
    from app.replacement.ground.erase import _needs_model_fill
    from app.replacement.jobs import _Job

    quad = [(60, 60), (240, 60), (240, 90), (60, 90)]
    occupied = np.zeros((160, 300), dtype=np.uint8)
    cv2.fillPoly(occupied, [np.asarray(quad, dtype=np.int32)], 255)
    occupied = cv2.dilate(occupied, _ellipse(_GROUND_RING_INNER_PX))

    designed = np.zeros((160, 300, 3), dtype=np.uint8)
    designed[:, :] = (220, 60, 30)  # the whole ring lives inside one solid band
    job = _Job(erase_quads=[quad], bg_color=(220, 60, 30), tile=None, dst_quad=None)
    assert not _needs_model_fill(designed, job, occupied)

    graded = np.zeros((160, 300, 3), dtype=np.uint8)
    graded[:, :] = np.linspace(90, 200, 300).astype(np.uint8)[None, :, None]  # drift along the line
    job = _Job(erase_quads=[quad], bg_color=(145, 145, 145), tile=None, dst_quad=None)
    assert _needs_model_fill(graded, job, occupied)


def test_inpaint_ground_router_ignores_a_designed_boundary_the_fill_never_touches() -> None:
    # A solid panel with a contrasting graphic just outside the line (inside the ring): the
    # boundary is another surface the flat fill never touches — counting it routed solid-panel
    # lines to the model, whose near-total-hole fill then smeared that very graphic. The line's
    # OWN ground is flat, so the router must keep the flat paint.
    from app.replacement.ground.erase import _ellipse
    from app.replacement.ground.erase import _GROUND_RING_INNER_PX
    from app.replacement.ground.erase import _needs_model_fill
    from app.replacement.jobs import _Job

    quad = [(60, 60), (240, 60), (240, 90), (60, 90)]
    job = _Job(erase_quads=[quad], bg_color=(180, 40, 50), tile=None, dst_quad=None)
    occupied = np.zeros((160, 300), dtype=np.uint8)
    cv2.fillPoly(occupied, [np.asarray(quad, dtype=np.int32)], 255)
    occupied = cv2.dilate(occupied, _ellipse(_GROUND_RING_INNER_PX))

    panel = np.zeros((160, 300, 3), dtype=np.uint8)
    panel[:, :] = (180, 40, 50)               # solid red panel
    panel[:, 0:52] = (250, 250, 250)          # a white graphic just left of the line
    panel[40:52, :] = (60, 90, 200)           # a blue rule just above it
    assert not _needs_model_fill(panel, job, occupied)


def test_inpaint_ground_router_models_a_cross_gradient_on_dark_ground_only() -> None:
    # A smooth illumination gradient ACROSS the line (top vs bottom): per-band tests never
    # see it. On dark ground the same absolute drift is proportionally huge (Weber) and a
    # flat plate shows — model. On bright ground it stays well under the relative threshold
    # — flat.
    from app.replacement.ground.erase import _ellipse
    from app.replacement.ground.erase import _GROUND_RING_INNER_PX
    from app.replacement.ground.erase import _needs_model_fill
    from app.replacement.jobs import _Job

    # A tilted line: two word quads at different heights, so the line's bbox is tall and the
    # above/below bands sit ~160px apart — a shallow smooth gradient (0.35 value/px, like a
    # lit panel) accumulates ~55 of drift between them while staying invisible to the
    # per-band and texture tests.
    quads = [[(60, 60), (160, 60), (160, 90), (60, 90)],
             [(160, 160), (260, 160), (260, 190), (160, 190)]]
    occupied = np.zeros((400, 320), dtype=np.uint8)
    for quad in quads:
        cv2.fillPoly(occupied, [np.asarray(quad, dtype=np.int32)], 255)
    occupied = cv2.dilate(occupied, _ellipse(_GROUND_RING_INNER_PX))

    def vertical_gradient(top, bottom):
        img = np.zeros((400, 320, 3), dtype=np.uint8)
        img[:, :] = np.linspace(top, bottom, 400).astype(np.uint8)[:, None, None]
        return img

    dark = vertical_gradient(30, 170)   # cross ~55 on luma ~75: a plate would show
    job = _Job(erase_quads=quads, bg_color=(75, 75, 75), tile=None, dst_quad=None)
    assert _needs_model_fill(dark, job, occupied)

    bright = vertical_gradient(115, 255)  # same absolute drift on luma ~160: stays flat
    job = _Job(erase_quads=quads, bg_color=(160, 160, 160), tile=None, dst_quad=None)
    assert not _needs_model_fill(bright, job, occupied)


def test_inpaint_border_pads_only_where_the_hole_touches_a_mirrorable_border() -> None:
    # Corner case: the mask reaches the top-left image corner and the strip is plain
    # background -> mirror-pad top+left. Counter-cases: a near-fully masked strip or a
    # featureful strip (a mounting hole: mirroring it makes a pair the model repeats as
    # a period) must NOT be mirrored.
    plain = np.full((100, 200, 3), 245, dtype=np.uint8)
    crop_mask = np.zeros((100, 200), dtype=np.uint8)
    crop_mask[0:20, 0:60] = 255  # a line clipped into the top-left corner
    pads = border_pads(plain, crop_mask, (0, 0, 200, 100), (400, 400))
    assert pads[0] > 0 and pads[2] > 0  # top and left mirrored
    assert pads[1] == 0 and pads[3] == 0  # bottom/right: crop not on the image border

    full = np.full((100, 200), 255, dtype=np.uint8)  # near-fully masked strip: no anchor
    assert border_pads(plain, full, (0, 0, 200, 100), (100, 200)) == (0, 0, 0, 0)

    featureful = plain.copy()
    featureful[4:12, 100:116] = (220, 60, 30)  # a distinctive UNMASKED mark in the top strip
    assert border_pads(featureful, crop_mask, (0, 0, 200, 100), (400, 400))[0] == 0

    inset = np.zeros((100, 200), dtype=np.uint8)
    inset[40:60, 40:160] = 255  # mask nowhere near the crop edge
    assert border_pads(plain, inset, (0, 0, 200, 100), (100, 200)) == (0, 0, 0, 0)


def test_inpaint_context_window_adds_margin_and_clamps_to_the_image() -> None:
    mask = np.zeros((200, 300), dtype=np.uint8)
    assert context_window(mask) is None
    mask[80:100, 10:120] = 255  # touches the left margin zone
    x0, y0, x1, y1 = context_window(mask)
    assert x0 == 0  # clamped: 10 - margin(>=32) < 0
    assert y0 <= 80 - 32 and y1 >= 100 + 32  # at least the minimum context ring
    assert x1 >= 120 + 32 and x1 <= 300


def test_inpaint_mode_reconstructs_the_gradient_the_flat_fill_would_flatten(tmp_path) -> None:
    # The acceptance direction for the Tier-2 fill (2026-07-06 verdict): reconstruct the
    # ground instead of painting one colour. On a vertical gradient the flat fill leaves a
    # constant band; LaMa must continue the gradient through the erased region. Model-based
    # and GPU-only, so this runs where the service runs (dc1) and skips elsewhere.
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("inpaint fill is GPU-only")
    from app.core.config import load_settings
    from pathlib import Path

    if not Path(load_settings().inpaint.model_path).expanduser().exists():
        pytest.skip("no LaMa checkpoint on this machine")

    input_path = tmp_path / "in.png"
    img = Image.new("RGB", (300, 160), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    for y in range(160):
        shade = 60 + y
        draw.line([(0, y), (300, y)], fill=(shade, shade, shade))
    for x in range(44, 196, 14):
        draw.rectangle((x, 60, x + 6, 90), fill=(255, 255, 255))
    img.save(input_path)
    unit = {"translated_text": "aa",  # narrow, anchored left: x>150 keeps no tile
            "members": [{"cell_id": 1, "text": "WARNING", "translate": True,
                         "bbox": {"left": 40, "top": 58, "width": 160, "height": 34}}]}

    flat = np.asarray(Image.open(BytesIO(
        render_translated_image(input_path, [dict(unit)], erase_fill_mode="flat")
    )).convert("RGB")).astype(np.int16)
    inpaint = np.asarray(Image.open(BytesIO(
        render_translated_image(input_path, [dict(unit)], erase_fill_mode="inpaint")
    )).convert("RGB")).astype(np.int16)

    # Sample the erased band right of the rendered tile. Flat paints it one colour; the
    # reconstruction must follow the gradient (rows keep their own shade, within noise).
    rows = np.arange(64, 86)
    expected = 60 + rows
    filled_rows = inpaint[64:86, 150:190].mean(axis=(1, 2))
    flat_rows = flat[64:86, 150:190].mean(axis=(1, 2))
    assert np.ptp(flat_rows) < 4  # the flat band really is flat, or the test proves nothing
    assert np.abs(filled_rows - expected).mean() < 8
    assert np.ptp(filled_rows) > 10  # follows the gradient's spread, not one shade


def test_company_member_with_preserve_dropped_field_stays_out_of_the_spend_cell() -> None:
    # When the company field is preserve-dropped from the pairs ("Amazon" -> "Amazon", unchanged),
    # the company member's best remaining match is the LONG spend field — on scattered 1-2 char
    # noise it reached 0.67 and joined the spend cell: the row erase then swallowed the company
    # name and the spend text rendered at the company's column. First/last table rows (short
    # names: Amazon 0.67, Unilever 0.62) hit this; the middle rows stayed under the threshold.
    unit = {
        "field_translations": [("$23,5 miljard aan advertising & promotional costs in 2025",
                                "$23,5 miljard aan reclame- & promotiekosten in 2025")],
        "members": [
            {"text": "Amazon", "translate": True, "bbox": {"left": 50, "top": 388, "width": 160, "height": 58}},
            {"text": "$23,5 miljard a", "translate": True, "bbox": {"left": 656, "top": 385, "width": 300, "height": 58}},
            {"text": "advertising &", "translate": True, "bbox": {"left": 655, "top": 470, "width": 300, "height": 56}},
            {"text": "promotional co", "translate": True, "bbox": {"left": 656, "top": 554, "width": 300, "height": 51}},
        ],
    }
    cells = _split_table_row(unit)
    assert cells is not None and len(cells) == 1
    member_texts = [m["text"] for m in cells[0]["members"]]
    assert "Amazon" not in member_texts  # unplaced -> original pixels stay at column 1
    assert "$23,5 miljard a" in member_texts  # the true wrapped fragments still bind


def test_baseline_angle_is_exact_for_parallel_lines_of_unequal_length() -> None:
    # A long line above a short last line, both at a true 6°: with only y de-meaned the shared
    # intercept dragged the fit shallow (~4.5°); de-meaning x per cluster makes it exact.
    slope = math.tan(math.radians(6.0))

    def quad(cx: float, cy: float) -> list[tuple[float, float]]:
        return [(cx - 5, cy - 5), (cx + 5, cy - 5), (cx + 5, cy + 5), (cx - 5, cy + 5)]

    long_line = [quad(x, x * slope) for x in (0, 250, 500, 750, 1000)]
    short_line = [quad(x, 80 + x * slope) for x in (0, 150, 300)]
    angle = _baseline_angle([long_line, short_line], fallback=0.0)
    assert abs(angle - 6.0) < 0.05


def test_group_size_mode_selects_min_or_median() -> None:
    # "min": one under-measured (lowercase) line drags the block to its size. "median": the
    # block keeps the element's typical size. Unknown modes fall back to "min".
    planes = [{"target": 16}, {"target": 16}, {"target": 10}]
    assert _group_size(planes, "min") == 10
    assert _group_size(planes, "median") == 16
    assert _group_size(planes, "auto") == 10  # unknown -> the safe default


def test_lone_fullwidth_punctuation_does_not_make_a_line_cjk() -> None:
    # One retained "！" must not shrink the group to the CJK ratio or reroute the font.
    assert fold_lone_fullwidth_punctuation("DANGER！") == "DANGER!"
    assert not is_cjk_text(fold_lone_fullwidth_punctuation("DANGER！"))
    # Real CJK text keeps its fullwidth punctuation.
    assert fold_lone_fullwidth_punctuation("小心！") == "小心！"


def test_single_cjk_character_translation_still_renders(tmp_path) -> None:
    # "PUSH" -> "推": one han character is a full word, not OCR noise — the original must be
    # erased and the translation drawn, not silently skipped.
    input_path = tmp_path / "input.png"
    img = Image.new("RGB", (200, 100), (255, 255, 255))
    ImageDraw.Draw(img).text((30, 40), "PUSH", fill=(0, 0, 0))
    img.save(input_path)
    units = [{
        "translated_text": "推",
        "members": [{"cell_id": 1, "text": "PUSH", "translate": True,
                     "bbox": {"left": 25, "top": 35, "width": 60, "height": 25}}],
    }]
    png = render_translated_image(input_path, units)
    out = Image.open(BytesIO(png)).convert("RGB")
    before = np.asarray(Image.open(input_path).convert("RGB").crop((25, 35, 85, 60)))
    after = np.asarray(out.crop((25, 35, 85, 60)))
    assert not np.array_equal(before, after)


def test_top_stray_redundant_word_does_not_starve_the_body_render() -> None:
    # A neighbouring element's word (a heading's "dieren", shared with the body below) was aligned
    # into the body and OCR placed it on its own tiny line ABOVE the real text. That sliver plane
    # starved the unit's width fit so the whole long translation fell under the condense floor and
    # rendered NOTHING, leaving the original Dutch showing. The top stray line must be dropped so
    # the body renders on its two real lines.
    base = Image.new("RGB", (360, 200), (40, 60, 160))
    line1 = {"text": "Ze worden hier opdringerig", "translate": True,
             "bbox": {"left": 20, "top": 80, "width": 300, "height": 18}}
    line2 = {"text": "de betreffende dieren te slachten", "translate": True,
             "bbox": {"left": 20, "top": 106, "width": 300, "height": 18}}
    stray = {"text": "dieren", "translate": True,  # redundant (carried by line2) + tiny line above
             "bbox": {"left": 150, "top": 58, "width": 34, "height": 16}}
    translated = "This makes them pushy and they will bite, which forces us to slaughter the animals"
    clean = {"translated_text": translated, "members": [line1, line2]}
    strayed = {"translated_text": translated, "members": [stray, line1, line2]}

    clean_jobs = _plan_group(base.copy(), [clean])
    strayed_jobs = _plan_group(base.copy(), [strayed])

    assert len(clean_jobs) == 2  # two real text lines
    assert len(strayed_jobs) == 2  # stray dropped -> body still renders, not starved to nothing


def test_reproduced_in_keeps_a_token_absent_from_the_translation() -> None:
    # A price the translation does not contain (most receipt/menu prices) stays preserved.
    assert _reproduced_in({"text": "8,50"}, "French fish soup with fennel") is False


def test_sample_region_colors_dark_bg_gets_white_fg() -> None:
    image = Image.new("RGB", (40, 40), (20, 40, 160))  # dark blue
    bg, fg = sample_region_colors(image, {"left": 0, "top": 0, "width": 40, "height": 40})
    assert bg == (20, 40, 160)
    assert fg == (255, 255, 255)  # low luminance -> white text


def test_sample_region_colors_light_bg_gets_black_fg() -> None:
    image = Image.new("RGB", (40, 40), (240, 240, 240))
    bg, fg = sample_region_colors(image, {"left": 0, "top": 0, "width": 40, "height": 40})
    assert bg == (240, 240, 240)
    assert fg == (0, 0, 0)


def test_chromatic_ink_renders_in_its_measured_color() -> None:
    image = Image.new("RGB", (40, 40), (255, 255, 255))
    image.paste((180, 30, 40), (10, 10, 30, 30))  # brand-red glyph block
    bg, fg = sample_region_colors(image, {"left": 0, "top": 0, "width": 40, "height": 40})
    assert bg == (255, 255, 255)
    assert fg == (180, 30, 40)


def test_antialiased_edge_majority_does_not_wash_the_core_ink() -> None:
    # Thin strokes: washed edge pixels OUTNUMBER the pure stroke cores (360 vs 270 here). A
    # median over every deviating pixel would land on the washed edge colour; core selection
    # is relative to the polarity extreme, so the pure ink survives.
    image = Image.new("RGB", (60, 60), (255, 255, 255))
    for y in (10, 25, 40):
        image.paste((165, 177, 233), (14, y, 44, y + 7))      # washed stroke envelope
        image.paste((30, 60, 200), (14, y + 2, 44, y + 5))    # pure stroke core
    _bg, fg = sample_region_colors(image, {"left": 0, "top": 0, "width": 60, "height": 60})
    assert fg == (30, 60, 200)


def test_achromatic_grey_ink_keeps_its_measured_level() -> None:
    # Mid-grey ink (shadowed receipt print) renders as soft neutral grey at its own level,
    # not hard black — the tint is dropped, the level survives.
    image = Image.new("RGB", (40, 40), (255, 255, 255))
    image.paste((70, 70, 80), (10, 10, 30, 30))
    _bg, fg = sample_region_colors(image, {"left": 0, "top": 0, "width": 40, "height": 40})
    assert fg == (71, 71, 71)  # luminance of the measured ink, neutralised


def test_near_black_document_ink_still_snaps_to_pure_black() -> None:
    # Laser-print black measures ~L20: within the pole margin, so clean documents keep
    # crisp pure black instead of a pointless (20,20,20).
    image = Image.new("RGB", (40, 40), (255, 255, 255))
    image.paste((20, 20, 25), (10, 10, 30, 30))
    _bg, fg = sample_region_colors(image, {"left": 0, "top": 0, "width": 40, "height": 40})
    assert fg == (0, 0, 0)


def test_dark_blob_in_the_cell_cannot_hijack_light_grey_text_to_black() -> None:
    # Reply-bar archetype: light-grey placeholder text shares its bbox with a dark avatar
    # blob (an 11% pixel minority in the real cell). The blob is thick, text strokes are
    # thin; the shape filter drops the blob so the text keeps its own soft grey level.
    image = Image.new("RGB", (120, 60), (255, 255, 255))
    for y in (18, 28, 38):  # three thin grey "text" strokes
        image.paste((143, 143, 143), (34, y, 110, y + 3))
    image.paste((20, 20, 20), (6, 16, 30, 40))  # dark avatar blob
    _bg, fg = sample_region_colors(image, {"left": 0, "top": 0, "width": 120, "height": 60})
    assert fg == (143, 143, 143)


def test_bimodal_ink_votes_by_mass_not_by_extreme() -> None:
    # Reply-bar "+" archetype: a bold BLACK glyph (stroke-shaped, so the blob filter keeps
    # it) beside light-grey placeholder text. Two real inks in one cell: the 76% grey mass
    # wins and sets the level; the black minority no longer drags the line to (0,0,0).
    image = Image.new("RGB", (120, 60), (255, 255, 255))
    for y in (18, 28, 38):  # grey placeholder strokes (~76% of the ink mass)
        image.paste((143, 143, 143), (40, y, 116, y + 3))
    image.paste((10, 10, 10), (10, 27, 34, 31))  # black "+": horizontal bar
    image.paste((10, 10, 10), (20, 17, 24, 41))  # black "+": vertical bar
    _bg, fg = sample_region_colors(image, {"left": 0, "top": 0, "width": 120, "height": 60})
    assert fg == (143, 143, 143)


def test_bg_gradient_blob_is_dropped_even_as_the_ink_majority() -> None:
    # Shiny dark box: the border ring samples the darkest corner, so a brighter box-gradient
    # PATCH (5x the glyph mass) also "deviates" and would drag any population vote onto the
    # box. The patch is a thick blob, the glyphs are thin strokes: the shape filter drops the
    # patch whole and the dim-white text renders at its own soft level.
    image = Image.new("RGB", (100, 60), (20, 20, 20))
    image.paste((80, 60, 45), (10, 10, 46, 46))  # glossy gradient patch (blob, majority mass)
    for y in (18, 28, 38):                       # dim-white glyph strokes beside it
        image.paste((200, 200, 200), (52, y, 94, y + 3))
    _bg, fg = sample_region_colors(image, {"left": 0, "top": 0, "width": 100, "height": 60})
    assert fg == (200, 200, 200)


def test_badge_number_and_arrow_unprinted_by_members_are_stripped() -> None:
    # "2 Tile Tabs" / "4 → Go To Arrow": the VLM transcribes the badge digit (and leader
    # arrow) into the heading's hint line while the badge graphic stays in the image — the
    # translation must not print them again next to the intact badge.
    unit = {"level": "header", "members": [{"text": "Tile Tabs"}]}
    assert _strip_unprinted_lead("2 Tegel-tabbladen", unit) == "Tegel-tabbladen"
    unit = {"level": "header", "members": [{"text": "Go To Arrow"}]}
    assert _strip_unprinted_lead("4 → Ga Naar Pijl", unit) == "Ga Naar Pijl"


def test_printed_enumerator_and_body_numbers_stay() -> None:
    # A number a member actually prints stays (a step heading whose big digit OCR'd), and
    # body-level text is never touched (a translator may digitise a written number there).
    unit = {"level": "header", "members": [{"text": "2"}, {"text": "Lever je pakket in"}]}
    assert _strip_unprinted_lead("2 Lever je pakket in", unit) == "2 Lever je pakket in"
    unit = {"level": "body", "members": [{"text": "twee weken de tijd"}]}
    assert _strip_unprinted_lead("2 weken de tijd", unit) == "2 weken de tijd"


def _frame(xmin: float, xmax: float, ymin: float, ymax: float) -> tuple:
    return ((1.0, 0.0), (0.0, 1.0), xmin, xmax, ymin, ymax)


def test_planes_justified_needs_flush_edges_last_line_exempt() -> None:
    # A LaTeX paragraph: left edges flush, right edges flush except the short last line.
    flush = [{"frame": _frame(100, 1450 + d, 30 * i, 30 * i + 24)} for i, d in enumerate((0, 3, -2, 4))]
    flush.append({"frame": _frame(100, 900, 150, 174)})  # short last line
    assert _planes_justified(flush) is True
    # Ragged-right body text: right edges scatter by whole words.
    ragged = [{"frame": _frame(100, 1450 - 90 * i, 30 * i, 30 * i + 24)} for i in range(5)]
    assert _planes_justified(ragged) is False
    # Too few lines to trust the evidence.
    assert _planes_justified(flush[:3]) is False


def test_planes_justified_allows_first_line_indent_and_hyphen_outlier() -> None:
    # The LaTeX-abstract archetype: line 0 indented +43px (right edge still flush), one
    # body line -22px short on the right (clipped hyphen), the rest flush within ±1px.
    planes = [{"frame": _frame(143, 1450, 0, 27)}]                       # indented first line
    planes += [{"frame": _frame(100 + d, 1450 + r, 30 * (i + 1), 30 * (i + 1) + 27)}
               for i, (d, r) in enumerate(((0, 0), (1, -22), (-1, 1), (0, 0), (1, -1)))]
    planes.append({"frame": _frame(100, 700, 210, 237)})                  # short last line
    assert _planes_justified(planes) is True
    # A LEFTWARD first-line shift is not an indent — genuinely ragged.
    shifted = [dict(p) for p in planes]
    shifted[0] = {"frame": _frame(30, 1450, 0, 27)}
    assert _planes_justified(shifted) is False


def test_justify_feasibility_mirrors_the_gap_bounds() -> None:
    # The block-consistency gate mirrors the drawing: a modest leftover is feasible,
    # doubling a two-word line is not, one word never is — and a slightly TOO-LONG line
    # counts as feasible (it squeezes onto the margin, ending flush all the same).
    font = load_font(20, "woorden hier")
    line = "deze regel heeft vijf woorden"
    assert _justify_feasible(font, line, font.getlength(line) * 1.07) is True
    assert _justify_feasible(font, "twee woorden", 2.0 * font.getlength("twee woorden")) is False
    assert _justify_feasible(font, "woord", 500.0) is False
    assert _justify_feasible(font, line, font.getlength(line) * 0.96) is True   # squeezable
    assert _justify_feasible(font, line, font.getlength(line) * 0.85) is False  # beyond floor


def test_justify_overhang_squeezes_to_margin_within_floor() -> None:
    # Inside a justified block a slack-packed line would poke out of the flush margin:
    # within the floor it squeezes onto the margin; a pathological overhang keeps its
    # width; a fitting line is untouched.
    assert _justify_overhang_width(1040.0, 1.0, 1000.0) == 1000   # 4% overhang -> squeeze
    assert _justify_overhang_width(1120.0, 1.0, 1000.0) is None   # 12% -> leave (floor)
    assert _justify_overhang_width(980.0, 1.0, 1000.0) is None    # fits -> untouched
    assert _justify_overhang_width(1300.0, 0.8, 1000.0) == 1000   # condense counts first


def test_justified_line_spans_target_and_respects_caps() -> None:
    font = load_font(20, "woorden hier")
    line = "deze regel heeft vijf woorden"
    natural = font.getlength(line)
    # A realistic leftover (the balanced wrap keeps lines near-full): ~7% over four gaps.
    image = _justified_text_image(font, line, natural * 1.07, (0, 0, 0), 26)
    assert image is not None and image.width == int(round(natural * 1.07))
    # Cap: stretching a two-word line across double its width would gape — fall back.
    assert _justified_text_image(font, "twee woorden", 2.0 * font.getlength("twee woorden"), (0, 0, 0), 26) is None
    # A single word (or spaceless CJK line) has no gaps to spread.
    assert _justified_text_image(font, "woord", 500.0, (0, 0, 0), 26) is None


def test_fit_text_single_line_fits_box() -> None:
    fitted = fit_text("Exit", max_width=200, max_height=40, wrap=False)
    assert fitted.lines == ["Exit"]
    assert fitted.line_height <= 40
    assert fitted.font.getlength("Exit") <= 200


def test_fit_text_wraps_long_text_in_narrow_box() -> None:
    text = "You are a guest in the habitat of horses"
    fitted = fit_text(text, max_width=120, max_height=400, wrap=True)
    assert len(fitted.lines) > 1
    assert all(fitted.font.getlength(line) <= 120 for line in fitted.lines)


def test_render_translated_image_draws_into_translatable_member(tmp_path) -> None:
    input_path = tmp_path / "input.png"
    Image.new("RGB", (200, 100), (255, 255, 255)).save(input_path)
    units = [
        {
            "translated_text": "Hallo",
            "members": [
                {"cell_id": 1, "text": "Hi", "translate": True,
                 "bbox": {"left": 20, "top": 30, "width": 140, "height": 40}},
            ],
        }
    ]

    png = render_translated_image(input_path, units)
    out = Image.open(BytesIO(png)).convert("RGB")
    assert out.size == (200, 100)

    region = np.asarray(out.crop((20, 30, 160, 70)))
    # text was drawn (black on white) -> the region is no longer all-white
    assert region.min() < 128


def test_render_reflows_a_block_over_its_original_planes(tmp_path) -> None:
    # Two lines of one block (a wrapped dish): the joined translation re-breaks freely;
    # line 2 may exceed its own plane width (up to the block's widest plane), so the
    # long second translation still lands readable on plane 2 instead of shrinking alone.
    input_path = tmp_path / "input.png"
    Image.new("RGB", (400, 120), (255, 255, 255)).save(input_path)
    units = [
        {"translated_text": "Grilled steak with", "block_id": 7, "level": "body",
         "members": [
             {"cell_id": 1, "text": "Biefstuk van de grill", "translate": True,
              "bbox": {"left": 20, "top": 10, "width": 300, "height": 30}}]},
        {"translated_text": "red wine jus and fries on top", "block_id": 7, "level": "body",
         "members": [
             {"cell_id": 2, "text": "jus en frites", "translate": True,
              "bbox": {"left": 20, "top": 50, "width": 120, "height": 30}}]},
    ]
    png = render_translated_image(input_path, units)
    out = Image.open(BytesIO(png)).convert("RGB")
    line1 = np.asarray(out.crop((20, 10, 320, 40)))
    line2 = np.asarray(out.crop((20, 50, 320, 90)))
    assert line1.min() < 128  # text drawn on plane 1
    assert line2.min() < 128  # text drawn on plane 2, wider than its own 120px plane


def test_single_unit_spanning_two_printed_lines_reflows_over_both(tmp_path) -> None:
    # Element-level hint: ONE unit whose members lie on two physical lines. The planes
    # come from line-clustering the members, so the translation reflows over both.
    input_path = tmp_path / "input.png"
    Image.new("RGB", (400, 120), (255, 255, 255)).save(input_path)
    units = [
        {"translated_text": "Grilled steak with red wine gravy shallots and fries",
         "block_id": 13, "level": "body",
         "members": [
             {"cell_id": 1, "text": "Biefstuk van de grill met rode wijn", "translate": True,
              "bbox": {"left": 20, "top": 10, "width": 300, "height": 30}},
             {"cell_id": 2, "text": "jus, sjalotten en frites", "translate": True,
              "bbox": {"left": 20, "top": 50, "width": 200, "height": 30}}]},
    ]
    png = render_translated_image(input_path, units)
    out = Image.open(BytesIO(png)).convert("RGB")
    line1 = np.asarray(out.crop((20, 10, 320, 40)))
    line2 = np.asarray(out.crop((20, 50, 320, 90)))
    assert line1.min() < 128
    assert line2.min() < 128


def test_interleaved_leftover_does_not_break_a_reflow_group(tmp_path) -> None:
    # An OCR noise cell ("L") lands between a dish's two lines in reading order; its
    # leftover unit must not split the block back into per-line fitting.
    from app.replacement.render import _groups

    units = [
        {"id": 1, "block_id": 13, "level": "body"},
        {"id": 2, "block_id": None, "level": None},
        {"id": 3, "block_id": 13, "level": "body"},
    ]
    groups = _groups(units)
    assert [[u["id"] for u in g] for g in groups] == [[1, 3], [2]]


def test_render_erases_planes_left_over_after_reflow(tmp_path) -> None:
    # The translation needs fewer lines than the original: the unused plane still gets
    # erased so no source text peeks through.
    input_path = tmp_path / "input.png"
    base = Image.new("RGB", (400, 120), (255, 255, 255))
    ImageDraw.Draw(base).rectangle((30, 55, 130, 75), fill=(0, 0, 0))  # "source text" on plane 2
    base.save(input_path)
    units = [
        {"translated_text": "Hello", "block_id": 3, "level": "body",
         "members": [
             {"cell_id": 1, "text": "Hallo wereld dit is lang", "translate": True,
              "bbox": {"left": 20, "top": 10, "width": 300, "height": 30}}]},
        {"translated_text": "yes", "block_id": 3, "level": "body",
         "members": [
             {"cell_id": 2, "text": "vervolgregel", "translate": True,
              "bbox": {"left": 20, "top": 50, "width": 120, "height": 30}}]},
    ]
    png = render_translated_image(input_path, units)
    out = np.asarray(Image.open(BytesIO(png)).convert("RGB"))
    plane2 = out[52:78, 25:135]
    assert plane2.min() > 200  # the black source text is erased, nothing redrawn there


def test_group_planes_snap_to_one_background_shade(tmp_path) -> None:
    # Three planes of one element on a lightly textured background sample three
    # slightly different greys; they must snap to ONE shade. Short translation ->
    # planes 2 and 3 are erase-only, so their fill colour is directly observable.
    input_path = tmp_path / "input.png"
    base = Image.new("RGB", (400, 160), (120, 120, 120))
    ImageDraw.Draw(base).rectangle((0, 60, 400, 110), fill=(132, 132, 132))
    ImageDraw.Draw(base).rectangle((0, 110, 400, 160), fill=(144, 144, 144))
    base.save(input_path)
    units = [
        {"translated_text": "Hi", "block_id": 1, "level": "title",
         "members": [
             {"cell_id": 1, "text": "DE SCHOEN", "translate": True,
              "bbox": {"left": 50, "top": 10, "width": 300, "height": 40}},
             {"cell_id": 2, "text": "WERKT ALS", "translate": True,
              "bbox": {"left": 50, "top": 62, "width": 300, "height": 40}},
             {"cell_id": 3, "text": "JIJ DAT DOET.", "translate": True,
              "bbox": {"left": 50, "top": 114, "width": 300, "height": 40}}]},
    ]
    png = render_translated_image(input_path, units)
    out = np.asarray(Image.open(BytesIO(png)).convert("RGB"))
    fill2 = out[80, 340]   # inside plane 2's erase, right of any text
    fill3 = out[130, 340]  # inside plane 3's erase
    assert (fill2 == fill3).all()  # one snapped shade, not two band colours


def test_centered_element_anchors_on_its_plane_center(tmp_path) -> None:
    # The VLM alignment hint: a centered element's line anchors on the plane's centre
    # instead of its left edge. A short translation of a wide centered original must
    # land in the middle, not at the left margin.
    input_path = tmp_path / "input.png"
    Image.new("RGB", (400, 100), (255, 255, 255)).save(input_path)
    units = [
        {"translated_text": "Hi", "alignment": "center", "block_id": 1, "level": "title",
         "members": [
             {"cell_id": 1, "text": "Hallo wereld dit is breed", "translate": True,
              "bbox": {"left": 100, "top": 30, "width": 200, "height": 30}}]},
    ]
    png = render_translated_image(input_path, units)
    out = np.asarray(Image.open(BytesIO(png)).convert("RGB"))
    columns = np.where(out.min(axis=(0, 2)) < 128)[0]
    assert len(columns) > 0
    drawn_center = (columns.min() + columns.max()) / 2
    assert abs(drawn_center - 200) < 25  # centred on the plane (plane spans 100..300)
    assert columns.min() > 120  # and NOT anchored at the left edge


def test_render_skips_translation_that_cannot_fit_the_footprint(tmp_path) -> None:
    # A chat-reply "translation" on a pictogram-sized cell does not fit even at the
    # smallest font; the footprint rule wins — the original pixels stay untouched
    # instead of an erase plane far beyond the cell.
    input_path = tmp_path / "input.png"
    Image.new("RGB", (200, 100), (10, 10, 10)).save(input_path)
    units = [
        {"translated_text": "I cannot translate this because you only provided one letter",
         "members": [
             {"cell_id": 1, "text": "i", "translate": True,
              "bbox": {"left": 5, "top": 5, "width": 12, "height": 14}}]},
    ]
    png = render_translated_image(input_path, units)
    out = np.asarray(Image.open(BytesIO(png)).convert("RGB"))
    assert out.max() <= 10


def test_render_skips_units_without_translation(tmp_path) -> None:
    input_path = tmp_path / "input.png"
    Image.new("RGB", (60, 60), (10, 10, 10)).save(input_path)
    units = [
        {"translated_text": "", "members": [
            {"cell_id": 1, "text": "x", "translate": True, "bbox": {"left": 5, "top": 5, "width": 40, "height": 20}}]},
        {"translated_text": "skip", "members": [
            {"cell_id": 2, "text": "9", "translate": False, "bbox": {"left": 5, "top": 30, "width": 40, "height": 20}}]},
    ]
    png = render_translated_image(input_path, units)
    out = np.asarray(Image.open(BytesIO(png)).convert("RGB"))
    # nothing translatable rendered -> image stays the solid dark colour
    assert out.max() <= 10


def test_allowed_width_spends_slack_only_over_verified_clean_background() -> None:
    # The 4% width slack is only spendable over pixels the planner VERIFIED as clean background
    # (plane["slack_px"], measured by the extend-fit scan): 4% of a page-wide line is tens of
    # pixels, enough to cross into an adjacent layout panel the original kept a margin to. A
    # clean margin (or no measurement — tilt, centered, tests) keeps the full 4%.
    from app.replacement.text.wrap import _allowed_width

    assert _allowed_width({}, 1100.0) == pytest.approx(1144.0)                  # unmeasured: 4%
    assert _allowed_width({"slack_px": 44.0}, 1100.0) == pytest.approx(1144.0)  # clean: full 4%
    assert _allowed_width({"slack_px": 17.0}, 1100.0) == pytest.approx(1117.0)  # panel at 17px
    assert _allowed_width({"slack_px": 0.0}, 1100.0) == pytest.approx(1100.0)   # ink right away


def test_split_table_row_drops_field_printed_in_another_unit() -> None:
    # The VLM unmerges a repeated table column (an icon-margin caption emitted as field 1 of
    # EVERY row) while the caption's cells exist once, in another unit: a row unit without
    # caption members must render ONLY its own column — prepending the caption translation to
    # every row was the bug. The drop needs POSITIVE evidence the field is printed in another
    # unit; without it (a merged-box order code garbled past the match threshold) the cautious
    # reflow stands, so text is never dropped from the image.
    from app.replacement.layout.tables import _split_table_row

    row = {
        "translated_text": "Bijschrift vertaald Inhoud vertaald.",
        "field_translations": [
            ("Icon Caption Title", "Bijschrift vertaald"),
            ("The actual row content text.", "Inhoud vertaald."),
        ],
        "members": [{
            "cell_id": 1, "text": "The actual row content text.", "translate": True,
            "bbox": {"left": 400, "top": 100, "width": 600, "height": 30},
        }],
    }
    cells = _split_table_row(row, ["Icon Caption", "Title"])  # caption cells live elsewhere
    assert cells is not None and len(cells) == 1
    assert cells[0]["translated_text"] == "Inhoud vertaald."
    # no sibling evidence (a garbled merged-box code): no split, the old reflow renders ALL fields
    assert _split_table_row(row, []) is None


def test_wrapped_caption_column_does_not_merge_into_content_column() -> None:
    # The close-cells merge gauges its gap on a LINE height: a caption column wrapped over two
    # lines doubles its union height, which used to double the allowed gap and glue the caption
    # onto the content column beside it. Single-line pairs (date | weekday) keep merging.
    from app.replacement.layout.tables import _should_merge_table_cells

    def cell(members):
        return {"members": members}

    wrapped_caption = cell([
        {"bbox": {"left": 140, "top": 1000, "width": 210, "height": 30}},
        {"bbox": {"left": 210, "top": 1032, "width": 80, "height": 30}},
    ])
    content = cell([{"bbox": {"left": 390, "top": 1010, "width": 700, "height": 30}}])
    assert _should_merge_table_cells(wrapped_caption, content) is False  # gap 40 > 0.75x30

    date = cell([{"bbox": {"left": 100, "top": 100, "width": 80, "height": 30}}])
    weekday = cell([{"bbox": {"left": 195, "top": 100, "width": 50, "height": 30}}])
    assert _should_merge_table_cells(date, weekday) is True  # gap 15 <= 0.75x30


def _pitch_plane(top: float, bottom: float, height: float) -> dict:
    return {"frame": ((1.0, 0.0), (0.0, 1.0), 0.0, 100.0, top, bottom), "true_height": height}


def test_snap_line_pitch_regularises_wobble_and_a_glyph_inflated_top() -> None:
    # A uniformly-leaded paragraph measured with quad wobble, plus ONE line whose top the
    # quad inflated by a sparse tall glyph — its extra true_height is the alibi. Every top
    # must land on one uniform grid, including the inflated one (whose render otherwise
    # near-collides with the line above it).
    from app.replacement.layout.planning import _snap_line_pitch

    noise = [0, 1, -2, 1, -1, -14, 0]  # index 5: inflated by a tall glyph
    heights = [30, 30, 30, 30, 30, 44, 30]
    planes = [
        _pitch_plane(100 + 40 * i + n, 100 + 40 * i + n + h, h)
        for i, (n, h) in enumerate(zip(noise, heights))
    ]
    _snap_line_pitch(planes)
    tops = [plane["frame"][4] for plane in planes]
    deltas = [b - a for a, b in zip(tops, tops[1:])]
    assert max(deltas) - min(deltas) < 0.01  # one uniform pitch
    assert tops[5] - (100 + 200 - 14) > 10   # the inflated top moved back down onto the grid


def test_snap_line_pitch_leaves_designed_structure_untouched() -> None:
    # Two false-friends of the tall-glyph case, both with normal line heights (no alibi):
    # a paragraph gap inside one group (a leaflet's two stacked paragraphs) and an extra
    # OCR plane squeezed between grid lines. Both must no-op WHOLE.
    from app.replacement.layout.planning import _snap_line_pitch

    for tops in ([100, 140, 180, 244, 284, 324], [100, 140, 180, 200, 240, 280]):
        planes = [_pitch_plane(top, top + 30, 30) for top in tops]
        _snap_line_pitch(planes)
        assert [plane["frame"][4] for plane in planes] == tops


_GF_TINOS = __import__("pathlib").Path.home() / ".local/share/fonts/gf/Tinos-Regular.ttf"


@pytest.mark.skipif(not _GF_TINOS.exists(), reason="mapped family fonts not provisioned")
def test_load_font_falls_back_per_character_for_uncovered_symbols() -> None:
    # The mapped family faces cover Latin but not the misc-symbol blocks — an academic
    # paper's affiliation markers (⋄, ⋆) rendered as tofu boxes. Such characters must draw
    # in DejaVu (which covers them) while the words keep the family face; a covered-only
    # line must keep the plain single-face path.
    from pathlib import Path
    from app.replacement.text.fit import FallbackFont

    font = load_font(20, "Aakanksha Naik ⋄ Pao ⋆", family="Times New Roman", weight=400)
    assert isinstance(font, FallbackFont)
    faces = {Path(getattr(face, "path", "")).name for face, _ in font.runs("Naik ⋄")}
    assert faces == {"Tinos-Regular.ttf", "DejaVuSans.ttf"}
    assert font.getlength("Naik ⋄") > font.getlength("Naik ")

    plain = load_font(20, "Aakanksha Naik", family="Times New Roman", weight=400)
    assert not isinstance(plain, FallbackFont)


def test_mixed_ink_body_prose_demotes_accent_lines_to_the_base_ink(tmp_path) -> None:
    # A body paragraph whose last source line is a chromatic citation run: the translation
    # re-wraps over the planes, so per-line colour would land on the wrong words — the group
    # renders in its achromatic base ink instead. The same mix on a header-level element
    # keeps its per-line inks (a two-tone heading is per-line by design).
    def blue_ink(level):
        img = Image.new("RGB", (460, 200), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        rows = [(30, (0, 0, 0)), (70, (0, 0, 0)), (110, (0, 0, 150))]
        for top, color in rows:
            for x in range(40, 380, 14):
                draw.rectangle((x, top, x + 6, top + 22), fill=color)
        path = tmp_path / f"{level}.png"
        img.save(path)
        unit = {"translated_text": "vertaalde alinea met heel wat woorden erin en nog meer",
                "block_id": 1, "level": level,
                "members": [
                    # bbox margins keep the border-ring bg sample on white paper
                    {"cell_id": i, "text": f"regel {i}", "translate": True,
                     "bbox": {"left": 30, "top": top - 5, "width": 366, "height": 33}}
                    for i, (top, _) in enumerate(rows)
                ]}
        png = render_translated_image(path, [unit])
        arr = np.asarray(Image.open(BytesIO(png)).convert("RGB")).astype(int)
        return int(((arr[:, :, 2] - arr[:, :, 0] > 60) & (arr[:, :, 2] > 90)).sum())

    assert blue_ink("body") == 0
    assert blue_ink("header") > 0


def test_translation_preserving_source_skips_the_render_and_keeps_the_original() -> None:
    # Untranslated content — names with affiliation symbols the OCR read differently (or not
    # at all), emails, a bare identical line — must match on WORDS and leave the original
    # print standing. Number-only lines localize their separators, so they need exact
    # equality; genuinely translated text must keep rendering.
    from app.replacement.layout.planning import _translation_preserves_source

    assert _translation_preserves_source("Benjamin Newman*⋆ Yoonjoo Lee ⋆",
                                         "* Benjamin Newman * Yoonjoo Lee") is True
    assert _translation_preserves_source("⋆ University of Washington ⋄ KAIST",
                                         "University of Washington KAIST") is True
    assert _translation_preserves_source("a@b.edu, c@d.kr", "a@b.edu, c@d.kr") is True
    assert _translation_preserves_source("58,41", "58,41") is True   # exact number: preserved
    assert _translation_preserves_source("58.41", "58,41") is False  # localized separator: renders
    assert _translation_preserves_source("Universiteit van Washington",
                                         "University of Washington") is False
    assert _translation_preserves_source("iets", "") is False


def test_split_table_row_cells_carry_their_own_field_source() -> None:
    # After a '|' field split each cell must compare its translation against ITS OWN members'
    # source (the parent row's source_text would make every split field read as changed and
    # defeat the identity-preserve on untranslated fields).
    unit = {
        "source_text": "Benjamin Newman Yoonjoo Lee",
        "field_translations": [
            ("Benjamin Newman", "Benjamin Newman*⋆"),
            ("Yoonjoo Lee", "Yoonjoo Lee ⋆"),
        ],
        "members": [
            {"cell_id": 1, "text": "Benjamin Newman", "translate": True,
             "bbox": {"left": 100, "top": 50, "width": 220, "height": 30}},
            {"cell_id": 2, "text": "Yoonjoo Lee", "translate": True,
             "bbox": {"left": 600, "top": 50, "width": 180, "height": 30}},
        ],
    }
    cells = _split_table_row(unit, ())
    assert cells is not None and len(cells) == 2
    sources = sorted(cell["source_text"] for cell in cells)
    assert sources == ["Benjamin Newman", "Yoonjoo Lee"]


def test_restore_printed_lead_marker_readds_a_lost_footnote_star() -> None:
    # The hint parse eats a leading "*" as markdown noise and the translator may drop the
    # symbol; the OCR member text is the print evidence, so the printed marker comes back.
    # A translation that kept a marker of its own (also a lookalike variant) stays untouched,
    # and a unit whose print has no lead marker never gains one.
    from app.replacement.layout.markers import _restore_printed_lead_marker

    footnote = {"members": [{"text": "*Equal contributions."}]}
    assert _restore_printed_lead_marker("Gelijke bijdragen.", footnote) == "*Gelijke bijdragen."
    assert _restore_printed_lead_marker("⋆Gelijke bijdragen.", footnote) == "⋆Gelijke bijdragen."
    plain = {"members": [{"text": "Equal contributions."}]}
    assert _restore_printed_lead_marker("Gelijke bijdragen.", plain) == "Gelijke bijdragen."
    starred_word = {"members": [{"text": "* niet aan een woord vast"}]}
    assert _restore_printed_lead_marker("los sterretje", starred_word) == "los sterretje"
