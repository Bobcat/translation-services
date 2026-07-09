from __future__ import annotations

from io import BytesIO

import cv2
import numpy as np
import pytest
from PIL import Image
from PIL import ImageDraw

import math

from app.replacement.ground.color import sample_region_colors
from app.replacement.ground.inpaint import border_pads
from app.replacement.ground.inpaint import budget_scale
from app.replacement.ground.inpaint import context_window
from app.replacement.ground.inpaint import work_scale
from app.replacement.text.fit import _dominant_script
from app.replacement.text.fit import fit_text
from app.replacement.text.fit import fold_lone_fullwidth_punctuation
from app.replacement.text.fit import is_cjk_text
from app.replacement.text.angle import _baseline_angle
from app.replacement.text.size import _group_size
from app.replacement.render import _plan_group
from app.replacement.render import _reproduced_in
from app.replacement.render import _split_table_row
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
    from app.replacement.render import _bullet_geometry

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


def test_clean_right_extension_stops_at_ink_protected_cells_and_the_cap() -> None:
    from app.replacement.render import _clean_right_extension

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
    # must go to the model.
    from app.replacement.ground.erase import _ellipse
    from app.replacement.ground.erase import _GROUND_RING_INNER_PX
    from app.replacement.ground.erase import _needs_model_fill
    from app.replacement.jobs import _Job

    quad = [(60, 60), (240, 60), (240, 90), (60, 90)]
    job = _Job(erase_quads=[quad], bg_color=(220, 60, 30), tile=None, dst_quad=None)
    occupied = np.zeros((160, 300), dtype=np.uint8)
    cv2.fillPoly(occupied, [np.asarray(quad, dtype=np.int32)], 255)
    occupied = cv2.dilate(occupied, _ellipse(_GROUND_RING_INNER_PX))

    designed = np.zeros((160, 300, 3), dtype=np.uint8)
    designed[:, :] = (220, 60, 30)  # the whole ring lives inside one solid band
    assert not _needs_model_fill(designed, job, occupied)

    graded = np.zeros((160, 300, 3), dtype=np.uint8)
    graded[:, :] = np.linspace(90, 200, 300).astype(np.uint8)[None, :, None]  # drift along the line
    assert _needs_model_fill(graded, job, occupied)


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
