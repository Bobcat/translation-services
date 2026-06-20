from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image
from PIL import ImageDraw

from app.replacement.color import sample_region_colors
from app.replacement.fit import _dominant_script
from app.replacement.fit import fit_text
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
