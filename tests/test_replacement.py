from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image
from PIL import ImageDraw

from app.replacement.color import sample_region_colors
from app.replacement.fit import fit_text
from app.replacement.render import render_translated_image


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
