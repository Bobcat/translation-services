from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image

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
