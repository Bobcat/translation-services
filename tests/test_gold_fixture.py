from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from app.grouping.units import TranslationUnit
from app.grouping.vlm import parse_grouping_output
from app.translation.gold import load_fixture_for_image
from app.translation.gold import resolve_gold_units


def _unit(unit_id: int, source_text: str, hint_index: int | None) -> TranslationUnit:
    return TranslationUnit(
        id=unit_id,
        order=unit_id,
        members=[],
        bbox={"left": 0, "top": 0, "width": 1, "height": 1},
        source_text=source_text,
        hint_index=hint_index,
    )


def _write_fixture(directory: Path, *, image_bytes, hashes=None, vlm_output="", blocks=()) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    image_sha256 = hashes if hashes is not None else hashlib.sha256(image_bytes).hexdigest()
    (directory / "demo.json").write_text(
        json.dumps(
            {
                "name": "demo",
                "image_sha256": image_sha256,
                "vlm_output": vlm_output,
                "reference_blocks": list(blocks),
            }
        ),
        encoding="utf-8",
    )


def test_load_fixture_for_image_matches_by_canonical_hash(tmp_path: Path) -> None:
    fixtures = tmp_path / "_gold"
    image = tmp_path / "input.jpg"
    image.write_bytes(b"canonical-image-bytes")
    _write_fixture(fixtures, image_bytes=b"canonical-image-bytes",
                   vlm_output="[Image Classification: x]\n[Level 3 / Body] hallo", blocks=["hello"])

    fixture = load_fixture_for_image(image, fixtures_dir=fixtures)
    assert fixture is not None and fixture.name == "demo"
    assert fixture.reference_blocks == ("hello",)

    other = tmp_path / "other.jpg"
    other.write_bytes(b"different-bytes")
    assert load_fixture_for_image(other, fixtures_dir=fixtures) is None


def test_load_fixture_accepts_a_list_of_hashes(tmp_path: Path) -> None:
    fixtures = tmp_path / "_gold"
    image = tmp_path / "input.jpg"
    image.write_bytes(b"copy-two")
    _write_fixture(
        fixtures,
        image_bytes=b"",
        hashes=[hashlib.sha256(b"copy-one").hexdigest(), hashlib.sha256(b"copy-two").hexdigest()],
        blocks=["hello"],
    )
    assert load_fixture_for_image(image, fixtures_dir=fixtures) is not None


def test_resolve_gold_units_maps_by_hint_index() -> None:
    """Canned hint and reference are 1:1, so a unit picks its reference block by hint_index."""
    blocks = ("First.", "Second.", "Third.")
    units = [
        _unit(1, "eerste", hint_index=0),       # -> blocks[0]
        _unit(2, "derde", hint_index=2),         # -> blocks[2]
        _unit(3, "GM", hint_index=None),         # no hint line -> unmatched
        _unit(4, "", hint_index=0),              # empty -> skipped
    ]
    results, unmatched = resolve_gold_units(units, blocks)
    by_id = {r.unit_id: r for r in results}

    assert by_id[1].translated_text == "First." and by_id[1].translation_route == "gold_fixture"
    assert by_id[2].translated_text == "Third."
    assert by_id[3].translated_text == "" and by_id[3].translation_route == "gold_unmatched"
    assert by_id[4].translation_route == "skipped_empty"
    assert unmatched == [3]


def test_registered_afstand_fixture_pins_hint_and_reference() -> None:
    """The committed fixture's canned VLM output parses to as many hint lines as there are
    reference blocks (1:1) — the property the reference run relies on. Skips when the local-only
    source image is absent (testset/ is gitignored)."""
    image = Path(__file__).resolve().parents[1] / "testset" / "afstand-houden.jpg"
    if not image.exists():
        pytest.skip("local-only testset asset (testset/ is gitignored) not present")
    fixture = load_fixture_for_image(image)
    assert fixture is not None and fixture.name == "afstand-houden"
    hint = parse_grouping_output(fixture.vlm_output)
    assert len(hint.units) == len(fixture.reference_blocks) == 10
    assert fixture.reference_blocks[1] == "You are a guest in the habitat of horses."
