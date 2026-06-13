"""Reference-translation override — an opt-in eval harness, not the prod path.

A request may carry ``translation_fixture`` (any non-empty value, conventionally "auto").
The pipeline then recognises the canonical ingested image by its sha256 against the fixtures
in ``testset/_gold/`` and, instead of calling llm-pool, simulates BOTH model responses from
the fixture: the VLM grouping hint (so the units are pinned and deterministic) and the
translation (the reference image's text). Because the canned hint and the canned reference
are authored 1:1, each unit maps onto its reference block by ``hint_index`` — no OCR/hint
text matching. This pins the translation to a reference so a rendered run isolates the
OCR -> grouping -> re-placement quality for side-by-side comparison.

Opt-in only: a normal request never touches this. No fixture matches the image ->
``GoldFixtureError``. A unit that did not align to a hint line stays untranslated
(route ``gold_unmatched``).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from app.grouping.units import TranslationUnit
from app.translation.translate import TranslatedUnit


_GOLD_DIR = Path(__file__).resolve().parents[2] / "testset" / "_gold"


class GoldFixtureError(RuntimeError):
    """No registered reference fixture matches the image of an opt-in fixture run."""


@dataclass(frozen=True)
class GoldFixture:
    name: str
    image_hashes: tuple[str, ...]
    vlm_output: str  # the canned VLM grouping response (raw, parsed like a real one)
    reference_blocks: tuple[str, ...]  # the reference translation, 1:1 with the hint lines


def image_identity(input_path: Path) -> str:
    """sha256 of the canonical ingested image bytes. The pipeline's ``input_path`` is
    already the canonicalised upload, so this is stable for a given source image."""
    return hashlib.sha256(Path(input_path).read_bytes()).hexdigest()


def load_fixtures(fixtures_dir: Path = _GOLD_DIR) -> list[GoldFixture]:
    fixtures: list[GoldFixture] = []
    if not fixtures_dir.is_dir():
        return fixtures
    for path in sorted(fixtures_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        raw_hashes = data.get("image_sha256")
        hashes = [raw_hashes] if isinstance(raw_hashes, str) else list(raw_hashes or [])
        fixtures.append(
            GoldFixture(
                name=str(data.get("name") or path.stem),
                image_hashes=tuple(str(h).strip().lower() for h in hashes if str(h).strip()),
                vlm_output=str(data.get("vlm_output") or ""),
                reference_blocks=tuple(str(block) for block in (data.get("reference_blocks") or [])),
            )
        )
    return fixtures


def load_fixture_for_image(input_path: Path, *, fixtures_dir: Path = _GOLD_DIR) -> GoldFixture | None:
    identity = image_identity(input_path)
    for fixture in load_fixtures(fixtures_dir):
        if identity in fixture.image_hashes:
            return fixture
    return None


def resolve_gold_units(
    units: list[TranslationUnit], reference_blocks: tuple[str, ...]
) -> tuple[list[TranslatedUnit], list[int]]:
    """Map each unit onto its reference block by ``hint_index`` — the canned hint and the
    reference are 1:1, so this is a straight positional pick (no text matching). Returns the
    translated units and the ids of units that aligned to no hint line (left untranslated)."""
    unmatched: list[int] = []
    results: list[TranslatedUnit] = []
    for unit in units:
        source_text = str(unit.source_text or "").strip()
        if not source_text:
            results.append(TranslatedUnit(unit.id, unit.source_text, "", "", "skipped_empty"))
            continue
        index = unit.hint_index
        if index is not None and 0 <= index < len(reference_blocks):
            results.append(
                TranslatedUnit(unit.id, unit.source_text, reference_blocks[index], "gold_fixture", "gold_fixture")
            )
            continue
        unmatched.append(unit.id)
        results.append(TranslatedUnit(unit.id, unit.source_text, "", "", "gold_unmatched"))
    return results, unmatched
