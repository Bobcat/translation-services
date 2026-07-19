"""The fixture (frozen replay inputs) and snapshot (approved expected output) data model.

A fixture freezes everything the deterministic chain needs: the OCR ``cells``, the ``raw_hint``
string the VLM produced, and the translations. The translations are split by what they are a
function of: ``hint_translations`` (keyed by ``hint_index``) is the translation of each hint line —
a pure function of the frozen hint, independent of how align groups cells; ``leftover_translations``
(keyed by a member cell id) covers the cells that matched no hint line, which are inherently
per-cell. Neither keys on an align *output* (the old anchor-cell key was such an output, so an align
change that moved the anchor silently detached the translation). A snapshot records the approved
``expected_units`` (the align output, order-sensitive), ``ignored_cells``, and ``reocr`` (the
rendered image read back). Both are plain JSON on disk under ``testset/_regression/<name>/<variant>/``.
"""
from __future__ import annotations

import hashlib
import re
import json
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Fixture:
    image_sha256: str
    cells: list[dict[str, Any]]
    raw_hint: str
    # hint_index (str) -> {"translated_text": str, "field_translations": list[[src, tr]] | None}
    # The translation of a hint line; attached at replay to whichever unit matched that line.
    hint_translations: dict[str, dict[str, Any]]
    # member-cell-id (str) -> same entry shape; for leftover units (matched no hint line). Attached
    # at replay by cell membership, so a cell joining/leaving the unit does not detach it.
    leftover_translations: dict[str, dict[str, Any]]
    request_flags: dict[str, Any]
    grouping_model: str
    target_lang: str
    # The layout regions align ran with at capture time (empty for fixtures captured before
    # layout evidence existed — replay then feeds align None, the pre-layout path).
    layout_regions: list[dict[str, Any]] = field(default_factory=list)

    @property
    def preserve_heuristic_text(self) -> bool:
        # A flag a replay must re-apply: it shapes the unit set that is BOTH compared against the
        # snapshot and fed to render. The remaining flags are recorded for provenance only — they
        # acted on the translation side of the freeze boundary, so their effect is baked into the
        # frozen translations.
        return bool(self.request_flags.get("preserve_heuristic_text", True))

    @property
    def render_size_mode(self) -> str:
        # Also re-applied at replay: it changes the render itself, so a fixture approved under
        # "median" must replay under "median" or it is born-failing. The "min" fallback is the
        # HISTORIC capture default, not the current one (schemas default became "median"
        # 2026-07-06): fixtures captured before the flag existed were rendered with "min" and
        # must keep replaying that way. Do not sync this literal with the schema default.
        return str(self.request_flags.get("render_size_mode") or "min")

    @property
    def erase_fill_mode(self) -> str:
        # Same rule: the fill mode changes the render, so replay must reproduce it.
        return str(self.request_flags.get("erase_fill_mode") or "flat")

    @property
    def preserve_unchanged_text(self) -> bool:
        # Same rule: the flag also gates the render-layer identity-preserve, so replay must
        # reproduce it. Fixtures captured before default to False (everything re-renders).
        return bool(self.request_flags.get("preserve_unchanged_text", False))

    @property
    def width_fit_mode(self) -> str:
        # Same rule again; fixtures captured before the flag existed rendered "footprint".
        return str(self.request_flags.get("width_fit_mode") or "footprint")

    @property
    def size_metric_mode(self) -> str:
        # Same rule again; fixtures captured before the flag existed rendered "extent".
        return str(self.request_flags.get("size_metric_mode") or "extent")

    @property
    def size_cohort_mode(self) -> str:
        # Same rule again; fixtures captured before the flag existed rendered "off".
        return str(self.request_flags.get("size_cohort_mode") or "off")

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_sha256": self.image_sha256,
            "cells": self.cells,
            "raw_hint": self.raw_hint,
            "hint_translations": self.hint_translations,
            "leftover_translations": self.leftover_translations,
            "request_flags": self.request_flags,
            "grouping_model": self.grouping_model,
            "target_lang": self.target_lang,
            "layout_regions": self.layout_regions,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Fixture":
        return cls(
            image_sha256=str(data["image_sha256"]),
            cells=list(data["cells"]),
            raw_hint=str(data.get("raw_hint") or ""),
            hint_translations=dict(data.get("hint_translations") or {}),
            leftover_translations=dict(data.get("leftover_translations") or {}),
            request_flags=dict(data.get("request_flags") or {}),
            grouping_model=str(data.get("grouping_model") or ""),
            target_lang=str(data.get("target_lang") or ""),
            layout_regions=list(data.get("layout_regions") or []),
        )


@dataclass(frozen=True)
class Snapshot:
    # Align output, order-sensitive: one entry per unit (see ``expected_unit_of``).
    expected_units: list[dict[str, Any]]
    ignored_cells: list[int]
    # The rendered image read back by OCR: [{"text", "left", "top", "width", "height"}], reading order.
    reocr: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "expected_units": self.expected_units,
            "ignored_cells": self.ignored_cells,
            "reocr": self.reocr,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Snapshot":
        return cls(
            expected_units=list(data.get("expected_units") or []),
            ignored_cells=[int(c) for c in (data.get("ignored_cells") or [])],
            reocr=list(data.get("reocr") or []),
        )


def sha256(image_bytes: bytes) -> str:
    return hashlib.sha256(image_bytes).hexdigest()


def anchor_key(unit: dict[str, Any]) -> str:
    """The unit's anchor cell id (the member with the lowest reading ``order``) as a string. Used as
    the capture-time key for a leftover unit's translation; replay re-attaches it by cell membership,
    so it only has to be *a* member, not still the anchor."""
    members = unit.get("members") or []
    anchor = min(members, key=lambda m: int(m.get("order") or 0))
    return str(anchor["cell_id"])


def expected_unit_of(unit: dict[str, Any]) -> dict[str, Any]:
    """The align fields of a unit, compared order-sensitively in a regression: the ordered member
    cells plus the hint-derived labels AND the render-relevant fields align derives — the per-member
    translate flags and the font family/weight. Including those localises a render diff that stems
    from an align change here (upstream) instead of only surfacing as a pixel diff. Translation/text
    is excluded (a frozen input, not an align output)."""
    members = sorted(unit.get("members") or [], key=lambda m: int(m.get("order") or 0))
    return {
        "cells": [int(m["cell_id"]) for m in members],
        "member_translate": [bool(m.get("translate", True)) for m in members],
        "hint_index": unit.get("hint_index"),
        "level": unit.get("level"),
        "alignment": unit.get("alignment"),
        "font_family": unit.get("font_family"),
        "font_weight": unit.get("font_weight"),
        "bullet": bool(unit.get("bullet", False)),
        "bullet_marker": unit.get("bullet_marker"),
        "block_id": unit.get("block_id"),
    }


def load(variant_path: Path) -> tuple[Fixture, Snapshot]:
    fixture = Fixture.from_dict(json.loads((variant_path / "fixture.json").read_text()))
    snapshot = Snapshot.from_dict(json.loads((variant_path / "snapshot.json").read_text()))
    return fixture, snapshot


def save(variant_path: Path, fixture: Fixture, snapshot: Snapshot) -> None:
    variant_path.mkdir(parents=True, exist_ok=True)
    (variant_path / "fixture.json").write_text(json.dumps(fixture.to_dict(), ensure_ascii=False, indent=1))
    save_snapshot(variant_path, snapshot)


def save_snapshot(variant_path: Path, snapshot: Snapshot) -> None:
    """Overwrite just the snapshot (re-baseline) — the fixture inputs stay."""
    variant_path.mkdir(parents=True, exist_ok=True)
    (variant_path / "snapshot.json").write_text(json.dumps(snapshot.to_dict(), ensure_ascii=False, indent=1))


def source_path(variant_path: Path) -> Path | None:
    """The fixture's own source image (``source.<ext>``) — the exact canonical bytes the snapshot
    was rendered on. The fixture is self-contained: replay renders on this, not on ``testset/``."""
    for child in sorted(variant_path.glob("source.*")):
        if child.is_file():
            return child
    return None


def _raw_hint(llm_calls: list[dict[str, Any]]) -> str:
    for call in llm_calls:
        if "grouping" in str(call.get("role") or "").lower():
            return str((call.get("response") or {}).get("output_text") or "")
    return ""


def build_fixture(response: dict[str, Any], *, source_bytes: bytes) -> Fixture:
    ocr = response.get("ocr") or {}
    metadata = response.get("metadata") or {}
    units = ocr.get("translation_units") or []
    hint_translations: dict[str, dict[str, Any]] = {}
    leftover_translations: dict[str, dict[str, Any]] = {}
    for unit in units:
        entry = {
            "translated_text": unit.get("translated_text") or "",
            "field_translations": unit.get("field_translations"),
        }
        hint_index = unit.get("hint_index")
        if hint_index is not None:
            hint_translations[str(hint_index)] = entry      # the hint line's translation
        else:
            leftover_translations[anchor_key(unit)] = entry  # a cell with no hint line
    return Fixture(
        image_sha256=sha256(source_bytes),
        cells=ocr.get("cells") or [],
        raw_hint=_raw_hint(response.get("llm_calls") or []),
        hint_translations=hint_translations,
        leftover_translations=leftover_translations,
        request_flags={
            "preserve_heuristic_text": bool(metadata.get("preserve_heuristic_text", True)),
            "preserve_unchanged_text": bool(metadata.get("preserve_unchanged_text", False)),
            "use_geometry_columns": bool(metadata.get("use_geometry_columns", True)),
            "render_size_mode": str(metadata.get("render_size_mode") or "min"),
            "erase_fill_mode": str(metadata.get("erase_fill_mode") or "flat"),
            "width_fit_mode": str(metadata.get("width_fit_mode") or "footprint"),
            "size_metric_mode": str(metadata.get("size_metric_mode") or "extent"),
            "size_cohort_mode": str(metadata.get("size_cohort_mode") or "off"),
        },
        grouping_model=str(metadata.get("grouping_model") or ""),
        target_lang=str(metadata.get("target_lang_code") or ""),
        layout_regions=list(metadata.get("layout_regions") or []),
    )


def _next_variant(lang_dir: Path) -> str:
    """The next free ``vN`` under a ``<name>/<lang>`` dir (max existing + 1, so a deleted variant
    never collides)."""
    nums = []
    if lang_dir.exists():
        for child in lang_dir.iterdir():
            match = re.fullmatch(r"v(\d+)", child.name)
            if child.is_dir() and match:
                nums.append(int(match.group(1)))
    return f"v{(max(nums) + 1) if nums else 1}"
