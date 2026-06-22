"""The fixture (frozen replay inputs) and snapshot (approved expected output) data model.

A fixture freezes everything the deterministic chain needs: the OCR ``cells``, the ``raw_hint``
string the VLM produced, and the per-unit ``translations`` (keyed by the unit's anchor cell). A
snapshot records the approved ``expected_units`` (the align output, order-sensitive),
``ignored_cells``, and ``reocr`` (the rendered image read back). Both are plain JSON on disk
under ``testset/_regression/<name>/<variant>/``.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Fixture:
    image_sha256: str
    cells: list[dict[str, Any]]
    raw_hint: str
    # anchor-cell-id (str) -> {"translated_text": str, "field_translations": list[[src, tr]] | None}
    translations: dict[str, dict[str, Any]]
    request_flags: dict[str, bool]
    grouping_model: str
    target_lang: str

    @property
    def preserve_heuristic_text(self) -> bool:
        # The one flag a replay must re-apply (it filters the unit set fed to render). The other
        # flags are recorded for provenance only — they shaped the now-frozen translation.
        return bool(self.request_flags.get("preserve_heuristic_text", True))

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_sha256": self.image_sha256,
            "cells": self.cells,
            "raw_hint": self.raw_hint,
            "translations": self.translations,
            "request_flags": self.request_flags,
            "grouping_model": self.grouping_model,
            "target_lang": self.target_lang,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Fixture":
        return cls(
            image_sha256=str(data["image_sha256"]),
            cells=list(data["cells"]),
            raw_hint=str(data.get("raw_hint") or ""),
            translations=dict(data.get("translations") or {}),
            request_flags=dict(data.get("request_flags") or {}),
            grouping_model=str(data.get("grouping_model") or ""),
            target_lang=str(data.get("target_lang") or ""),
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
    """The unit's anchor cell id (the member with the lowest reading ``order``) as a string — the
    stable, unique key for attaching a frozen translation to a re-grouped unit."""
    members = unit.get("members") or []
    anchor = min(members, key=lambda m: int(m.get("order") or 0))
    return str(anchor["cell_id"])


def expected_unit_of(unit: dict[str, Any]) -> dict[str, Any]:
    """The align fields of a unit, compared order-sensitively in a regression: the ordered member
    cell ids plus the hint-derived labels. Translation/text is deliberately excluded (it is a
    frozen input, not an align output)."""
    members = sorted(unit.get("members") or [], key=lambda m: int(m.get("order") or 0))
    return {
        "cells": [int(m["cell_id"]) for m in members],
        "hint_index": unit.get("hint_index"),
        "level": unit.get("level"),
        "alignment": unit.get("alignment"),
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
    (variant_path / "snapshot.json").write_text(json.dumps(snapshot.to_dict(), ensure_ascii=False, indent=1))
