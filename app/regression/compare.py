"""Diff a replay against its snapshot. Pure functions; each returns a list of human-readable
mismatch strings (empty list = pass)."""
from __future__ import annotations

from collections import Counter
from typing import Any


def diff_units(
    expected: list[dict[str, Any]],
    actual: list[dict[str, Any]],
    expected_ignored: list[int],
    actual_ignored: list[int],
) -> list[str]:
    """Exact, order-sensitive align comparison: unit count, then each unit's ordered cells + label
    fields, then the ignored-cell set. Any difference is a regression in ``align.py``."""
    diffs: list[str] = []
    if len(expected) != len(actual):
        diffs.append(f"unit count {len(actual)} != expected {len(expected)}")
    for index, (exp, act) in enumerate(zip(expected, actual)):
        for key in exp:
            if exp.get(key) != act.get(key):
                diffs.append(f"unit[{index}].{key}: {act.get(key)!r} != expected {exp.get(key)!r}")
    if sorted(expected_ignored) != sorted(actual_ignored):
        diffs.append(f"ignored_cells {sorted(actual_ignored)} != expected {sorted(expected_ignored)}")
    return diffs


def _norm(text: str) -> str:
    return "".join(ch.lower() for ch in str(text) if ch.isalnum())


def _centroid(row: dict[str, Any]) -> tuple[float, float]:
    return (row["left"] + row["width"] / 2.0, row["top"] + row["height"] / 2.0)


def diff_reocr(
    expected: list[dict[str, Any]],
    actual: list[dict[str, Any]],
    *,
    max_shift: float = 3.0,
) -> list[str]:
    """Behavioural render comparison on the re-OCR of the rendered image. Token-recall must be 1.0
    (no missing or extra normalized tokens) and each matched region's centroid must be within
    ``max_shift`` px. On the capture machine (same fonts) this is effectively exact; the few px
    absorb sub-pixel anti-aliasing."""
    diffs: list[str] = []
    exp = [(_norm(r["text"]), *_centroid(r)) for r in expected if _norm(r["text"])]
    act = [(_norm(r["text"]), *_centroid(r)) for r in actual if _norm(r["text"])]

    missing = Counter(t for t, _, _ in exp) - Counter(t for t, _, _ in act)
    extra = Counter(t for t, _, _ in act) - Counter(t for t, _, _ in exp)
    if missing:
        diffs.append(f"missing rendered text: {dict(missing)}")
    if extra:
        diffs.append(f"extra rendered text: {dict(extra)}")

    remaining = list(act)
    for token, ex, ey in exp:
        candidates = [(i, a) for i, a in enumerate(remaining) if a[0] == token]
        if not candidates:
            continue
        index, hit = min(candidates, key=lambda ia: (ia[1][1] - ex) ** 2 + (ia[1][2] - ey) ** 2)
        shift = ((hit[1] - ex) ** 2 + (hit[2] - ey) ** 2) ** 0.5
        if shift > max_shift:
            diffs.append(f"'{token}' moved {shift:.0f}px (> {max_shift:.0f}px)")
        remaining.pop(index)
    return diffs
