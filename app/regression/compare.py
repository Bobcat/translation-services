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


def _words(text: str) -> list[str]:
    """Per-word normalized tokens (split on whitespace first), so it does not matter whether OCR
    grouped a line into one box or several."""
    return [w for w in (_norm(part) for part in str(text).split()) if w]


def _centroid(row: dict[str, Any]) -> tuple[float, float]:
    return (row["left"] + row["width"] / 2.0, row["top"] + row["height"] / 2.0)


def diff_reocr(
    expected: list[dict[str, Any]],
    actual: list[dict[str, Any]],
    *,
    max_shift: float = 3.0,
) -> list[str]:
    """Behavioural render comparison on the re-OCR of the rendered image. Text is compared as a WORD
    multiset, so OCR grouping a line into one box vs several ("ah pizza" as one segment vs "ah" +
    "pizza") never reads as missing/extra. Position is then checked per segment: each segment that
    matches by full normalized text must have its centroid within ``max_shift`` px. On the capture
    machine (same fonts) this is effectively exact; the few px absorb sub-pixel anti-aliasing."""
    diffs: list[str] = []

    exp_words = Counter(word for row in expected for word in _words(row["text"]))
    act_words = Counter(word for row in actual for word in _words(row["text"]))
    missing = exp_words - act_words
    extra = act_words - exp_words
    if missing:
        diffs.append(f"missing rendered text: {dict(missing)}")
    if extra:
        diffs.append(f"extra rendered text: {dict(extra)}")

    # A segment whose word-grouping changed has no full-text match and is skipped for position — its
    # content is already covered by the word check above.
    exp_seg = [(_norm(r["text"]), *_centroid(r)) for r in expected if _norm(r["text"])]
    act_seg = [(_norm(r["text"]), *_centroid(r)) for r in actual if _norm(r["text"])]
    remaining = list(act_seg)
    for token, ex, ey in exp_seg:
        candidates = [(i, a) for i, a in enumerate(remaining) if a[0] == token]
        if not candidates:
            continue
        index, hit = min(candidates, key=lambda ia: (ia[1][1] - ex) ** 2 + (ia[1][2] - ey) ** 2)
        shift = ((hit[1] - ex) ** 2 + (hit[2] - ey) ** 2) ** 0.5
        if shift > max_shift:
            diffs.append(f"'{token}' moved {shift:.0f}px (> {max_shift:.0f}px)")
        remaining.pop(index)
    return diffs
