"""Cluster consecutive translation units into render-groups."""
from __future__ import annotations

from typing import Any


def _groups(units: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Consecutive units of one VLM block at one level reflow together — a wrapped
    dish, a body paragraph. The level guard keeps a heading from merging into its
    body text. Leftovers (no block — an OCR noise cell interleaved in reading order)
    stay alone but do NOT break the surrounding block's run, or one stray cell would
    split a dish back into per-line fitting.

    A BULLET unit always starts its own flow: a list item is a semantic boundary,
    and reflowing text across it garbles every item (measured on a hint variant
    that put a heading and four bullet items in one block — the heading's plane
    received the first item's opening words). Non-bullet continuation units after
    it still join the item's run via the same block/level key."""
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] | None = None
    previous: tuple[Any, Any] | None = None
    for unit in units:
        key = (unit.get("block_id"), unit.get("level"))
        if key[0] is None:
            groups.append([unit])
            continue
        if current is not None and key == previous and not unit.get("bullet"):
            current.append(unit)
        else:
            current = [unit]
            groups.append(current)
        previous = key
    return groups
