"""The detector-free text signals: digit anchors and the script-aware text volume.

Anchors are the translation-invariant content — numbers survive any correct translation.
Signatures are normalized exactly as far as measurement showed to be benign (NFKC, OCR
confusion folding, grouping separators, leading zeros) and matched document-wide as a
multiset, with a residual glue/split resolution over reading-order adjacency. The named
limits (numeral rewriting like "238k" -> "238.000", digit-free prose) live in
docs/benchmark-method.md.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Any


# --- anchors + volume: detector-free text signals ---------------------------
# Grouping/decimal separators BETWEEN digits ("1,234.56" / "1.234,56" / "1 234"): stripped
# before run extraction, so locale reformatting by a correct translator shares one signature.
# Point/comma strip between any digits. The spaced forms (space, optionally after a point/
# comma: "1 234 567", "235. 000") only glue when they LOOK like thousand grouping: exactly
# three digits follow AND at most three digits precede - "#1 2025" (rank + year), "18 07"
# (a spaced date) and "In 2025, 119 students" (a prose comma) must stay separate numbers.
_DIGIT_SEP_RE = re.compile(r"(?<=\d)(?:[.,](?=\d)|(?<!\d{4})[.,]?[\u00a0\u2009 ](?=\d{3}(?!\d)))")


# Digit runs of >=2: single digits are list-marker/footnote noise, not anchors.
_DIGIT_RUN_RE = re.compile(r"\d{2,}")


# CJK scripts have no whitespace word boundaries: each kana/han/hangul char is one volume
# unit; the rest of the text counts whitespace-delimited word tokens.
_CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7af]")


_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


_CONFUSION_TOKEN_RE = re.compile(r"\S+")


_CONFUSABLE = {"o": "0", "O": "0", "l": "1", "I": "1"}


_TRANSPARENT = ".,\u00a0\u2009"  # separators skipped when checking digit adjacency


def _fold_ocr_confusions(text: str) -> str:
    """Fold the measured OCR digit-confusion classes: o/O is a misread zero ("2o25",
    "235,ooo", "(80O)") and l/I a misread one ("2o2l", "$5l3,750") — all observed on one
    geometric-sans source render. A confusable folds only when it touches a digit (looking
    through grouping separators, iterated so "ooo" chains onto a leading digit). That
    adjacency requirement keeps unit words glued to a number intact: "$927million" must not
    yield a phantom "11" from its ells. Folding both sides unifies each confusion class, so
    a misread on one side no longer counts the other side's correct text as lost."""
    def fold(match: re.Match) -> str:
        token = match.group(0)
        if not any(ch.isdigit() for ch in token) or not any(ch in _CONFUSABLE for ch in token):
            return token
        chars = list(token)

        def digit_neighbour(index: int, step: int) -> bool:
            j = index + step
            while 0 <= j < len(chars) and chars[j] in _TRANSPARENT:
                j += step
            return 0 <= j < len(chars) and chars[j].isdigit()

        changed = True
        while changed:
            changed = False
            for i, ch in enumerate(chars):
                if ch in _CONFUSABLE and (digit_neighbour(i, -1) or digit_neighbour(i, 1)):
                    chars[i] = _CONFUSABLE[ch]
                    changed = True
        return "".join(chars)
    return _CONFUSION_TOKEN_RE.sub(fold, text)


def _digit_signatures(text: str) -> list[str]:
    """The digit signatures of one segment text: NFKC -> confusion folding -> separator
    stripping -> digit runs -> numeric normalization. One path, shared by counting and
    evidence. Leading zeros are stripped so "07 July" equals a localized "7 juli" — and a
    number below 10, however written, is no anchor (same status as a bare single digit)."""
    normalized = _fold_ocr_confusions(unicodedata.normalize("NFKC", text))
    runs = _DIGIT_RUN_RE.findall(_DIGIT_SEP_RE.sub("", normalized))
    return [sig for run in runs if len(sig := run.lstrip("0")) >= 2]


def _page_signature_sequences(pages: list[dict[str, Any]]) -> list[list[str]]:
    """Per page, the ordered digit signatures over its segments in reading order — the
    adjacency evidence the glue/split resolution below needs. Page-level (not per segment):
    a number wrapped across a line break lands in two OCR segments ("aged 50-" / "59 years"),
    and that wrap is exactly a case the resolution must see as adjacent."""
    return [
        [
            signature
            for segment in page.get("segments") or []
            for signature in _digit_signatures(str(segment.get("text") or ""))
        ]
        for page in pages
    ]


def _match_anchors(
    source_pages: list[dict[str, Any]], target_pages: list[dict[str, Any]]
) -> dict[str, Any]:
    """Document-wide multiset match of digit anchors (document-wide: reflow across page
    boundaries must not count as loss), plus a residual glue/split resolution: OCR sometimes
    reads one number as two ("513.750" wrapping into "513" + "750") or two numbers as one
    ("50-59" with a lost dash becoming "5059"). A leftover mismatch is forgiven when one
    side's signature equals the concatenation of two signatures that sit ADJACENT in reading
    order on the other side — adjacency keeps this from inventing matches. Known, accepted
    limit: numeral-word regrouping (e.g. "238k" written out as "238.000", or a CJK myriad
    form) changes the signature and reads as loss — named in the design doc."""
    source_sequences = _page_signature_sequences(source_pages)
    target_sequences = _page_signature_sequences(target_pages)
    source_counts = Counter(sig for seq in source_sequences for sig in seq)
    target_counts = Counter(sig for seq in target_sequences for sig in seq)
    survived = sum(min(count, target_counts[sig]) for sig, count in source_counts.items())
    missing = Counter({sig: count - target_counts[sig]
                       for sig, count in source_counts.items() if count > target_counts[sig]})
    added = Counter({sig: count - source_counts[sig]
                     for sig, count in target_counts.items() if count > source_counts[sig]})

    def _pair_available(counter: Counter, x: str, y: str) -> bool:
        return counter[x] >= 2 if x == y else counter[x] >= 1 and counter[y] >= 1

    # The target glued two adjacent source numbers into one.
    for sequence in source_sequences:
        for x, y in zip(sequence, sequence[1:]):
            if _pair_available(missing, x, y) and added[x + y] >= 1:
                missing[x] -= 1
                missing[y] -= 1
                added[x + y] -= 1
                survived += 2
    # The target split one source number into two adjacent pieces.
    for sequence in target_sequences:
        for x, y in zip(sequence, sequence[1:]):
            if _pair_available(added, x, y) and missing[x + y] >= 1:
                added[x] -= 1
                added[y] -= 1
                missing[x + y] -= 1
                survived += 1

    return {
        "total": sum(source_counts.values()),
        "survived": survived,
        "missing": {sig: count for sig, count in missing.items() if count > 0},
        "added": {sig: count for sig, count in added.items() if count > 0},
    }


def anchor_details(measurement: dict[str, Any]) -> dict[str, Any]:
    """The evidence behind the anchors axis, for the view's click-through: which digit anchors
    of the source are missing from the translation (and which are new), each located at the
    segment that carries it — page, bbox, and a text excerpt — so a reviewer can verify a score
    in two clicks instead of trusting it. Pure function over a stored measurement."""
    source_pages = list((measurement.get("source") or {}).get("pages") or [])
    target_pages = list((measurement.get("translated") or {}).get("pages") or [])
    match = _match_anchors(source_pages, target_pages)
    return {
        "anchors_source": match["total"],
        "anchors_survived": match["survived"],
        "missing": _locate_anchor_occurrences(source_pages, match["missing"]),
        "added": _locate_anchor_occurrences(target_pages, match["added"]),
    }


def _locate_anchor_occurrences(
    pages: list[dict[str, Any]], needed: dict[str, int]
) -> list[dict[str, Any]]:
    """For each signature in ``needed``, the first N carrying segments (occurrences are
    interchangeable — any N of them make the count). Reading order: pages, then segments."""
    remaining = dict(needed)
    out: list[dict[str, Any]] = []
    for page in pages:
        if not any(count > 0 for count in remaining.values()):
            break
        page_number = int(page.get("index", 0)) + 1
        for segment in page.get("segments") or []:
            text = str(segment.get("text") or "")
            for signature in _digit_signatures(text):
                if remaining.get(signature, 0) > 0:
                    remaining[signature] -= 1
                    out.append(
                        {
                            "signature": signature,
                            "page": page_number,
                            "text": text[:160],
                            "bbox": dict(segment.get("bbox") or {}),
                        }
                    )
    out.sort(key=lambda entry: (entry["page"], entry["signature"]))
    return out


def _volume_units(pages: list[dict[str, Any]]) -> int:
    """Script-aware text volume: CJK characters count one each, everything else per word
    token. Comparable between the two sides of one document; across documents only as a
    ratio (language pairs inflate differently, but identically for every system on a row)."""
    units = 0
    for page in pages:
        for segment in page.get("segments") or []:
            text = str(segment.get("text") or "")
            units += len(_CJK_RE.findall(text))
            units += sum(1 for _ in _TOKEN_RE.finditer(_CJK_RE.sub(" ", text)))
    return units
