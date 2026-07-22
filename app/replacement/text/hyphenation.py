"""Soft-hyphen break points for filling a justified line, via pyphen's TeX dictionaries.

A translation into a compounding language (German, Dutch) turns "grounded question answering"
into one 18-glyph word. When that word will not fit the line it is on, it wraps whole, and the
line it left ends 200px short — a gap justification cannot close, because a space can only stretch
so far (wrap._JUSTIFY_MAX_EXTRA). The fix is the same one a typesetter uses: break the word with a
hyphen so its head fills the short line. pyphen supplies the break points; the fitting decision
(which break, whether it helps) stays in the justify code that has the font and the target width.
"""
from __future__ import annotations

try:
    import pyphen
except ImportError:  # pyphen is a hard dependency, but never fail a render if the wheel is absent
    pyphen = None  # type: ignore[assignment]


def make_hyphenator(target_lang: str) -> "pyphen.Pyphen | None":
    """A pyphen dictionary for ``target_lang`` (a code like ``nl`` / ``de`` / ``en_US``), or
    ``None`` when the language does not hyphenate with Latin soft-hyphens (CJK, and anything
    pyphen has no dictionary for) or the wheel is missing. ``None`` means "do not hyphenate",
    which is exactly right for CJK — those scripts break per character, not per syllable."""
    if pyphen is None:
        return None
    lang = str(target_lang or "").strip().replace("-", "_")
    if not lang:
        return None
    base = lang.split("_")[0].lower()
    languages = {key.lower(): key for key in pyphen.LANGUAGES}
    # Prefer an exact match, then the bare base code (pyphen ships e.g. both "nl" and "nl_NL"),
    # then any regional variant of the base.
    for candidate in (lang.lower(), base):
        if candidate in languages:
            chosen = languages[candidate]
            break
    else:
        variants = sorted(key for key in pyphen.LANGUAGES if key.lower().startswith(base + "_"))
        if not variants:
            return None
        chosen = variants[0]
    try:
        return pyphen.Pyphen(lang=chosen)
    except Exception:
        return None
