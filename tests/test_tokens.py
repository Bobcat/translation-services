"""Tokenizer guarantees.

The CJK pass must add tokens for Han/Kana/Hangul *without* changing tokenization of any non-CJK
(in particular Latin-script) source — see app/grouping/tokens.py. These tests lock both halves:
the equality against the historical tokenizer over a Latin corpus, and the new CJK behaviour.
"""
import re
import unicodedata

from app.grouping.tokens import _CJK
from app.grouping.tokens import _tokens


def _old_tokens(text: str) -> list[str]:
    """The tokenizer as it was before the CJK pass: NFKD, lowercase, strip combining marks,
    then runs of [a-z0-9]. Kept here as the reference the new tokenizer must equal on non-CJK."""
    normalized = unicodedata.normalize("NFKD", str(text or "").lower())
    stripped = "".join(char for char in normalized if not unicodedata.combining(char))
    return re.findall(r"[a-z0-9]+", stripped)


# Latin-script corpus, including the tricky cases the zero-regression argument hinges on:
# accented letters (NFKD → ASCII), letters with no decomposition (ø, ß — always dropped), and
# symbols whose NFKD lands in another alphabet (µ → Greek μ, Ω → Greek Ω — must still drop).
LATIN_CORPUS = [
    "THE SHOE WORKS IF YOU DO.",
    "Nike Sweet Classic High is both comfortable and stylish.",
    "color options, this sneaker is the perfect choice for everyday casual wear",
    "Café crème — naïve façade",
    "Smørrebrød Straße ø ß",
    "5 µg/mL at 50 Ω",
    "€ 8,50  1,69 B  -2,00",
    "www.nike.com  https://example.org/path  info@x.io",
    "Bezorging door PostNL",
    "",
    "   ",
    "1. item  2) next",
]


def test_latin_tokenization_is_unchanged():
    for text in LATIN_CORPUS:
        assert _tokens(text) == _old_tokens(text), text


def test_cjk_ranges_never_collide_with_latin_tokens():
    # The Latin pass uses exactly [a-z0-9]; no CJK range code point is an ASCII letter/digit, so
    # the two passes are disjoint and the CJK pass can only ever ADD tokens to non-CJK text.
    for char in "abcdefghijklmnopqrstuvwxyz0123456789":
        assert not re.match(f"[{_CJK}]", char)


def test_cjk_text_tokenizes_per_character():
    # Pure-CJK line: one token per ideograph, punctuation dropped (。、 are not in the ranges).
    assert _tokens("感，快来入手一双吧。") == ["感", "快", "来", "入", "手", "一", "双", "吧"]


def test_mixed_latin_and_cjk():
    # The product-name line from the Nike body: Latin words first, then the CJK characters.
    assert _tokens("Nike Sweet 既舒适") == ["nike", "sweet", "既", "舒", "适"]
