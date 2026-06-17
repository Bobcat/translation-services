# Testset observations

Per-image findings from running the pipeline on `testset/` screenshots: what goes wrong,
the diagnosed cause, and whether it is fixed or parked. Each entry names the test image so a
fix can be re-checked against it.

Last updated: 2026-06-16.

---

## `sandals-1.jpg` (phone screenshot, Dutch shopping order page, nl→en)

### Fixed

- **Brand/eyebrow merged into the title.** The VLM gave `[Header] Skechers` and
  `[Title] Skechers GO WALK ARCH FIT 2.0 SANDAL …` as separate elements, but the standalone
  "Skechers" OCR cell tied (token "skechers") between both hint lines and position broke the
  tie toward the title, so it was swallowed into the title unit — a narrow plane that
  collapsed the whole title to a tiny font (and left two empty erase boxes).
  Fix: `align._pick_hint` now prefers, among tied candidates, the hint line the cell **fully
  accounts for** (every token), then position. A short line that is a prefix of a longer one
  keeps its own cell.

- **Icon label merged into a text line (PostNL).** OCR read the orange "postnl" logo as a
  tiny cell "ostnl"; it fuzzy-matched ("ostnl" ⊂ "postnl") into the "Bezorging door PostNL"
  unit, dragging the unit's left edge + erase onto the icon and shrinking the line.
  Fix: `align._drop_icon_fragments` / `_is_icon_fragment` drop a member that is **much shorter
  than the line** (`_ICON_HEIGHT_RATIO`) and **token-redundant** (adds no new word) → `ignored`.

### Open / parked

- **Leading icon inside a single OCR box (pencil "Schrijf een review …").** Here there is no
  separate cell to drop: OCR drew one box whose left edge starts at the ✎ pencil (L50) while
  the recognised text starts after it (~L111). The box left drives both the erase and the
  left anchor, so the translation is drawn over the pencil and the pencil is erased.
  - No clean align-level signal: OCR gives the whole-line text + one box, no glyph positions.
    An ink-column profile shows `icon run (x2–41) | 20px gap | text (x61…)`, where the gap is
    ~2.5× the line's letter spacing — but a naive "leading run + large gap" trim regresses on
    legitimate short-token starts (`€ 58,41`, `1. item`, `I am…`, leading dashes/bullets),
    because a word space is itself larger than a letter gap. The pixel data alone can't tell
    "leading icon (not in the text)" from "leading short word (in the text)".
  - CPU cost of an ink-profile is negligible (numpy column-sum on a tiny crop, gated to
    suspicious cells); the real cost is the false-positive surface and the maintenance of a
    fragile threshold for a rare-ish pattern.
  - Better directions (for a later decision): (a) check whether PaddleOCR can return
    **per-word boxes** — then the text-start vs line-box-left is known exactly, no pixels;
    (b) use the **VLM as the disambiguator** — it sees the icon, so only trim when it confirms
    a leading icon. Reactive variant preferred: only on a *suspicious* cell (box much wider
    than the text needs) send that crop to the VLM ("leading icon? where does text start?"),
    so normal pages pay nothing and the model adjudicates the ambiguous case. OCR stays
    authoritative for coordinates; the VLM supplies the semantic fact, the exact trim comes
    from a then-safe local measure.

- **`|` field whose value is OCR-garbled, re-rendered as garble.** The VLM read the order
  number correctly and marked the line as a label|value field: `Bestelnummer | AB100X/XY`.
  OCR garbled the value badly ("Besteinummer A8l00X/xYvX"). The garbled value tokens diluted
  the cell's match score below `_MATCH_THRESHOLD` (≈0.3 < 0.4), so the whole cell became a
  **leftover** and fell to the per-unit fallback, which translated the OCR text →
  "Order number A8l00X/xYvX". Two things lost: the VLM's correct value `AB100X/XY`, and the
  field structure (the value is a non-translatable code that should be kept verbatim). Worth
  considering: let a strong label match anchor the line even when the value half is garbled,
  and take a non-translatable `|` value from the VLM's clean reading rather than OCR.
  OCR confidence corroborates weakly: the garbled line scores **0.88** vs **0.98–0.9999** for
  the clean lines around it — lower, but a line-level aggregate (can't localise the garbled
  value half) and well above the `ocr.min_confidence` floor, so usable only as a soft
  "trust the VLM here" hint, not a clean discriminator.

- **Minor.** White erase boxes don't match the green card background; the phone status bar
  (12:13 / kik / 99) is treated as footer text; the top "Bestelnummer" line is slightly
  clipped by the screenshot edge.


## `weather-2.jpg` (phone screenshot, dark-theme weather app, Dutch, nl→en)

Renders well overall (dark background, white text re-rendered white, weather icons preserved,
"Peking"→"Beijing", "Multi-day forecast", "Show details"). The hard parts:

- **VLM classification label leaks into the rendered text (intermittent).** On the table-like
  forecast rows the VLM drifts from `[label]: text` to `[label] | field | [label] | field`,
  using ` | ` as the label/text separator and re-emitting the classification label mid-line:
  `[Level 3 / Body | Roboto | 14pt | 400] | 17 jun | Vandaag | [Level 3 / … | 400] | 21° / 31°`.
  `parse_grouping_output` only strips the LEADING `[label]`, so the embedded one (and a stray
  leading `|`) survive into the hint text and get translated/rendered ("Jun 17 Today [Level 3
  / Body Roboto 14pt"). VLM-non-deterministic (one run leaked, the next did not). Fix:
  parser-side — strip any embedded `[Level…]/[Metadata…]` label anywhere in the line and tidy
  the leftover `|` separators, keeping the real field `|`s. Optionally tighten the prompt
  (label once, with `:`); the parser guard is the real defence.

- **Date/day columns overlap in `|` field rows.** The OCR field boxes overlap horizontally —
  e.g. "21 jun" L88–242 and the day cell "n Zo" L216–338, so the day box starts (216) left of
  where the date box ends (242). The per-field renderer draws each `|` field at its cell box,
  so the day overwrites the date's last digit → "Jun 2Sun", "Jun 2Tue", "Jun 18Tomorrow".
  OCR also garbled some day names ("Zo"→"n Zo", "Di"→"n Di"). Fix direction: de-collide
  sequential fields left-to-right (a field's text shouldn't start before the previous field's
  drawn extent + a gap), rather than trusting overlapping OCR boxes.

- **The clock time renders twice.** OCR has one clock cell "21:31". The VLM additionally read a
  timestamp inside the kik notification and attached it as the hint for the "kik. Me" element
  (`'kik. Me' → '21:31 kik Mc'`), so the structured route injects a second "21:31". The phone
  status-bar / notification row is noisy and arguably should not be translated at all.


## `weather-1.jpg` (phone screenshot, light/blue weather app, Dutch, nl→en)

The dominant problem is **render sizing**, not the duplication it first looks like: the
translation is drawn far too small, so it neither covers nor erases the original, and the large
original glyph stays visible next to a tiny translation.

- **Translation collapses to a tiny font next to an un-erased original.** A forecast row's date
  is a big day number "8" (a non-translatable cell, kept) + a small "jun" label + the day name;
  the `8 jun` field translates to "Jun 8" but is fit into the narrow original "jun" footprint, so
  the condense-floor shrinks it to a tiny font ("8 Jun 8" = big original "8" + minuscule "Jun 8").
  Same on the air-quality line: "Goed 35" → "Good 35" rendered minuscule next to the un-erased
  large "35", and the pollen line ("Zeer veel pollen" → an un-erased "pollen" + tiny translation).
  This is render-sizing fragility (a narrow original plane drags the whole rendered text down via
  the condense floor) — the same root as the sandals title collapse, here dominant across the
  screen. The erase box shrinks with the tiny text, so the large original is not covered.
  Fix belongs in `app/replacement/render.py`: don't let a narrow/short plane dominate the group
  size, and size the erase to the original footprint rather than to the (too-small) rendered text.


## `smoking-1.jpg` (photo of a cigarette pack at an angle, Dutch warning, nl→en)

A photographed 3D object (not a screenshot), tilted ~-20°. OCR and the VLM both handle it
**perfectly** (confidence 1.00 on every cell, all text correct; VLM structure correct). The only
weak spot is **render placement of tilted text**: the heading and "Camel Blue 80" are fine, but
the dense fine-print warning line ("Look now! Stop … www.ikstopnu.nl … 0800-1995 … Or call the
stopline") overlaps — many small fragments at slightly varying angles (-13° to -23°) grouped and
reflowed on top of each other.

The angles are fairly **uniform** (~-20°), so it is mostly rotation, not strong perspective. Idea
(parked): a **global deskew** stage — rotate the whole image by the median per-cell text angle so
text is ~horizontal, then group/render on the straightened image (far more robust for the dense
line). We already compute per-cell angles (`app/ocr/merging.py`); this is the planned
"orientation rescue" stage as a global pre-rotation. Open choices: present the straightened
result vs rotate the render back onto the original photo; and rotation-only vs full perspective
unwarp (overkill while angles are uniform). Caveat: multi-plane objects (warning panel vs the
"Camel Blue 80" lid) — one global angle straightens the dominant plane only.


## Grouping prompt — v3 candidate (stricter format), 2026-06-17

A reworked grouping prompt was compared against the current one across the testset (one quick
pass each), then stress-tested 8× per image. It uses a STRICT output line
`[<t|h|b|m>|<font>|<size>pt|<weight>|<l|c|r>]: <text>` — single-letter importance codes and a
**required** left/center/right alignment field — and keeps an `[Image classification: …]` header.

**Why it wins.** The current prompt drifts on table-like layouts: it omits the `:` separator
(0/14 colons on sandals, 0/6 on smoking, 0/24 on weather-1) and intermittently re-emits the
classification label mid-row, which leaks into the rendered text (the weather "Jun 17 Today
[Level 3 / Body …]" bug). v3 fixes the format:

- **colon 8/8 on every image** (was erratic),
- **embedded-label leak 0 across all 88 stress-test runs** (incl. the dense tables kassabon /
  weather / menukaart) — the leak never recurred,
- **no runaway** (one v2 variant looped to 227 emoji lines; v3 stays tight, sandals 11-11),
- **image classification 8/8**,
- **better font guess**: the Nike body returns "Georgia" (serif, correct) **6/6**, where this
  model with the current prompt wobbled to Helvetica (sans).

**Remaining soft spot — alignment determinism.** A centered multi-line paragraph (the Nike body)
flips `l`↔`c` ~50/50. Likely cause (to address later): the VLM judges the paragraph as one
rectangular block (→ reads the left edge → `l`) rather than as 3 staggered lines sharing a centre
(→ `c`). A wrong alignment is cosmetic (the renderer moves the anchor in-plane), but the centred
body left-aligns on ~half of renders. Direction: nudge the prompt to judge multi-line alignment
from the per-line edges.

**Adoption (pending, next session).** Swap the prompt to v3 **and** update the parser
(`app/grouping/vlm.py`): map the single-letter importance codes `t/h/b/m` → title/header/body/
footer, and parse the trailing `l/c/r` alignment field (`c` → center, recognise `r`). The current
`_level_of` / `_ALIGN_CENTER` only understand full words.


## `cartoon.jpg` (meme with speech bubbles, Dutch, nl→en)

Speech-bubble translation. Fixed: an inline non-translatable token ("1, 2, 3, 4?") was doubled
(kept original + reproduced by the translation) and, once erased, briefly pushed its line out of
the bubble. The renderer now erases a reproduced token's full footprint but takes the line's
text size/position from the translatable members only.

Open / parked — **rounded-bubble erase clips the border.** The erase is an axis-aligned rectangle
covering the original text extent (+pad). A speech bubble is rounded with an outline just outside
the text; when the translation is much shorter (e.g. "Hoedje van papier?" -> "Paper hat?"), the
wide erase wipes the bubble's right rounded border and spills a little onto the photo. A clean fix
needs the bubble shape (erase within the rounded outline / inpaint), so it is parked.
