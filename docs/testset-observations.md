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
