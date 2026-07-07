# Testset observations

Per-image findings from running the pipeline on `testset/` screenshots: what goes wrong,
the diagnosed cause, and whether it is fixed or parked. Each entry names the test image so a
fix can be re-checked against it.

Last updated: 2026-07-07.

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
  A stronger instance on `evil-clown-doll.jpg` (en→nl): OCR misread "out" (the red banner's
  "out of bed!") as **"tQut"** at **confidence 0.667** — a clear outlier against the 0.89–1.00
  of every other banner cell — and gave it an anomalous box (**h=120** vs ~64–85 for the real
  text). Unlike the sandals line-aggregate, here `confidence < ~0.7` **and** a height far above
  the line's median are two sharp, localised signals at once, so the worst garbles are flaggable
  → fall back to the VLM text for that cell. (The cell still binds to its caption, so the original
  "out" re-renders as the stray "tQut".)

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

### Fixed

- **Inline non-translatable token doubled ("1, 2, 3, 4?").** When OCR splits the bold "1, 2, 3, 4?"
  off the question into its own cell, it becomes a `translate:false` member the renderer keeps in
  place — while the structured translation (of the whole hint line) re-emits "1, 2, 3, 4?", so the
  token shows twice (the bold "|, 2, 3, 4?" beside the translated question). Fix:
  `render._reproduced_in` pulls a non-translatable member into the unit's erase when its text is
  reproduced in the translation AND the translation carries more than just that token, so the
  translation covers it once; a standalone token translating to itself (a lone price) stays put.
  An earlier fix (erase a reproduced token's full footprint + size the line from the translatable
  members) was reverted for making the cartoon worse — this is the narrow erase-only variant,
  outside that reverted line-geometry. Only manifests when OCR splits the token; when OCR keeps it
  in the question cell there is no duplicate either way (so it is intermittent, OCR-split-driven).

### Open / parked

- **rounded-bubble erase clips the border.** The erase is an axis-aligned rectangle
covering the original text extent (+pad). A speech bubble is rounded with an outline just outside
the text; when the translation is much shorter (e.g. "Hoedje van papier?" -> "Paper hat?"), the
wide erase wipes the bubble's right rounded border and spills a little onto the photo. A clean fix
needs the bubble shape (erase within the rounded outline / inpaint), so it is parked.


## `danger-1.jpg` / `danger-2.jpg` (multilingual warning + info signs, photos, →nl/zh/en)

These signs repeat the **same** message in several languages on one line/region
(`HÆTTA! DANGER!`, `GEFAHR! 危险`). It is a hard class — off-the-shelf translation tools garble
them more than we do. Two distinct problems, addressed differently.

### Fixed — translation prompt: translate every language, including repeats

- **Symptom.** On a line carrying the same word in two+ languages the model translated one and left
  the rest (`HÆTTA! DANGER!` → `GEVAAR! DANGER!`; `GEFAHR! 危险` → `GEVAAR! 危险`).
- **Diagnosis.** Recognition and the translation **input** are reliably correct; the variance is the
  translation **output** (vLLM serving non-determinism) and it is **prompt-driven**. The `###` block
  structure is preserved run to run — this is **not** cross-talk / block misalignment. The model
  just won't repeat the same target word, so it keeps the other-language copy.
- **Findings** (≈20 runs per candidate on a danger-2 input, scored "no source word left"):
  - a "count the source languages first" output-format prompt: **0/20** (always leaves DANGER + 危险);
  - reliable (**~18/20**): **flat prose**, lead *"Translate every word … even words already in
    another language"* plus an explicit *"translate each occurrence even if it produces the same word
    twice"* **with a concrete example**. The concrete example is **load-bearing**; an abstract one
    ("three words meaning 'welcome' → that word three times") works as well as a sign-specific one,
    so no curve-fit is needed;
  - putting that same rule inside a `# ROLE / # TASK / # INSTRUCTIONS` structure or a numbered list
    **breaks it** (0–4/18) — the markdown sectioning itself, not the wording. The core rule must stay
    in the flat lead.
  - **Measurement note:** compare candidates **interleaved** (A,B,C,A,B,C…), never in blocks —
    serving load is a large confound (the same prompt scored 0/20 then 12/16 in separate blocks).
- **Change.** The prompt (`data/prompts/translate_image_default/`, mirrored in the builtin in
  `app/translation/prompts/templates.py`) keeps the multilingual rule in the flat lead and appends a
  `{{category_instructions}}` slot — a section that materialises **only when populated** (an empty
  section measurably lowers reliability) — as the seam for later category-specific instructions
  (`_category_instructions()` in `app/translation/translate.py`, empty map for now). No-regression
  verified: 0 output diffs across the testset inputs × nl/zh/en (monolingual images unchanged).

### Open / parked — leftover doubling, and the VLM↔OCR-text strategy

- **Symptom.** A hint line the VLM read correctly is **split** because one OCR cell does not bind:
  the orphan becomes a leftover, is translated and rendered **separately**, and its translation shows
  up a second time over the same line (`HÆTTA` → a second "GEVAAR!"; `Local soup` → a second
  "Lokaal").
- **Two causes.** (1) **`æ`** — NFKD does not fold `æ→ae`, so the `[a-z0-9]` tokenizer splits
  `HÆTTA`→`{h, tta}`; OCR's `HAETTA` (=`haetta`) matches neither → leftover. (2) **`Local`** is a
  confident exact match to `Local soup` but `align._pick_hint`'s position guard rejects it (the
  y-based position estimate is unreliable in the horizontal menu-icon row).
- **Why the easy fixes are wrong.**
  - A **global `æ`-fold in `tokens._normalize` breaks `danger-1`**: it turns the hint token into a
    clean `haetta`, and `danger-1`'s **misread** `HATTA`→`hatta` then fuzzy-matches it (ratio 0.90)
    → binds to the title. The `æ`-split currently **shields** `danger-1` (fragments too short to
    fuzzy-match). The two photos read the same glyph differently — `danger-2` expands `æ`→`ae`,
    `danger-1` drops it to `a` — so no single normalization catches both.
  - A **position-guard skip for confident-single cells breaks `kassabon` / `adv-budgets`** (the
    guard is load-bearing for dense receipts/lists).
  - **Tooling note:** verify these with the real regression harness (`run_variant` + re-OCR on all
    fixtures); a hand-rolled cell-id signature missed the `hint_index` leftover→title flip that
    `run_variant` caught. OCR params are not a lever either — raising `text_det_limit_side_len` did
    not change the `æ` reading (stable per photo) and regressed `FJARAN`→`EJARAN`.
- **Designed, not built.** The discriminator is **exact-vs-fuzzy**: normalize the hint for the
  **exact** match (`danger-2` `HAETTA` binds exact, score 1.0) but keep the **fuzzy** fallback on the
  **original** `{h, tta}` tokens (`danger-1` `hatta` reaches a line only via fuzzy → still fails →
  stays leftover). Alternative: a **spatial gate** — bind an orphan only when it is adjacent to an
  already-bound cell of that VLM line (`danger-1`'s lone heading has none). Both are
  `danger-1`-safe by construction.
- **Parked (by agreement).** Leave as-is for now (quality already ahead of off-the-shelf translators
  on these signs). Design the general approach once we have **several hard-language signs where the
  VLM text and the OCR text diverge** (`æ/œ/ß/ø/þ/ð` — Icelandic/Danish/Norwegian/German/French),
  then lean on the VLM (which reads these correctly) to repair OCR-misread leftovers.


## `circus.jpeg` (tiny web image 307×164, red warning banner, en→fr)

### Fixed — junk strip at the banner top under the Tier-2 inpaint fill

- **Hallucinated shapes along the top edge of the red bar** (`erase_fill_mode="inpaint"`).
  The erase mask covered nearly the whole image (the WARNING quad spans y1–y50 of a ~52px
  banner), leaving ~1px of red context above; at 1:1 the model reads glyph-scale features of
  such a thumbnail-sized crop as texture to continue, and drags the JPEG edge junk across the
  fill. Fix: `inpaint.work_scale` upscales a crop whose **short side < 320px** toward the
  model's training scale (bicubic in, area out), **capped at 2×** and never past the pixel
  budget. The cap is load-bearing: on this image ~2× fills clean, but 3×/4× wash out to grey —
  the hole outgrows what the context ring supports. Masking the border sliver instead
  (extending the mask to the image edge) was tested and did not help. Validated: circus clean,
  `bullets-dashes.png` (226×223, second small fixture) clean, short side ≥ 320 takes the old
  path bit-for-bit.

### Open / parked — Tier-2 fill performance headroom, and a faint haze

- **Residual:** at 3× zoom a faint greenish haze remains across the reconstructed banner top;
  invisible at 1:1. Revisit only if it shows on a real image at viewing size.
- **Parked: CPU-side blend cost.** The inpaint delta over flat is small (~<0.5s on a 12Mpx
  photo) and almost none of it is GPU (forward ≈ 130ms at the 1.5Mpx budget). The rest is
  full-crop CPU work in `inpaint.inpaint_mask` — on a mask spanning the whole photo the crop is
  the whole image, so the fill upscale, the feather blur and the float32 blend each traverse
  ~12Mpx × 3 channels. If it ever matters: (1) limit the feather+blend to the mask's bounding
  rows instead of the full crop; (2) per-region crops (one forward per text block — less resize
  traffic AND sharper local texture; also the known quality follow-up for far-apart small lines
  on huge photos); (3) integer or GPU-side blend. Not worth the complexity at <0.5s.


## `items-levels-2.png` / `adv-budgets.jpg` — Tier-2 inpaint border and small-crumb fixes

### Fixed — dark corner smudge (`items-levels-2`, top-left)

- The "1." line's erase quad clips into the image corner; a hole touching the input border
  has no context on that side and the fill comes back unanchored (dark smudge). Fix:
  `inpaint.border_pads` mirror-pads a side (reflect-101, mask mirrored too) when the crop
  sits on the image border there, the mask reaches it, AND the mirrored strip is safe to
  mirror. Both guards are load-bearing, measured on the circus banner:
  - **anchored** (≥25% unmasked): mirroring a near-fully-masked strip (circus top) just
    enlarges the hole;
  - **near-uniform** (unmasked max channel std ≤ 20): mirroring a strip with a distinctive
    feature plants a copy next to the real one, and the Fourier-convolution model reads the
    pair as a period to REPEAT — circus's mounting holes became a dot row across the whole
    bar, seeping down through the fill (left/right strips: 30–34% unmasked but std ~100).
    Featureful borders keep the model's own boundary handling.

### Fixed — erased-glyph crumbs bleeding back through (`adv-budgets`)

- Two mechanisms, both inpaint-only (flat was clean): (1) the paste feather (Gaussian on the
  mask alpha) also weakened the mask INTERIOR — a 3px residue crumb never reaches alpha 1 at
  its centre, so the original ink blended back in; now the feather is outward-only
  (alpha clamped to 1 inside the mask). (2) on crops above the pixel budget the mask was
  downscaled with NEAREST, dropping 1–2px crumbs — the model then saw that ink as context
  and handed it straight back; now the mask resize is coverage-preserving (INTER_AREA > 0).

### Open / parked — dense overlay ink next to the hole (`adv-budgets` status-bar strip)

- Around the status-bar icons (alarm/bluetooth/wifi/battery, overlapped by the ghost
  heading) the reconstruction grows dark halos / re-invents glyph-ish smears: the hole is
  surrounded by dense unmasked icon ink and the model continues those shapes inward. This is
  acceptance-criterion territory (boundary overlays), not a mask bug — a whole-image diff vs
  flat shows the remaining dark deltas confined to that strip. Much reduced by the ground
  router below (most of the screenshot now takes the flat paint); small smears remain at the
  icon-adjacent jobs that route to the model. Candidate if it must improve: per-region crops
  at full resolution (sharper shape termination), the same follow-up parked under
  `circus.jpeg` above.


## Tier-2 inpaint — ground router: designed ground stays flat (circus/danger-1 wash-out)

### Fixed — unstable model reconstruction on designed flat/solid ground

- With the whole erase mask sent to the model, designed graphics reconstructed unstably
  run-to-run: washed-out streaks through the solid banner (circus, three consecutive runs,
  three different failures, seeping down through the bar), muddy bands in the white field
  below it, and a grey fill instead of the solid red panel behind `100°C` (danger-1). Root
  cause: a near-total hole on flat/solid ground gives the model almost no anchor, and OCR
  quad run-variance moves what little context there is — while the flat paint is right there
  BY CONSTRUCTION. Fix: `render._needs_model_fill` routes per job. The ring around the quads
  is split into side bands (above/below/left/right), each band into segments along the line;
  the ground is flat-safe when within every band the segment medians agree (Δ ≤ 20 per
  channel) — a designed band or panel is constant ALONG the line even when the sides differ
  from each other (red band above, white field below). `erase_fill_mode="inpaint"` is now a
  hybrid: flat paint by default, model only where the ground actually varies.
- Measured routing across archetypes (spread p75/max): circus 8/17 → fully flat; menukaart
  8/30 → 1 job to the model; adv-budgets 6/226 → 2 jobs; kassabon 14/46 → 8 crease/shadow
  lines to the model, evenly lit lines flat. Verified live: circus solid and stable over
  three runs, danger-1 red behind `100°C`, kassabon crumples still reconstruct, menukaart
  clean.
- Named limit: a designed boundary running through a line's LENGTH (a line spanning two
  panels) reads as varying and goes to the model — the safe direction. And jobs boxed in by
  other text (no ring) default to flat.


## Parked: VLM serving non-determinism (after the `*…:*` prompt lock-in)

The grouping prompt is locked (`*<t|h|b|m>|<font>|<size>pt|<weight>|<l|c|r>:*` single-star label
+ "bullet = a NON-alphanumeric marker"); the parser absorbs the wrapper/code/marker drift. What
remains is the model still varying its *structural* choices run to run on the same image + greedy
decode — not a wrapper the parser can normalise. By agreement these are parked (prompt/model is the
cause, not our code):

- **kassabon — `JOUW VOORDEEL. 2,44` field split is intermittent.** Some runs emit it as a table
  row (`JOUW VOORDEEL | 2,44`, the amount lands in the BEDRAG column); others as one text line
  (`JOUW VOORDEEL. 2,44`, the amount kept in place as a `translate:false` member, not column-
  aligned). Same image, same model — the `|` between label and amount comes and goes.

- **unit-count wobble (no leaks).** `bol-philips` occasionally collapses to a couple of units, and
  `weather-2` / `book-cover` swing ±1–2 units between runs. The dense `weather-1` multi-day table
  is the recurring case: the model merges/splits its rows unpredictably. No leak — the text that
  survives is clean — but the element count is not stable. A real fix is structural (constrain row
  merging), separate from the prompt/parser work.


## Idea: a category-conditioned grouping prompt (instruction load destabilises the secondary labels), 2026-06-20

A focused measurement of the secondary label fields (font_family, alignment, `|` field
structure) — the ones the parser cannot normalise away — calling the 26B-A4B grouping VLM N
times per image on the same input at greedy decode (`temp=0, top_k=1`), comparing the live
("full") prompt against trimmed variants. The harness is throwaway (lived in `/tmp`); the
findings are the point.

**The secondary labels are unstable under the full prompt, stable under a lean prompt.**

- `nike-ad` body paragraph — **font**: full prompt → Helvetica ~19–20/20 (consistent but the
  sans guess, not the serif the page uses); a **lean** prompt (rules 2.1 icons / 3 tables / 4
  field-values / 4.1 bullets removed, typography rules kept) → **Georgia 20/20**. So overfeeding
  does not make the font noisy here, it shifts the commitment to a (wrong) sans.
- `nike-ad` body paragraph — **alignment** (regardless of font): under the full prompt the
  centre-fraction *wanders between batches* — 100% (n=10), then 25% / 30% / 60% (n=20 each) on
  reruns; under lean it was 20/20 centre. A swing that large under identical input is not
  in-batch sampling (greedy), it is serving-level non-determinism (vLLM continuous batching +
  NVFP4 narrow margins) whose magnitude is load/time dependent — so isolated runs *understate*
  what production (grouping batched with translation traffic) sees.
- `kassabon` product rows — **`|` field count** per row: full prompt `{2:6, 3:2, 4:1, 5:1}`
  (KARNEMELK/GORGONZOLA/… all flap together); dropping one clause tightens it to `{2:9, 3:1}`.
  This is the wobble that feeds `_field_pairs` / `_split_table_row` and changes how the row is
  column-split run to run.

**Mechanism — the numbered-step-badge example is the receipt destabiliser.** The d7ce649 icon
example ("a numbered step badge — a digit inside a circle/disc") was added for ONE image
(`return-shipment`, an instructional/app screenshot), where it reliably strips the circled step
numbers (0/6 leak with it, 4/6 without). But it is the only icons example about **digits**: it
injects a "digits can be ignorable non-text" concept into a global prompt run on digit-heavy
images, and on a receipt that collides with the quantity/price columns → the `|` structure
wobbles. Removing just that clause tightens the kassabon pipes (above). The rest of the icons
rule (calendar/gear/magnifier/logo — unambiguously non-text) is free; this one is not. Note the
`nike-ad` instability is **not** cleanly this clause — removing only the badge example did not
steady nike alignment (still ~50/50); only the full lean did — so nike's wobble is broader
instruction load, not this example. (Caveat: single-batch evidence on a wandering distribution.)

**Direction — condition the prompt on image category, don't fatten one global prompt.** Keep the
general icons rule everywhere (it is load-bearing); ship the digit-sensitive badge example only
in the bucket where it applies (instructional / app-UI like `return-shipment`), not in the
receipt or ad buckets. Smallest first step: make just that one example conditional, not a full
prompt split. Putting both lean variants in one prompt with an "exit" at the top does **not**
recover the lean benefit — self-attention is global, all tokens stay in context and the model
must actively suppress the wrong branch (a fresh near-boundary decision). The benefit only comes
from not sending the irrelevant tokens, i.e. routing **outside** the model.

**Router cost (for the routing step).** Dominated by image *prefill* (vision tokens ∝ pixel
area), not decode — so MTP / output throughput is irrelevant for a one-token classification, and
because the 26B is an A4B MoE (~4B active/token incl. prefill) a separate dense 2B buys only
~2× active-FLOPs while costing a second resident model + VRAM contention on the 16 GB card. Best
effort/reward: route through the **already-warm 26B on an aggressively downscaled thumbnail**
(coarse bucket reads on gestalt, not glyphs; the model already emits `*Image classification:*`),
one classification token out — a few % of the real grouping call. A discriminative CLIP-head /
small CNN is the sub-ms floor if even that matters, at the cost of a new component.

**Separate track — tune the grouping image-token cap.** The full-res grouping call is the
heaviest stage, because dynamic high-res tiling on big images explodes the vision-token count. A
cap cuts that AND reduces cross-image variance (images arrive at wildly different resolutions →
different tiling → different behaviour). This is already capped server-side for our model: the
pool sets `vllm_mm_processor_kwargs.max_soft_tokens: 560` on the gemma-4-26b-a4b NVFP4 serve
entry (the Gemma equivalent of the `max_pixels` / `max_tiles` knob the Qwen-style models in the
same config use). So the grouping image is mapped to ≤560 image tokens before prefill regardless
of input resolution — prefill is already bounded, and the font/alignment/`|` wobble above already
happens *at* 560. This "track" is therefore not a pre-resize in `_data_uri` but **tuning that
560** (a clamp from above; lowering it cheapens prefill but, since 560 is already the ceiling the
model sees, almost certainly costs the resolution-sensitive axes). A separate router path would
instead want its own entry with a much smaller cap (~64–128). Risk lands precisely on the resolution-sensitive
outputs — font guess, `|` column discrimination, and the hint→OCR token matching the aligner
relies on (a worse VLM reading → more cells fall to leftover). OCR stays full-res (authoritative
for text/boxes); only the VLM image is the variable. Test with a resolution sweep, prompt fixed,
measuring prefill latency vs the font/alignment/`|` stability metrics above plus hint→OCR
alignment success, to find the sweet spot (which may sit close to full-res).

**Routing — the worked-out shape (parked, nothing built).**

- **Hard dependency: the grouping VLM runs before OCR and routes it** (`translate_image.py`:
  hint → `resolve_ocr_language(hint.units)` → OCR). So there is no free pre-grouping signal
  (OCR text isn't available yet) — category-conditioning the grouping prompt needs a dedicated
  **pre-classify call**. Reordering OCR-first to route on its text would break the hint-routes-OCR
  design, so that is not a small change.
- **Phase 1 (minimal vertical slice):** add a `_USER_INSTRUCTION_LEAN` (rules 1 importance /
  2 reading-order / 5 font / 6 alignment + the OUTPUT-FORMAT block + the *bare* icon line; drop
  3 tables / 4 field-values / 4.1 bullets). Same strict label format → **parser unchanged** (no
  `|`/`@blt` on prose images is fine, the parser simply sees none). Route **conservatively**:
  lean only on a confident prose/display category, everything else (table / UI / unknown / low
  confidence) → the full prompt, so current behaviour is the default/fallback (bounded downside).
  Record the category + chosen prompt in the debug record.
- **Router options (undecided):** (a) **thumbnail through the warm 26B** — one classify token on
  a downscaled image, reuses infra, ~few % latency, generative but the coarse bucket is a
  high-margin decision so serving-noise won't flip it; (b) **discriminative CLIP-head / small CNN**
  — sub-ms, deterministic (argmax over logits), at the cost of a new component to train/host.
- **Bucket options (undecided):** minimal **2-way** (lean prose/display vs full default) first —
  smallest validation surface, captures the measured nike win; richer **3-way** (prose / table /
  ui, with the numbered-badge example moved into the ui-only prompt) only later.
- **Why not rush the 3-way:** the badge/step-marker handling is fragile at the edges — **sub-
  numbered steps (3a / 3b)**, lettered steps, or nested markers don't fit a single "circled
  digit" rule and can reintroduce the same leak/instability the badge clause caused. Keep buckets
  minimal until that is worked through. Parked by agreement; larger fixes take priority.


## Idea: re-OCR the rendered image as a render-fidelity check (and refine loop)

Re-running OCR on the *rendered* (translated) image and comparing its cell boxes to the source
cells — box for box — is a clean, concrete fidelity signal: it measures whether the renderer put
the translation where (and at the size) the original sat, which pixel-diffing muddles. It pinned
the `circus.jpeg` title cleanly: the rendered "WAARSCHUWING" cell is `h32 @ top9` against the
original "WARNING" `h45 @ top3` (the longer Dutch word pt-shrinks to fit the width, so it no longer
fills the red bar), where a pixel diff drowned in the legitimate white glyph pixels.

Because OCR (warm: ~140ms median, ~440ms on the 120-cell kassabon) + render (~100ms median, ~550ms
on kassabon) are cheap next to the grouping VLM (~1.1s median, 4s on kassabon) + LLM translation
that dominate latency — all measured on the dc1 dev box, less on the dc2 target — a **render →
re-OCR → compare to source boxes → adjust → re-render** loop is affordable for a few iterations:

- per unit, compare the rendered cell box (top / height / position / width-ratio) to its source box;
- flag a mismatch (e.g. a title >20% smaller, or shifted out of its band);
- adjust (relax the condense floor so a heading fills the band height instead of pt-shrinking,
  re-centre vertically, allow some overrun) and re-render;
- stop at a tolerance or a max iteration count.

Caveats: OCR *geometry* is reliable, but the rendered text itself can mis-read — match on box
position/size, not the exact string. Needs an explicit adjustment policy + a convergence stop or it
oscillates. Note this targets the same root as the faint bottom-edge streaks (the rendered cell not
sitting exactly on the source cell), without erase-margin tuning.


## Idea: cover OCR-clipped descenders by source-text letters, not colour

The OCR box ends ~3–5px above a descender's tip (g/j/p/q/y), so the original's tails bleed as faint
bottom-edge streaks under a translated body line. Extending the erase down covers them, but that
grows/notches a tight coloured band (the red WAARSCHUWING bar on circus). The two ends — a tight
erase (clean band, streaks bleed) vs a wide erase (streaks gone, band grown) — are a genuine trade
(swapping one regression for the other; observed flipping back and forth on circus vs adv-budgets).

Clean discriminator, no colours / no pixel reading: extend the bottom ONLY for a line whose SOURCE
text contains a descender letter (g/j/p/q/y), by one font ``descent``. All-caps / digit lines (a
WARNING title, "2025") have none → stay tight → tight bands stay safe. A descender line that does
sit in a band stays within it (a band is built on the line height, which already includes the
descent). Parked for the eventual erase rebuild — text-based, self-discriminating, no margin tuning.

**Follow-up (2026-07-05): built and measured — parked stays right.** Implemented in three
variants against the 46-fixture harness. (a) As written (per-word, text-only): 15/46 fixtures
change — per-word bottoms NOTCH the erase bands on textured paper (nike body), and the extension
eats undetected content sitting below a line (kassabon's un-OCR'd "13"/"4397" digits: hidden-miss
class). (b) Per-LINE extension (straight band edge) + flat-group gate + an ink-fraction guard on
the strip below: still 11/46 — nearly every flat body line carries a descender, so the mechanism
inherently retouches most renders, and the guard is untunable (real tails are sparse, but dense
neighbours and texture sit in the same fraction range). (c) The user's sharper
criterion — "extending is free as long as the bite is the same background" — implemented as three
pixel gates (foreign colour outside own glyph columns forbids; DENSE ink under own columns forbids;
only sparse tails erased): still 12/46, and the killer stands: thin real content directly below a
line (the receipt's handwritten "4397", a weather value losing its top) is OPTICALLY IDENTICAL to
descender tails within a descent-deep strip — no local pixel rule separates them, and re-OCR
segmentation churns broadly (nike) whenever bands grow. Conclusion after three variants: the
discriminator is right but the pay-off needs an erase that can afford to grow — with an
inpaint-grade fill (Tier-2/LaMa) the extension becomes visually free and this text-based rule is
the correct scope for it. Until then the slot sweep covers superseded lines (ghost words +
box-undershoot remnants there); streaks under TILED lines remain a named limit of the flat-fill
erase. Do not attempt a fourth local-heuristic variant.

**Generous-erase experiments for the residue (2026-07-05) — TRIED AND REVERTED.** Diagnosis
first: the residue is a descender/comma tail surviving below PaddleOCR's tight detection box.
The knob `text_det_unclip_ratio` does grow the box over the tail (measured: 2.5 moves the box
bottom past it) but inflates box height 58->79px, and the renderer sizes the font from box
height, so text comes out ~36% bigger — not free. The production translation apps show no
residue not from better OCR but because they erase generously (some paint clean white rounded
rectangles, no inpaint) and accept a visible card on texture; the residue is a
flat-fill-STINGINESS artifact.
Two generous-erase variants were built: (a) fill the gap BETWEEN a group's consecutive lines
(safe — that band is the group's own footprint; kassabon stayed bit-identical, only 5 benign
re-OCR re-segmentations), and (b) full per-line SLOTS reaching halfway to each neighbour plus a
descent past the block edge. Variant (b) was REJECTED on sight: on `cartoon` it ate the speech
bubbles' outlines and a chunk of the image — the slot reaches past the text into surrounding
GRAPHICS that are not another text unit, so the foreign-box guard (which only knows text member
boxes) does not protect them. Reverted entirely. Lesson: a group's inter-line band is safe, but
anything reaching toward a group's OUTER edge can hit non-text image content the pipeline has no
box for. The residue stays a named flat-fill limit; a real fix needs an inpaint-grade erase (so
over-reach is reconstructed, not destroyed) — which is exactly why the big tools inpaint or
overlay opaque cards.
