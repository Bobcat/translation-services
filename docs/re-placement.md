# Stage #8 — Re-placement: design directions

Goal: render the translated text back into the image so the output is a *usable
translated image*, not just the debug overlay + JSON. Per translation unit: remove
or cover the original text and put the translation in its place, natural enough for
the travel use case (signs, menus, receipts, labels).

This document captures the directions and the chosen first path. It is a design
note, not a spec — it should grow as we build.

## The two sub-problems

Every approach (from SRNet, "Editing Text in the Wild") splits into:

1. **Erase** — remove the original text and restore the background underneath.
2. **Render** — draw the translation in the freed region: font, size, colour,
   alignment, orientation; `flow` reflow vs `field` fit.

## Directions (erase × render)

| Tier | Erase | Render | Fit for us |
|---|---|---|---|
| **1 — simple replace** *(model-free)* | sample the local background colour, fill the unit's member bboxes | PIL: draw the translation, fitted to the region, at the polygon angle | fast, no extra model; good for the **flat surfaces** our text sits on (sign panels, menu paper, receipt paper) even inside scene photos. Weak on textured/photographic text backgrounds. |
| **2 — inpaint + render** | a text-removal model (**LaMa**: local, promptless, latency-friendly; SD as quality/fallback) | same PIL render | nicer erase on textured/photographic backgrounds; one more model to host. |
| **3 — scene-text editing** | end-to-end **SRNet / MOSTEL / diffusion-STE** | model renders text in the *exact* original style | best look, but heavy, word-level, trained on synthetic, struggles with length mismatch and non-Latin. Research-grade; out of scope for now. |

Orthogonal to "replace": **overlay** (translation as a clean, semi-opaque box over
the unit — original pixels kept, background-agnostic) and **callout / side-by-side**.
These sidestep erasing entirely and are robust fallbacks when inpainting fails on a
real photo.

## What carries over from our pipeline

- OCR cells stay **authoritative** for text + **bbox + polygon**.
- Translation units (`flow` / `field`), each with members (cell_id, bbox, `translate`)
  and a `translated_text`.
- `translate: false` members (bare price / URL / code) are **left untouched** — never
  erased, never re-rendered. This is the "mask only what we replace" rule (see below).

## Learnings from manga-image-translator

(Codex' analysis: `~/manga-image-translator-analysis.md`; repo: zyddnys/manga-image-translator.)
Its post-translation stages *are* our #8 (`mask refinement → inpainting → rendering`).

- **Mask only what we render**, after filtering/translation — don't erase regions
  with no useful translation. Matches our `translate:false` / `ignored` handling.
- **Render → RGBA box, then homography-warp to the target polygon** — handles
  rotation/perspective. We already have the cell polygons.
- Their OCR emits **fg/bg colour**; PaddleOCR does not → we need a **colour sampler**
  (estimate text + background colour from the cell region).
- **Replace can fail on real photos** (texture, lighting, perspective) → keep
  overlay/callout as a fallback mode.
- **Debug artifacts per stage** (mask, inpainted, rendered) make tuning tractable —
  we already emit per-stage overlays; continue that for #8.

## Data-model extensions we'll need

- Per-unit/member **render metadata**: `fg_color`, `bg_color` (sampled), `angle`
  (from the polygon), and a `render_mode`.
- **Target region**: union bbox + polygon for `flow` (reflow); per-member bbox for
  `field`.

## Open challenges (render side)

- **Length mismatch** (translation longer/shorter, e.g. "ACHTUNG!" → "WAARSCHUWING!")
  → font auto-fit / wrap within the region.
- **Font unknown** → a neutral signage/system font; estimate weight/size from the box.
- **Colour sampling** robustness (shadow, reflection, gradient backgrounds).
- **Orientation / perspective** via polygon homography.
- **`flow`** (reflow the translation across the union region) vs **`field`** (fit into
  the single bbox, preserving the column).
- **Style/caps** preservation where it matters (an all-caps sign → all-caps output).

## Chosen path

1. **v1 — Tier 1 simple replace (model-free).** Per unit: sample the local background
   colour, fill the member bbox(es), render the fitted translation at the polygon
   angle. `field` → own bbox; `flow` → reflow over the union. `translate:false`
   members and `ignored` cells are left untouched. Emit a re-placement debug overlay
   and the final rendered image as artifacts.
2. **Upgrades:** Tier 2 LaMa inpaint for textured backgrounds; overlay/callout mode as
   a robustness fallback; a better colour sampler.

## v1 status (Tier 1, built)

Implemented in `app/replacement/` (`color` sampling, `fit` font-sizing, `render`)
and wired into the pipeline; the rendered image is exposed as the **`rendered`**
artifact. Per unit: take the union region of the translatable members, sample the
background colour, erase that region, draw the fitted translation (single-line for
`field`, wrapped for `flow`) in a contrasting colour. `translate: false` members and
ignored cells are left untouched. Verified on the testset (nike clean; menu dishes
translated with prices kept in their columns; sign bodies translated).

Applied refinements (after the first cut):

- font size from the **original cell height** (not the region) → consistent,
  natural sizes instead of huge text in roomy regions;
- **two-pass** draw (cover all regions, then draw all text) → no original text
  peeking through behind a later unit;
- left-aligned.

### Tier-1 v2 — background-matched box (Google Lens look)

The user anchored the quality bar to **Google Lens / Translate camera mode** on the
`menukaart` photo. Reverse-read, Lens does exactly Tier-1: per text block, an
**opaque, rounded box filled with the locally-sampled background colour** (so it
reads as erased on a flat surface) with the translation drawn **horizontally**
inside. Crucially it does **not** correct perspective — the menu is tilted, the
boxes and text are axis-aligned, the box just covers the slanted original. And no
inpainting. So the model-free ceiling is higher than the first cut suggested; the
two real gaps were our own bugs:

- `color.sample_region_colors` took the **median of the whole box**, which mixes in
  the text strokes → a muddy fill. Now it medians a **thin border ring** of the
  region (`color._border_pixels`) → a clean background colour that blends (paper on
  the menu, blue on the afstand sign, red on the danger sign).
- the renderer covered the region with a sharp rectangle and could overflow. Now
  `render._plan_unit` draws an **opaque rounded box sized to cover the original *and*
  fit the translation**, with padding; single-char/empty units are skipped (kills
  the stray-`i`/`E` boxes). Verified offline on the testset via the source-text
  stand-in harness (no live translation needed to judge erase/box/fit).

### Tier-1 v3 — polygon-aware (perspective), DeepL bar

Correction to v2: the reference photo is **DeepL** image translation, and the v2
"perspective is secondary" call was wrong. DeepL renders **regular-weight text at the
original size, rotated to follow the page tilt**. Two render bugs were exposed on the
tilted `menukaart`:

- **size** — sizing from the axis-aligned bbox height is wrong: a tilted line's bbox
  height is inflated by `h·cosθ + w·sinθ`, so long/slanted lines went huge and short
  ones tiny *for the same original size* (the user's "Portion of fries" vs "Pointed
  cabbage and carrots"). Fix: the OCR **polygon** gives the tilt-invariant true line
  height (`geometry.line_height`); size from that.
- **angle** — text was upright. Fix: the polygon gives the angle; render each unit to
  a flat RGBA tile and **warp it onto the oriented region with OpenCV**
  (`render._composite`, `cv2.getPerspectiveTransform`).

Plumbing: the OCR cell polygon now propagates into `UnitMember.polygon`
(`grouping/units.py`, `grouping/align.py`); the renderer reads it (falls back to the
bbox quad when absent). Font is regular DejaVu, sized `true_height * 0.9`. Block
height is bounded to the original region (+slack) so a longer translation **shrinks**
rather than exploding into one giant word per line (the nike headline failure).
Verified live (the polygon path can't be exercised by the offline source-text harness
— the saved `units.json` predates the plumbing).

Known Tier-1 ceiling / next improvements:

- **Textured / photographic backgrounds**: a flat box still scars where the text
  sits on a *non-flat* surface (the `bol-philips` battery / blue circle, photo
  behind text). This is the real Tier-1 limit → **LaMa (Tier 2)**.
- **Grouping**: dense menus pack units close together, and one dish is sometimes
  split into a `field` + a `flow` (different sizes) — a grouping-quality concern, not
  a render bug. True perspective is approximated as **per-unit rotation** (oriented
  bbox), not a per-line trapezoid; revisit if strong perspective shows up.
- uniform sans font (DejaVu — no Roboto/Liberation installed), no style/weight match;
  colour is border-median bg + black/white contrast text (real text-colour sampling
  later).

## References

- [ImageTra — Real-Time Translation for Texts in Image (IJCNLP 2025 demo)](https://aclanthology.org/2025.ijcnlp-demo.1/) —
  modular toolkit with **the same stack we already have**: PaddleOCR + LLM
  translation + **LaMa** inpainting + render (font selection, colour sampling from
  surrounding pixels, orientation preservation, layout-fit for length variation).
  Code: `github.com/hour/imagetra`. Best concrete reference for #8.
- [SRNet — Editing Text in the Wild](https://www.arxiv-vanity.com/papers/1908.03047/)
- [Towards Scene-Text to Scene-Text Translation](https://www.semanticscholar.org/paper/Towards-Scene-Text-to-Scene-Text-Translation-Susladkar-Gatti/2162ca1d06f299c99e18f3e5f292bc6d529bb4a9)
- [MOSTEL — Stroke-Level Scene Text Editing](https://www.semanticscholar.org/paper/Exploring-Stroke-Level-Modifications-for-Scene-Text-Qu-Tan/b8fe2b02776208670906c89f8b8d361074fc87d5)
- zyddnys/manga-image-translator (+ `~/manga-image-translator-analysis.md`)
