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

## References

- [SRNet — Editing Text in the Wild](https://www.arxiv-vanity.com/papers/1908.03047/)
- [Towards Scene-Text to Scene-Text Translation](https://www.semanticscholar.org/paper/Towards-Scene-Text-to-Scene-Text-Translation-Susladkar-Gatti/2162ca1d06f299c99e18f3e5f292bc6d529bb4a9)
- [MOSTEL — Stroke-Level Scene Text Editing](https://www.semanticscholar.org/paper/Exploring-Stroke-Level-Modifications-for-Scene-Text-Qu-Tan/b8fe2b02776208670906c89f8b8d361074fc87d5)
- zyddnys/manga-image-translator (+ `~/manga-image-translator-analysis.md`)
