# Inline-math islands — design

Translating born-digital documents whose prose carries inline mathematics
(scientific papers, technical reports). Ratified 2026-07-19; measured evidence
from two arXiv documents run through the pipeline as-is.

## The problem, measured

The text-layer extractor (`app/pdf/textlayer.py`) drops a whole line as soon
as one span uses a math font — deliberate slice-A policy: math cannot be
re-typeset (embedded subset fonts, stacked layout), so its pixels stay. On
office documents that is rare. On a scientific paper it produced a compound
defect, observed on a 15-page NIPS-style paper (arXiv 1706.03762):

1. **Double coverage at micro size.** The grouping VLM reads the page image
   and hints the *whole* paragraph; the translation therefore covers the whole
   paragraph; align maps it onto the few surviving (math-free) cells; the fit
   then crams all of it into that smaller footprint. Result: the dropped line
   stays in English at full size *and* the complete translation renders
   unreadably small around it — the same content twice, in two languages, in
   two sizes. ~22% of the document's characters sat in dropped lines (30-55%
   on formula-heavy pages).
2. **Literal TeX in running text.** The VLM transcribes formula images as
   LaTeX; the translator passes it through; the render shows `$d_k$`-style
   macros in body prose.
3. **Total loss on pure-CM documents.** The font filter
   (`_MATH_FONT_RE`) classifies `CMR<digit>` — Computer Modern *Roman*, the
   body face of classic LaTeX — as math. A pure-CM document (measured:
   arXiv math/0211159, body font CMR12) extracts **0 cells**: the whole
   document would pass through untranslated.

What already works and must stay working: math-free paragraphs re-typeset
cleanly (serif, justified); display formulas and vector figures stay
untouched; tables survive with their numbers.

## Principle

Math is never translated and never re-typeset — it is source ink. But the
*line* around it is prose and must translate. So the line stays one cell and
each maths run becomes an **island**: a pixel crop from the source raster
that flows as an unbreakable box inside the re-typeset translation. The
translator sees a placeholder; the reader sees the original formula
typography inside the translated sentence.

This is also the class where both external archetypes fail structurally:
re-typesetting systems mangle or drop formulas (they cannot set the subset
fonts), raster-overlay systems keep the formula but degrade the prose around
it. Source-ink transplantation inside re-typeset translation is the hybrid
middle this pipeline already is. The benchmark's anchors axis measures the
outcome directly (digits inside formulas).

## Phases

Each phase is separately shippable and validated (unit suite, image
regression 43/43, pdf regression, a live run on the measured paper).

**Phase 0 — squeeze preserve-floor (render side, declared sizes).** The
pt-shrink loop in `app/replacement/layout/planning.py::_plan_group` already
gives up (`return []`, pixels stay, no erase) when even the minimum size
cannot fit — but the floor is absolute (`_MIN_RENDER_SIZE`). Add a *relative*
floor for groups whose every plane declares its em size (text-layer cells): a
fit below ~0.55× the declared size is not a rendering, it is a duplicate
caption under leftover ink; give up instead. This kills the worst artifact of
the compound defect on its own — pages degrade to "line stays in source
language", never worse than today. Measured scope limit: ink-derived targets
(the OCR path) are exempt — the scanned-document fixtures legitimately ship
deep-shrunk small label work, so extending the floor there is a separate
decision against those baselines, not a free generalization.

**Phase 1 — font reclassification (only).** The current taxonomy is wrong.
Math = `CMMI* / CMSY* / CMBSY* / CMEX* / MSAM* / MSBM* / *Math*`. Text =
`CMR* / CMBX* / CMTI* / CMTT* / CMSS* / CMSL* / CMCSC*` (Roman, bold,
italic, typewriter, sans, slanted, small-caps are text faces). The drop
trigger itself is unchanged in this phase: a line containing a true-math
span still drops whole. Direct win: the pure-CM class goes from 0%
extraction to its prose; CM headings and CM-italic theorem text survive.
Sequencing decision (moved out of this phase during build): the
majority-math drop rule and the sandwich absorption (the `= 8` inside
`h = 8` — measured span pattern: prose spans, then CMMI/CMR runs, then
prose spans) belong with the islands in phase 2. Without islands, keeping a
minority-math line would erase the source's math typography and re-typeset
it as plain text — a fidelity regression as an intermediate state. Every
phase must be shippable on its own.

**Phase 2 — islands in extraction.** A maximal run of math spans (plus
absorbed sandwiched text spans and spaces) becomes one island per line:
`{bbox_px, height, baseline offset}`; the cell text carries `⟦M1⟧` at that
position; the cell carries the island list as data — same pattern as the
existing `marker` field, source-agnostic, so no PDF concept leaks into
shared code.

**Phase 3 — translation contract.** One short prompt rule ("keep `⟦Mn⟧`
tokens exactly as they are") plus a deterministic gate: the placeholder
multiset in the reply must equal the input's, else the unit falls back to
preserve (never worse than today). Re-ordering is *allowed* — linguistically
correct; the island follows its slot in the translated sentence. Side
benefit: the translator never sees math text again, so the literal-TeX
artifact disappears.

**Phase 4 — render transplantation.** In the reflow an island is an
unbreakable "word" of fixed width (the same machinery that keeps URLs
whole), cropped from the source raster *before* erase, scaled by the
target/source size ratio (known: text-layer cells carry `size_px`), pasted
baseline-aligned.

**Phase 5 — measurement.** The measured paper (and a pure-CM document) join
`testset/pdf`; benchmark before/after; a document fixture once quality
stands.

## Build notes (2026-07-19, phases 2-4 live)

Three findings from the first live runs, each now part of the design:

- **Island units translate per unit from their cell text.** The structured and
  hint-line translation paths re-translate the VLM's hint lines, which carry
  the VLM's own (TeX) reading of the math and never the ⟦Mn⟧ tokens — on the
  first live run every island unit fell through the token gate to preserve.
  Units whose source text carries tokens now skip the hint paths and go
  per-unit on the token-bearing cell text (measured after the fix: 35/35
  units translated with the exact token multiset, zero gate hits).
- **Cell geometry comes from the prose glyphs.** An island glyph (a radical)
  reaches above the text band; a cell bbox inflated by it shifts the render
  anchor a band up and two lines print on top of each other (measured twice
  per formula-dense page). The cell bbox/polygon now unions only prose
  glyphs; the islands' own ink is erased through their recorded boxes, added
  to the plane's erase quads.
- **Prose share and dominance decide, extraction-side.** The line share cap,
  the minimum prose words and the dominant-text-family test (operator names
  of a display equation are set in the math ecosystem's roman, not the body
  face) together classify formula lines; the CM text faces collapse to one
  family so pure-CM prose passes.

## Named limits

- Tall inline math (fractions, stacked scripts) can exceed the line pitch:
  v1 squeezes the island into the line box. A very tall island's source crop
  can also catch a sliver of the neighbouring line's ink (its glyph box
  genuinely overlaps that band). Named limit, not a blocker.
- A display equation's annotation line with genuine body-face prose ("where
  head_i = Attention(…)") passes the prose tests and re-typesets with
  transplants — content-correct, typographically mediocre.
- The ``translategemma`` mode has no instructions channel for the token rule;
  the deterministic gate degrades its island units to preserve.
- The VLM hint transcribes math as TeX and will not match the placeholder
  text — the prose tokens must carry the align match. Needs a probe on the
  measured paper.
- Target-language word order around a fixed island can stay awkward; the
  translator decides, the gate only guards preservation.
- The author-grid duplication on the paper's title page is a separate align
  defect, outside this design.

## Out of scope

The OCR path and the native render backend are unchanged; math itself is
never translated.
