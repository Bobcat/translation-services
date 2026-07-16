# PDF translation — research and high-level design

Status: design proposal, no code yet. Written 2026-07-16 from: profiling of the
3-PDF testset, a feasibility probe on native PDF editing, a stage map of the
image pipeline, and research into commercial and open-source document
translators. Sources are listed at the end.

## 1. Goal

A `translate_pdf` workflow: PDF in, translated PDF out, layout preserved,
with a document-level quality benchmark from day one.

"PDF" is not one format. A page sits somewhere on a continuum:

- **scanned** — one big image, no text objects. Exactly our image-pipeline case.
- **born-digital** — vector text runs with fonts, positions, graphics. A
  typesetting *result*; the paragraph/reading-order semantics are usually gone.
- **hybrid** — scan plus an invisible OCR text layer, or mixed pages.

Page class is a per-page property, not a per-document one. The design routes
per page.

## 2. What the text layer actually gives us (testset evidence)

Profiled with PyMuPDF (`get_text("dict")`, font tables, redaction probe) on the
three born-digital testset PDFs (`pdf-01` simple / `pdf-02` designed brochure /
`pdf-03` scientific paper):

- **Exact text and geometry, sub-line granularity.** Spans carry text, bbox,
  font name, size, color, bold/italic flags. This beats OCR on fidelity and is
  free.
- **No reliable reading order in untagged documents.** In `pdf-02` the
  content-stream order puts list items before their headings. In `pdf-03`
  (LaTeX) the stream order happens to be correct. Only `pdf-01` is tagged, and
  tagging is rare in the wild.
- **Untranslatable spans exist and must be detected.** `pdf-03` math shatters
  into ~6000 spans of glyph soup (symbol fonts, unmapped chars); `pdf-02` has
  private-use-area icon glyphs. Both would poison translation if treated as text.
- **Fonts are a real problem even when present.** `pdf-01` uses non-embedded
  Arial next to embedded duplicates. Embedded fonts are *subsets*: they only
  contain the glyphs the source text used, so they generally cannot render
  translated text. Font substitution from our own font stack is required in
  every output mode; embedded-font reuse is an optimization, not a baseline.
- **Typographic tricks encode structure in style runs.** Small-caps as two font
  sizes ("I." + "NTRODUCTION"), rotated margin text, per-word spans. Span→unit
  assembly needs the same tolerance our OCR-cell grouping already has.
- **Designed-document text is mostly vector, not pixels.** In `pdf-02` the
  donut-chart percentages and legend labels are ordinary text spans. In a
  native output mode they are translatable without touching pixels.

Testset gap: no scanned and no hybrid PDF yet. Both classes must be added
before the page classifier and the bitmap fallback can be validated.

## 3. How others do it

Three architectures exist in production (research 2026-07-16):

1. **Raster underlay + text overlay — Google.** Render the page to an image,
   draw translated runs on top, optionally erase the original text pixels with
   a separate "text removal server" (opt-in, `enable_shadow_removal_native_pdf`
   in the Cloud Translation proto). Robust and cheap; output is a picture of a
   page — even for born-digital input, confirmed by a structural teardown of
   the output (full-page JPEG XObject with text drawn on top). No reflow;
   typeface not preserved.
2. **Editable-format round-trip — DeepL (current), Azure, the CAT industry.**
   DeepL: PDF → OCR → DOCX → translate → regenerate PDF; OCR is on the
   critical path even for clean digital text, and DeepL itself ranks this path
   below native-format input. Fit is solved as page-level font-size
   optimization with a published constraint order: keep page count > keep
   positions > keep headline/body size *ratios* > keep global size uniformity.
   Azure routes scans to internal OCR (Document Intelligence since 2026) and
   re-typesets a native text PDF with acknowledged reflow. CAT tools
   (Trados, memoQ, Phrase, Smartcat) convert PDF→DOCX and never produce a PDF.
3. **Full reconstruction from page images — DeepL's announced VLM direction.**
   Rasterize everything, extract text + layout with a VLM/layout models,
   re-typeset from scratch. DeepL has publicly committed to this as the
   successor to its DOCX pipeline. It is, stage for stage, the architecture
   our image pipeline already implements (VLM hint + layout model + OCR +
   reflow render).

The open-source paper-translator family (PDFMathTranslate/pdf2zh and its
successor engine BabelDOC) is a fourth, PDF-native variant: char-level
extraction (forked pdfminer), layout classes from DocLayout-YOLO, formula
masking by font-name/unicode heuristics, then content-stream *regeneration*
with iterative font-scale fit. Known failure modes we should not repeat:
references merged into one paragraph, tables off by default, more untranslated
leftover blocks per page than DeepL (2.85 vs 2.33 in BabelDOC's own paper),
and scanned input "solved" with white cover rectangles.

Takeaways:

- Our raster pipeline is not the legacy path — it is the architecture the
  field is converging on. The PDF work should wrap it, not replace it.
- Nobody re-uses embedded fonts; everybody substitutes and refits.
- The published fit heuristic worth copying: preserve font-size *ratios*
  between levels rather than absolute sizes, and optimize per page, not per
  text box.

## 4. What we already have

From the stage map of `app/tasks/translate_image.py`:

- **Pixel-coupled stages:** VLM hint, PP-DocLayout, OCR, render (erase/inpaint
  + draw). These need a raster — which any PDF page can provide via rendering.
- **Source-agnostic stages:** align/grouping, field geometry, translation.
  They consume `cells` (text + bbox + polygon) and hint/layout structures, and
  never touch pixels. PDF text-layer lines can be fed in as synthetic cells
  with a pt→px transform; nothing downstream changes.
- **Protection mechanism exists:** `UnitMember.translate = false` and the
  layout `PRESERVE_LABELS` gate already express "keep this region as-is" —
  formulas and icon glyphs map onto it.
- **Missing entirely:** any document/multi-page plumbing. The service is one
  raster in, one raster out.

Feasibility probe (scratchpad, not committed): PyMuPDF
`add_redact_annot`/`apply_redactions` + `insert_htmlbox` on `pdf-01` removed
only the text objects, kept the vector callout frames intact, and re-set a
longer Dutch paragraph with auto-wrap and shrink-to-fit. The native write-back
path is real; the work is in font matching and unit granularity, not in the
mechanics.

## 5. Options

### A — rasterize everything

Render each page (~160 dpi) → existing image pipeline per page → assemble the
rendered pages into an output PDF (optionally with an invisible text layer for
searchability).

- Works for every page class including scans; document plumbing is the only
  new code; all existing quality machinery and the regression harness apply.
- Output is raster: large files, no selectable text, fidelity capped by dpi.
  OCR noise on text that was already perfect in the file. Full VLM+OCR cost
  per page.

### B — hybrid: raster analysis, native text, per-page output backend

Analysis identical to A (render page → VLM hint + PP-DocLayout). Two changes:

- **Text source:** for born-digital pages, skip OCR and build cells from
  text-layer lines. Exact text, exact boxes, no recognition errors, less GPU.
- **Render backend per page:** scanned/hybrid pages keep the bitmap backend
  (erase/inpaint + draw). Born-digital pages get a native backend: redact the
  source text objects, insert translated text with our font stack. Vector
  graphics, photos and file size survive; no inpainting needed because erasing
  a text *object* leaves the background genuinely intact.

### C — editable-format round-trip

PDF → DOCX-like intermediate → translate → regenerate. Rejected: lossy
conversion we don't control, OCR on the critical path for clean text, doesn't
fit the unit-level pipeline or the benchmark story. DeepL, who runs this
architecture, is publicly moving off it (option 3 above).

**Recommendation: B, built in phases that each ship as a working A-subset.**
A is not a competing end state but B's phase 0 and its permanent fallback: any
page (or page where the native backend fails validation) can always take the
bitmap path.

## 6. Proposed pipeline

```
PDF in
  └─ intake: parse (PyMuPDF), reject encrypted, page census
       └─ per page: classify  scanned | hybrid | born-digital
            ├─ render analysis raster (fixed dpi)
            ├─ VLM hint + PP-DocLayout          (unchanged, parallel)
            ├─ text source:  OCR (scanned/hybrid) | text-layer cells (digital)
            ├─ align → field geometry → translate   (unchanged)
            └─ render backend:
                 bitmap (existing)  →  raster page
                 native (redact+insert, digital pages, phase 2) → edited page
  └─ assemble output PDF (original page sizes, metadata), artifacts per page
```

Notes:

- **Page classifier:** text-layer coverage of visible ink decides the class.
  Concrete signal needs tuning on scanned fixtures (e.g. extractable-char count
  vs OCR sample on a downscaled render). Per page, with a document-level
  summary in the response.
- **Text-layer cells:** PyMuPDF lines (not spans) map to cells; pt→px at the
  analysis dpi. Span styles (font, size, bold, color) ride along as cell
  metadata — the render stage currently derives these from pixels and VLM
  hints, so exact values are an upgrade.
- **Protected spans:** symbol/math fonts, unmapped CIDs, private-use glyphs →
  `translate: false` members; contiguous runs become preserve regions. Same
  placeholder idea as the OSS family, but decided by deterministic code, not
  prompt text.
- **Runtime:** a document request fans out to per-page jobs on the existing
  queue; the document job aggregates and assembles. Per-page artifacts
  (`grouping.json`, `translation.json`, page renders) keep the existing
  `retranslate`/`rerender` flows working per document — cached cells make
  page re-translation VLM/OCR-free, exactly as today.
- **API:** extend `/v1/requests` with `application/pdf` upload and a
  `translate_pdf` task; per-page progress in the lifecycle record. Response
  carries the document artifact plus per-page entries. (Decision needed:
  page cap per request; suggest a low cap first, e.g. 25.)
- **Perf envelope:** A4 at 160 dpi ≈ 2.5 Mpx, the testset spreads ≈ 5 Mpx —
  inside the budget the hot-path work already handles. VLM hint dominates
  per-page latency; pages can pipeline through the existing queue.

## 7. Quality benchmark (day one)

The two published layout-fidelity metrics in the field are the same idea:
run a layout parser over source and translated pages, match regions, score
bbox overlap, exclude background. DeepL's ABOR (Average Bounding Box Overlap
Ratio, segmented with the open-parse library) and BabelDOC's BIoU both do
this. DeepL also published why pixel similarity (SSIM) failed first: it can't
tell a harmless line-wrap from a dropped sentence, and background images
dominate the score. That matches our own re-OCR philosophy — compare
behaviour, not pixels.

Benchmark v1 (per document, per page; source PDF vs translated PDF):

1. **Render both at the same dpi.** Same path as the analysis raster.
2. **Layout score:** PP-DocLayout on both; match regions class-aware by IoU
   (greedy or Hungarian); score = mean IoU of matches, with penalties for
   unmatched regions. Background excluded by construction.
3. **Structural flags:** page count, image-region count, table-region count
   equal? Any drift is a flag, not a score.
4. **Completeness:** OCR the translated render; count segments still in the
   source language (leftover detection) and segments missing entirely against
   the translated-units list. We already have `reocr_rows`; this generalizes it.
5. **Geometry/typography flags:** translated text overflowing its matched
   region; font-size *ratio* drift between levels within a page (the
   published perception finding: readers notice broken ratios, not absolute
   size changes).

Translation quality itself (adequacy/fluency) stays a separate axis — text-level
scoring on extracted unit pairs, not part of the layout benchmark.

This is deliberately cheap: two renders, two PP-DocLayout passes, one OCR pass,
plus counting. It runs on every testset document from phase 0, so every later
phase (text-layer cells, native backend) must beat its predecessor on the same
scoreboard. No open-source or commercial system publishes an OCR-round-trip
completeness check — the harness we already have is ahead of the field here.

## 8. Phasing

- **Phase 0 — document plumbing + option A.** PDF intake, page classifier
  (trivial version), per-page image pipeline, raster-page PDF assembly,
  per-page artifacts, benchmark harness v1. Ships: every PDF class translates,
  including scans. Quality equals today's image pipeline.
- **Phase 1 — text-layer cells.** Born-digital pages stop using OCR; exact
  text + style metadata feed align/translate. Protected-span detection.
  Output still bitmap. Ships: fidelity up, GPU cost down; benchmark shows the
  completeness delta.
- **Phase 2 — native render backend.** Redact+insert for born-digital pages
  behind a render flag (like `width_fit_mode`); font matching against our
  stack; per-page fallback to bitmap when validation fails (overflow,
  unresolved fonts). Ships: vector output, selectable text, small files.
- **Parked:** tagged-PDF structure preservation (accessibility), embedded-font
  subset reuse, DOCX/PPTX intake, per-region backend mixing on one page,
  invisible-text layer under raster pages.

Each phase is independently shippable and benchmarked against the previous one.

## 9. Risks and open decisions

- **Font matching** (phase 2) is the largest quality risk; every vendor
  substitutes, none documents how. Our render stack already solves family/
  weight/size selection for the bitmap path; the native path must reuse those
  decisions, not reinvent them.
- **Text expansion in dense pages** (`pdf-03` class) has no good answer
  anywhere; DeepL's published constraint order (page count first, ratios
  before uniformity) is the best-known heuristic and fits our existing
  cohort/size machinery.
- **Reading order for designed documents** is why analysis stays raster-based.
  Accepting that means VLM cost on every page; a later optimization may skip
  the VLM for simple single-column pages (classifier decides). Not phase 0.
- **Classifier thresholds** need scanned/hybrid fixtures that don't exist yet.
- **Decisions needed before phase 0:** page cap per request; analysis dpi
  (160 vs 200); whether raster-page output is an acceptable v1 contract;
  testset location (`testset/pdf/`, gitignored like `testset/`).

## 10. Out of scope

Implementation. This document proposes; nothing in `app/` changes with it.
Also out of scope: non-PDF document formats, accessibility tagging of output,
translation-quality (adequacy) scoring methodology.

## Sources

- DeepL — document-translation engineering blog (DOCX pipeline, font-size
  constraint order, ABOR/SSIM methodology):
  <https://www.deepl.com/en/blog/tech/improving-document-translation> and
  <https://www.deepl.com/en/blog/built-better-document-translation>
- DeepL — VLM reconstruction direction:
  <https://www.deepl.com/en/ai-labs/vlm>
- Google — Cloud Translation document translation docs:
  <https://docs.cloud.google.com/translate/docs/advanced/translate-documents>;
  proto with `enable_shadow_removal_native_pdf` / `is_translate_native_pdf_only`:
  <https://github.com/googleapis/googleapis/blob/master/google/cloud/translate/v3/translation_service.proto>;
  output teardown (raster underlay + text overlay):
  <https://github.com/pymupdf/PyMuPDF/discussions/2317>
- Microsoft — Azure Document Translation overview and FAQ:
  <https://learn.microsoft.com/en-us/azure/ai-services/translator/document-translation/overview>;
  scanned-PDF handling:
  <https://www.microsoft.com/en-us/translator/blog/2022/05/25/translate-scanned-pdf-documents-with-document-translation/>
- PDFMathTranslate (pdf2zh): <https://github.com/PDFMathTranslate/PDFMathTranslate-next>,
  paper <https://arxiv.org/abs/2507.03009>
- BabelDOC (successor engine; BIoU benchmark, 200-page eval):
  <https://github.com/funstory-ai/BabelDOC>, paper <https://arxiv.org/abs/2605.10845>
- argos-translate-files PDF handler (per-span redact + `insert_htmlbox`):
  <https://github.com/LibreTranslate/argos-translate-files>
- PyMuPDF text/redaction/insertion recipes:
  <https://pymupdf.readthedocs.io/en/latest/recipes-text.html>
