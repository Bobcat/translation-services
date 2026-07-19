# How we benchmark translated PDFs

What the numbers in the PDF-testing comparison mean, how they are produced, and
— just as important — which earlier metrics we abandoned and what measurement
taught us each time. The design history and API/storage details live in
`pdf-benchmark-regression-design.md`; this document is the method as it stands.

Status: describes scoring rev 5 (2026-07-18). The rev number is a change
counter of the scoring function — it identifies which code produced a stored
`scores.json` — not a maturity level.

---

## The principle

A benchmark run is a pure function over `(source.pdf, translated.pdf)`. No
access to pipeline internals, no assumptions about how the translation was
made: our own output and an externally produced translation are scored through
exactly the same lens. Both PDFs are rendered and read back with OCR — the
reader's view — and everything else derives from that.

Two layers, deliberately split:

- **Measurement** (expensive, environment-bound): render both documents at the
  analysis dpi, run the layout detector and OCR on every page, freeze the
  result as `measurement.json`. The pdf pair is ground truth and is never
  deleted.
- **Scoring** (pure code): every number derives from the frozen measurement,
  so a scoring change re-scores all stored history retroactively
  (`scripts/benchmark.py rescore`) — no GPU, no information loss. This is what
  allowed the metric to evolve as fast as it did.

Everything is calibrated against the **identity baseline**: the source
submitted as its own translation must score layout 100, anchors 100, volume
ratio 1.00 and unchanged 100%. If identity drifts, the measuring stick is
broken — fix that before reading anything else.

## The measuring stick today

Two tiers, because the signals differ in evidentiary strength.

**Text signals — detector-free (the first line of a cell):**

- **A — anchors.** The share of the source's *numbers* still present in the
  translation. Numbers are translation-invariant: whatever the style or
  language, "513,750" must come out the other side. Signatures are normalized
  so that only real loss counts (see below). This is the primary text-survival
  axis.
- **U — unchanged** (indicator). Share of source text left verbatim. A
  deliberate keep and a missed translation look identical in one pair, so U is
  read *across systems on the same document*: legitimate keeps are a property
  of the document and cancel out.
- **× — volume** (indicator). Translated/source text volume in script-aware
  units (a CJK character counts as one unit, other scripts count words). The
  language-pair component is constant per document row; translator style adds
  a few percent of spread — so only a clearly low outlier suggests dropped
  content. This is the coarse backstop for prose that contains no numbers.

**Layout-detector signals (the dimmed second line):**

- **L — layout.** The layout detector's regions on both renders, matched
  one-to-one by overlap (IoU), sqrt-area-weighted, with granularity artifacts
  (splits/merges/nested/duplicate detections) classified as "covered" and
  excluded rather than penalized.
- **T — typography.** OCR ink outside every detected region (stray text) and
  font-size ratios drifting between matched text regions.
- **⚑ flags.** Hard yes/no mismatches: page count (exact), image/table region
  count (detector-based, at the detector's confident threshold).
- **Noise dimming.** The share of detected region weight that found no
  counterpart is carried per run (`layout_noise_share`); above 0.2 the view
  dims L/T and marks them ≈ — matching noise, not quality, may dominate there.

**Evidence, not trust.** Clicking a cell shows the proof behind A: exactly
which numbers are missing, each with its page and surrounding text, above the
per-page region overlays. Every number in the matrix should be verifiable in
two clicks; this click-through is also how most of the normalization rules
below were found.

The comparison shows the **latest run** per system. It is informative, not a
ranking: fundamentally different approaches preserve different things by
design, and visual quality and translation adequacy are not measured at all.

## Anchor normalization

A source number and its translated rendering may legitimately differ in
surface form. The signature pipeline folds exactly the differences measurement
showed to be benign, and nothing else:

1. **NFKC** — full-width digits (１２３) become ASCII.
2. **OCR confusion folding** — o/O→0 and l/I→1, but only where the letter
   touches a digit (looking through grouping separators). Cause: a geometric
   sans with a circular zero ("2o25", "235,ooo", "$5l3,750" — all misreads of
   correct source text). The adjacency requirement keeps unit words intact:
   "$927million" must not yield a phantom "11" from its ells.
3. **Separator stripping** — "1,234.56" ≡ "1.234,56"; spaced grouping
   ("1 234 567", "235. 000") only when it *looks* like grouping: exactly three
   digits follow and at most three precede. "#1 2025" (a rank and a year),
   "18 07" (a date) and "In 2025, 119 students" (a prose comma) stay separate
   numbers.
4. **Numeric normalization** — leading zeros are stripped ("07 July" equals a
   localized "7 juli") and only numbers of ≥2 significant digits count; single
   digits are list-marker noise.
5. **Document-wide multiset matching** — reflow across page boundaries is not
   loss; a re-typesetting system may move content freely.
6. **Glue/split resolution** — OCR sometimes reads two adjacent numbers as one
   ("50-59" with a lost dash → "5059") or one number as two (a wrap splitting
   "513.750"). A leftover mismatch is forgiven only when one side's signature
   equals the concatenation of two signatures adjacent in reading order on the
   other side.

Named limits, accepted for now: numeral rewriting changes the signature and
reads as loss ("238k" written out as "238.000"; a CJK myriad form like
"5万" ⇄ "50,000"); prose without any digit is invisible to anchors (the volume
ratio is the backstop); and anchors ride on OCR quality of both renders, so a
glyph the OCR cannot read on either side still counts against the pair.

## What we abandoned, and why

Every retirement below was forced by a measured case, not by taste. The frozen
measurements made each one cheap to verify and retroactive to apply.

**"Completeness" → changed/unchanged/missing (rev 2).** The original completeness
axis mixed two things a pair cannot distinguish: a deliberately kept proper
noun and a missed translation. The split into observable states — changed,
unchanged, missing — moved the judgement out of the metric; the unchanged
share became an indicator to compare across systems instead of a score.

**Naive region matching → granularity-aware matching (rev 3).** The detector
reports splits, merges, nested and duplicate regions that are not layout
changes. Case: region pairs with visibly identical content scored as
lost+invented because one side's detector cut the block in two. Rev 3 dedupes
duplicates, classifies split/merge/nested coverage as "covered" (excluded),
and weights by sqrt(area) so captions cannot outvote page structure.

**Detector threshold 0.5 → 0.4, measurement-only (rev 4).** A page header —
the most prominent element on the page — vanished from an externally
re-rendered copy that is visually identical to the source: the detector kept
localising it correctly but split its confidence across competing classes
(doc_title 0.42 / paragraph_title 0.38 / header 0.35 / text 0.35), every
candidate under the model's 0.5 default. Dpi was ruled out (same at 300).
Measurement now detects at 0.4. Crucially this is *measurement-only*: the
threshold parameter feeds the model's internal postprocess, and lowering it
removes full-page image regions the translation pipeline's preserve/gate logic
depends on (measured on 14 of 265 testset pages) — so the pipeline keeps the
model default, and the two configurations are one explicit parameter apart.

**Retention axis → anchors + volume (rev 5).** Retention (100 − missing share)
was presented as text survival but rode entirely on the same region matching
as L: a source segment counted "missing" when its *region* found no
counterpart. Case: a re-typesetting system scored retention 84 while its
translated pages carried equal-or-more text than the source (per-page volume
ratios 1.05–2.08) and the worst-scoring page was visually complete — all 40
"missing" segments arrived via lost regions, zero via genuinely empty ones.
Region-gated retention punished reflow, not loss. It survives only as the
`region_retention` indicator; anchors and the volume ratio measure text
survival from text evidence.

**Median over runs → latest run.** Our stored runs accumulate *across code
versions*, so a median mixed old pipeline behaviour into the current stand —
apples and oranges whenever the deterministic part changed. The cell now
shows the latest run; run-to-run variance (VLM/translator wobble) is still
visible by scoring runs individually, and the Δ-ours column tracks the latest
run against our own best earlier one.

**Trust → evidence.** The single biggest usability change was not a metric
but the click-through: every anchor "loss" is listed with its page and
surrounding text. In its first hours it exposed, in order: the circular-zero
font confusion, a separator that glued a rank to a year, leading-zero dates
counted as loss, line-wrap glue/splits — and one genuine translation error in
our own output (a year rendered as a different year). The normalization rules
above are that list, fixed; the last item is the kind of finding the axis
exists for.

## Reading a row in practice

1. Flags first: a page-count mismatch means broken, not "minus points".
2. Then A with its evidence list — real loss shows here, with proof.
3. U and × against the *other systems on the same row*, not in isolation.
4. L/T last, and dimmed-≈ cells with extra suspicion: that is detector noise
   territory.
5. Remember the named blind spots: visual quality and translation adequacy are
   not in these numbers at all.

## Pointers

- Design, storage, API, regression coupling and the detector appendix:
  `pdf-benchmark-regression-design.md`.
- Implementation: `app/benchmark/measurement.py` (layer a),
  `app/benchmark/scoring/` (layer b, pure).
- CLI: `scripts/benchmark.py` — `measure`, `identity`, `rescore` (cheap,
  retroactive), `remeasure` (GPU, after measurement-layer changes), `report`.
