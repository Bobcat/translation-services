# PDF benchmark & regression design

How we measure translated-PDF quality (benchmark) and detect behavioural change
(regression) for the `translate_pdf` pipeline — and how the two combine into one
deterministic fix-loop and one workbench view.

Companion documents: `pdf-translation-design.md` (the pipeline this measures),
`regression-test-design.md` (the image harness this extends; its empirical
determinism findings carry over).

Status: design, agreed 2026-07-16. Nothing built yet.

---

## The two questions

- **Regression** detects *change*: replay frozen inputs through the
  deterministic chain, diff exactly against an approved snapshot. It says
  nothing about good or bad — only "behaviour differs".
- **Benchmark** measures *quality*: score an output PDF against its source on
  defined axes. It says nothing about the cause — only "this is the position".

Together they form the re-baseline decision that is currently taken by eye:
the replay diff shows *what* changed, the benchmark shows whether it got
*better*. They stay architecturally separate (different inputs, different
storage, different failure semantics) and meet only in the view.

---

## Benchmark

### Principle: a function over `(source.pdf, translated.pdf, target_lang)`

No access to pipeline internals, no assumptions about how the translation was
produced. A translation from an external system is just another pair with a
different system label; our own runs produce a `rendered.pdf` like any other
system. This is what makes calibration against other systems' output possible, and it
prevents us from scoring ourselves on knowledge only we have (units, cells,
grouping).

### Axes, not one number

A composite hides exactly the trade-offs we need to see: the published
benchmarks show a system can lead on layout overlap while leaving *more*
untranslated blocks than a competitor that re-typesets everything. Axes:

| Axis | Measures | How |
|---|---|---|
| **Layout** | does the structure survive? | PP-DocLayout on both renders; class-aware region matching (IoU); mean matched overlap, penalties for lost/invented regions |
| **Retention** | did the text survive? | re-OCR of the translated render: 100 − share of eligible source segments whose content is gone. Losing text is wrong under every interpretation |
| **Typography** | legible and proportional? | text overflowing its matched region; font-size *ratio* drift between levels (readers notice broken ratios, not absolute sizes) |
| **Structure flags** | hard yes/no | page count, image-region count, table-region count equal. Flags, not scores — a dropped page is not "−12 points", it is broken |

Next to the axes, one **indicator**: every eligible source segment lands in
exactly one of three observable states — **changed** (present, different from
the source; ~translated), **unchanged** (present, verbatim the source) or
**missing** (gone) — and the unchanged share is reported without a good/bad
scale attached. Deliberately kept text (a proper noun, "Yellow Card") and a
missed translation are indistinguishable from the pair alone, so intent is not
scored. The comparison resolves this for free: the legitimate-keep set is a
property of the *document* (every correct system keeps the same names), so the
cross-system unchanged-delta on the same document is the actionable signal.
Axis and indicator names are observations on purpose; "preserve" and
"leftover" stay pipeline jargon (deliberately-kept text; a unit without a hint
line) and never appear in benchmark vocabulary.

Translation *adequacy* (is the translation itself good — including whether an
unchanged segment *should* have been translated?) is deliberately not an axis
in v1: that is LLM-judge or human-judge territory, non-deterministic, and even
then soft. Later, as a separate, clearly-labelled advisory axis.

### Measurement pipeline (per document pair)

1. Render both PDFs at the same dpi (page by page; page-count mismatch → hard
   flag, score the aligned prefix).
2. PP-DocLayout on both sides → regions per page.
3. Region matching: class-aware, greedy on IoU, with two granularity filters
   (scoring v3, each motivated by a measured artifact): near-identical
   same-family detections on one side are deduped, and an unmatched region
   whose area is largely covered by same-family regions on the other side is
   "covered" — a detector split/merge/nested detection, excluded from the
   layout score entirely. Only truly lost source regions and truly invented
   target regions penalize, weighted by sqrt(area); the document score
   aggregates by region weight rather than per page. A detector miss on one
   side (content visibly present, no region reported) still counts — that is
   measurement noise the weighting can only dampen. Background is excluded by
   construction.
4. OCR both renders → segments per page (the reader's view, identical
   treatment for every system).
5. Text-fate split (changed | unchanged | missing), language-free: a target
   segment counts as unchanged when it matches a source segment verbatim
   (normalized) AND contains alphabetic words; segments in the non-prose
   classes (prices, codes, URLs) are not eligible to begin with. Language-ID
   at most as a secondary signal on segments of ≥3 words; LID on short
   segments is known to be unreliable.
6. Geometry/typography checks on matched regions.

### Determinism — where the variance actually is

The measurement chain is deterministic on identical input files: OCR is
bit-stable on identical pixels (measured; see `regression-test-design.md`),
rendering a PDF at a fixed dpi is deterministic, and scoring is pure code. So
**a score is a property of the pdf pair**: same pair, same score, cacheable;
any score change after a code change is real.

What varies is *our own system's output*: VLM grouping and translation wobble
run to run, and inpaint re-rolls its model fill (GPU float). Two live runs on
the same source therefore give two different output files, each with its own
exact score. Consequence for comparisons: our own column in the comparison
matrix shows a **spread over N pipeline runs** (min/median/max per axis); an
uploaded external translation is one static file with one fixed score.

Verified before building (measured 2026-07-16, probe scripts, throwaway):

- **PP-DocLayout is bit-stable on identical pixels.** Two in-process runs and
  two separate processes produced identical region lists (hash-equal) on 6
  diverse testset pages (3–91 regions each: simple text, infographic spread,
  scientific paper, scan, hybrid, mixed). The layout axis rests on solid
  ground.
- **Inpaint is deterministic in practice on this environment.** Two rerenders
  of the same cached inputs produced byte-identical PNGs on a photo-heavy scan
  page where the model fill demonstrably ran (a flat-fill rerender of the same
  inputs differs). Stronger than the regression doc's earlier assumption of
  GPU float wobble: on the current box/torch version, replay output —
  including inpaint — is byte-stable. Consequence: **"score unchanged" is
  exact**; any observed score drift on a passing replay is a real signal to
  investigate, never a band to absorb. If wobble reappears after a torch or
  hardware change, re-measure before adding tolerance.

### Scale and calibration

- **The numbers are preservation measurements, not a quality ranking.** All
  three axes measure what survived the translation; a system that returns the
  source unchanged therefore maxes every axis, and only the unchanged
  indicator exposes it. Reading order: structure flags first (is the document
  intact?), then the unchanged share (did the system actually translate?),
  then the axes — and "higher is better" only holds between systems whose
  unchanged shares are comparable. This also bounds the headroom readout: a
  gap against a system that barely translated is not attainable headroom.
- **Anchor the corners with constructed baselines.** The *identity baseline*
  (source submitted as its own translation) must score ~100 on layout, 100 on
  retention and 100% unchanged — everything kept, nothing translated. If
  identity does not reach ~100 on layout, the number measures detector noise,
  not quality — fix that before trusting anything else.
- **The comparison is a time-allocation instrument, not a contest.** Per-axis
  0–100 is operationally defined (e.g. layout = mean class-aware IoU × 100),
  but a 75 only gets meaning from the columns next to it — and the question
  those columns answer is where effort pays, not who ranks first. If nothing
  observed exceeds ~75 on a document class, that is probably what the class
  currently allows: move to the next bottleneck rather than polish the metric.
  If some system reaches 92 on an axis where we sit at 75, that gap is
  *proven-attainable headroom* — that is where the time goes. Reporting sorts
  by headroom (gap to the best observed result, our own runs included), not by
  rank.
- **No composite score in v1.** Revisit once real distributions over the
  testset exist; a weighted composite chosen before seeing data would encode
  guesses as policy.
- **What the benchmark deliberately does not measure:** visual quality. Re-OCR
  is blind to cosmetics — smears, casts, ugly fills pass as long as the text
  reads. A high score is not a visual verdict; the human spot-check stays.
  This warning belongs in the view UI itself.

### Frozen measurements: `scores = f_scoring(measurement)`

A benchmark run splits into (a) expensive, environment-bound inference —
render, PP-DocLayout, OCR — and (b) pure scoring code over its output. Layer
(a) is deterministic on identical pixels *today* but not across time (model
upgrade, other machine, other dpi); layer (b) is the code that will evolve.

Therefore every run persists layer (a)'s output as `measurement.json`
(regions + OCR segments per page, tens of KB). Scoring becomes a pure function
over it: any scoring change can be recomputed **retroactively over the whole
history, including external uploads**, keeping the comparison matrix internally
consistent across scoring versions. The "scores only comparable within a
version" constraint then applies only to the measurement layer. Side benefit:
scoring unit tests get real frozen measurement data as fixtures, and a
surprising score can be recomputed offline without a GPU.

Two rules make the schema-evolution risk manageable:

1. `measurement.json` carries a **schema version** plus the PP-DocLayout/OCR
   model versions and the render dpi. Additive fields break nothing; only a
   breaking change invalidates old measurements for new scoring.
2. **The pdf pairs are the ground truth and are never deleted.** A breaking
   schema change is then a re-measure pass (GPU time, no information loss),
   not data loss.

Page renders are *not* stored (the only heavy part; deterministically
reproducible from the PDFs at the recorded dpi) — at most cached for the
view's overlays.

### Storage, API, CLI

Benchmark data is a persistent dataset, outside the TTL'd `work_root`:

```
data/benchmark/<doc-id>/<system-label>/<run>/
  source.pdf  translated.pdf     # ground truth (re-measurable)
  measurement.json               # regions + OCR segments per page; schema/model versions, dpi
  scores.json                    # scoring version + per-axis results (derivable from measurement)
```

System labels are user input (data — external product names are fine there;
they do not enter code or docs). API sketch:

- `POST /v1/benchmark/run` — either `{request_id}` of a completed
  `translate_pdf` run, or an uploaded pair + system label.
- `GET /v1/benchmark/results` — the comparison matrix.

CLI (the headless twin, like `scripts/regress.py`): run over the whole
testset; `re-score-all` (cheap, pure CPU) and `re-measure-all` (expensive,
only on measurement-layer changes) as separate commands.

---

## Regression for translate_pdf

Per page, the deterministic chain of `translate_pdf` **is** the image chain:
align + render on frozen `(cells, raw hint, translations)`. A document fixture
is therefore essentially a list of page fixtures plus document-level checks:

- census (expected page classes, sizes, rotation),
- page count and page dimensions of the assembled output,
- rasterize/assemble are deterministic given dpi + engine version.

The existing fixture machinery is reused per page, not duplicated. Capture
needs no new pipeline work: a completed `translate_pdf` run already persists
per page exactly what a fixture freezes (`pages/page-NNN/grouping.json`,
`translation.json`, the raw hint inside `llm_calls.json`).

Comparison semantics per page are unchanged from `regression-test-design.md`
(exact align diff, re-OCR render diff with its measured tolerances).

---

## Benchmark-on-replay: the deterministic combination

The benchmark scores a file; a fixture **replay** produces that file
deterministically. So benchmark-on-replay is deterministic end to end, and the
two tools compose:

- **Documents you are not touching:** replay passes ⇒ same output ⇒ same score
  *by construction* (subject to the inpaint-wobble probe above). "Replay green
  and benchmark unchanged" is one check with the score as a free readout.
- **The document you are fixing:** the replay fails by design (behaviour
  changed — that was the point); benchmark-on-replay gives the quality delta
  of exactly that change, with zero VLM/translator noise in the comparison.
  The benchmark becomes the yardstick of the re-baseline decision: align diff
  matches the intent + score moved 71 → 78 ⇒ accept.

On accepting a new snapshot, its benchmark score is frozen alongside as the
**accepted score** — the baseline the Score tab diffs against.

Live-benchmark on fresh runs (with system spread) and external uploads is the
third, separate use: calibration, not the fix-loop.

---

## Workbench: one view, three tabs on the same document list

| Tab | Question | Determinism |
|---|---|---|
| **Replay** | did behaviour change? | exact (the existing regression-view pattern, document → pages) |
| **Score** | benchmark-on-replay + delta vs accepted score | deterministic |
| **Comparison** | how does our latest run move against our own best (Δ ours), with external measurements as informative reference — not a ranking | live runs (spread over N) + static external uploads |

The working pattern "fixing one document while the rest must stay green and
equal" is: focus one document in Replay+Score, "Run all" for the rest. Cell
detail (any tab): side-by-side page renders with overlays — matched regions
green, lost regions red, unchanged segments marked, overflow flagged.

Import flow (Comparison): pick a source document from the PDF testset, upload
the external translation, type a system label.

---

## Phasing

- **Slice 2a — probes + engine.** The two determinism probes (PP-DocLayout
  bit-stability; inpaint wobble vs score). Then the measurement layer +
  scoring layer + storage + CLI, run over testset v2 with the identity
  baseline. Output: real score distributions to calibrate the scale on.
- **Slice 2b — document fixtures + replay.** Document-fixture capture from a
  completed run, replay (per-page reuse + document checks), benchmark-on-replay,
  accepted-score freeze.
- **Slice 2c — API + view.** `/v1/benchmark/*`, the three-tab view, external
  import, the comparison matrix with our N-run spread, sorted by headroom.

Each slice is independently useful: 2a alone already scores any pair from the
CLI; 2b alone already guards the pipeline; 2c makes both routine.

---

## Out of scope (v1)

- Translation-adequacy scoring (LLM judge) — later, as an advisory axis.
- Visual/cosmetic quality — named blind spot of re-OCR-based measurement.
- A composite score — only after distributions over the testset are known.
- Cross-machine score comparability — scores are compared within one
  measurement environment (same models, same fonts, same dpi), like the
  regression harness.
