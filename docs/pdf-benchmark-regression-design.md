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
system. This is what makes calibration against other systems possible, and it
prevents us from scoring ourselves on knowledge only we have (units, cells,
grouping).

### Axes, not one number

A composite hides exactly the trade-offs we need to see: the published
benchmarks show a system can lead on layout overlap while leaving *more*
untranslated blocks than a competitor that re-typesets everything. Axes:

| Axis | Measures | How |
|---|---|---|
| **Layout** | does the structure survive? | PP-DocLayout on both renders; class-aware region matching (IoU); mean matched overlap, penalties for lost/invented regions |
| **Completeness** | is everything translated, nothing dropped? | re-OCR of the translated render: leftover segments (still source language) + vanished segments (source region with no text counterpart) |
| **Typography** | legible and proportional? | text overflowing its matched region; font-size *ratio* drift between levels (readers notice broken ratios, not absolute sizes) |
| **Structure flags** | hard yes/no | page count, image-region count, table-region count equal. Flags, not scores — a dropped page is not "−12 points", it is broken |

Translation *adequacy* (is the translation itself good?) is deliberately not an
axis in v1: it needs an LLM judge and is non-deterministic. Later, as a
separate, clearly-labelled advisory axis.

### Measurement pipeline (per document pair)

1. Render both PDFs at the same dpi (page by page; page-count mismatch → hard
   flag, score the aligned prefix).
2. PP-DocLayout on both sides → regions per page.
3. Region matching: class-aware, Hungarian/greedy on IoU. Matched pairs score
   overlap; unmatched source regions count as layout loss, unmatched target
   regions as layout invention. Background is excluded by construction.
4. OCR both renders → segments per page (the reader's view, identical
   treatment for every system).
5. Leftover detection, language-free first: a target segment is a leftover when
   it matches a source segment verbatim (normalized) AND contains alphabetic
   words AND is not in the deliberately-preserved classes (prices, codes,
   URLs — the preserve-heuristic line of thinking). Language-ID only as a
   secondary signal on segments of ≥3 words; LID on short segments is known
   to be unreliable.
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
exact score. Consequence for comparisons: our leaderboard column shows a
**spread over N pipeline runs** (min/median/max per axis); an uploaded external
translation is one static file with one fixed score.

To verify before building (measure, then design — the regression doc's rule):

- **PP-DocLayout bit-stability** on identical pixels (expected, same class of
  inference as OCR; the layout axis rests on it).
- **Inpaint wobble vs score identity**: a replay "pass" is not bit-identical
  (3px centroid tolerance, inpaint float wobble). Whether slightly different
  inpaint pixels can flip a PP-DocLayout region — and thus move a score while
  the replay passes — decides if "score unchanged" may be exact or needs a
  measured band.

### Scale and calibration

- **Anchor the corners with constructed baselines.** The *identity baseline*
  (source submitted as its own translation) must score ~100 on layout and 0 on
  completeness. If identity does not reach ~100 on layout, the number measures
  detector noise, not quality — fix that before trusting anything else.
- **The leaderboard is the product, not the absolute number.** Per-axis 0–100
  is operationally defined (e.g. layout = mean class-aware IoU × 100), but a
  75 only gets meaning from the columns next to it. If no system exceeds 75 on
  the infographic class, 75 is what that class currently allows — chasing 90
  there is chasing the metric.
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
history, including external uploads**, keeping the leaderboard internally
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
- `GET /v1/benchmark/results` — the matrix for the leaderboard.

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
| **Leaderboard** | how do we stand against other systems? | live runs (spread over N) + static external uploads |

The working pattern "fixing one document while the rest must stay green and
equal" is: focus one document in Replay+Score, "Run all" for the rest. Cell
detail (any tab): side-by-side page renders with overlays — matched regions
green, lost regions red, leftovers marked, overflow flagged.

Import flow (Leaderboard): pick a source document from the PDF testset, upload
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
  import, leaderboard with our N-run spread.

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
