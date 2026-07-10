# Regression test design

How we turn the manual, one-by-one eyeballing of pipeline output into an automatic
regression test — without being defeated by the pipeline's non-determinism.

Last updated: 2026-06-23.

---

## Goal

Replace "submit every testset image and check each result by hand" with an automatic test
that fails when a code change regresses the **alignment** (cell→unit grouping) or the
**render** (text re-placement) — the two deterministic stages where our bugs actually live.

The hard part: the end-to-end pipeline is **non-deterministic**. The grouping VLM and the
translator both vary run to run, so a naive "render again and compare" flags model drift as
if it were a regression.

---

## Empirical basis

Measured before designing (probe scripts, throwaway), so the design rests on data not
assumption:

- **OCR is bit-stable.** Re-OCR of identical pixels returns identical text *and* boxes. So
  OCR contributes zero comparison noise; any difference between two renders is a real render
  difference, not OCR jitter.
- **Render is deterministic given its inputs.** Across the full testset (6 runs/image), every
  time the resolved text was identical the centroid drift was ≤4px (i.e. 0 real movement).
  The render adds no variance of its own.
- **The variance lives upstream.** Per image the distinct-output count is the same at exact /
  token / upstream-translation granularity — it is the VLM grouping and the translation that
  wobble, not the render. Distribution over 25 images: **5 fully stable, ~16 with 2–4 variants,
  4 with every run distinct** (busy social-overlay / chatty-menu images).
- **The wobble is often a single used field.** Example: a 6-distinct image differed between two
  runs only in the `alignment` of one element (left vs centered) — a grouping/VLM confidence
  artifact, not a render bug. Unused hint fields (e.g. the label's font-size, which the
  renderer discards in favour of OCR true-height) wobble too but cannot affect output.

Conclusion: **freeze the non-deterministic upstream, re-run the deterministic chain.** Then a
single recorded expected result suffices and any diff is a genuine regression.

---

## Terminology

- **fixture** — the frozen *inputs* for one replay of one image: OCR cells, the raw VLM hint
  string, and the translations (per hint line, plus per-cell for leftovers). The replay-input.
- **snapshot** — the approved *expected output*: the align result (units + ignored cells) and
  the re-OCR of the rendered result.
- **approve** — record (or re-record) a snapshot + fixture.
- **regression run** — replay every fixture and diff against its snapshot.

---

## Freeze boundary

The renderer is a pure function of `(units, source image)` — with one exception:
`erase_fill_mode="inpaint"` re-rolls the model fill every render (GPU float
nondeterminism). Measured (2026-07-09, 9 replays vs snapshot on the two riskiest
fixtures): the re-OCR compare's tolerance absorbs that wobble completely, so
inpaint-pinned fixtures are allowed and double as a Tier-2 guard — a re-OCR fail on one
means the fill went genuinely wrong, not that the pin was invalid. Investigate a rare
flip before re-baselining it. Grouping/align is a pure function of `(cells, parsed
hint)`. So we freeze the three non-deterministic stage outputs and re-run everything
deterministic after them:

| live stage | in a regression run |
|---|---|
| `request_grouping_hint` (VLM) | **frozen** raw hint → `parse_grouping_output` re-runs (covers hint_parser) |
| `run_raw_ocr` → cells | **frozen** cells (skips OCR/GPU; OCR is deterministic anyway) |
| `group_cells_into_units` (align) | **re-runs** ← tests `align.py` |
| `translate_units` (gemma) | **frozen** → attached to re-grouped units |
| `render_translated_image` | **re-runs** ← tests `render.py` |

Freezing the *raw* hint string (not the parsed struct) keeps `hint_parser.py` in the test.

### Per-file coverage

| file | tested by |
|---|---|
| `ocr/*` | deterministic; optional cells snapshot (out of scope for now) |
| `hint_parser.py` | unit tests + the regression replay |
| `align.py` | direct unit tests on unit-structure **and** the regression replay |
| `vlm.py` (the prompt) | the stability sweep (separate, threshold-based — non-deterministic) |
| `render.py` | the regression replay (re-OCR diff) |
| `translate.py` | **not covered** by a snapshot (frozen). Out of scope / fuzzy eval |

Freezing the translation buys a deterministic align+render test at the cost of not
regression-testing translation quality — an accepted trade.

---

## Why the translation may be frozen as one variant

Translations vary slightly run to run (temperature-0, but still a few valid variants). We
freeze **one** correct variant per fixture: the regression run never re-translates, so that
variance simply does not occur in the test. We only need one fixed, correct translation as a
stable input to drive align+render.

A different valid translation can be **rendered** differently (length → wrap/fit), and a
different VLM hint can be **aligned** differently. To cover that real input spread we allow
**multiple fixtures per image**, each freezing a distinct valid `(hint, translation)` pair.
Each fixture is individually deterministic with its own single snapshot — this is **not** the
flaky "accept-set that grows to absorb non-determinism" we rejected; they are separate,
deliberately curated test cases. Stable images keep one fixture; wobbly images get a few.

---

## Identity — self-contained source image

A fixture **carries its own source image** (`source.<ext>`): the exact canonical-ingest bytes the
snapshot was rendered on (the upload, fetched from the request's `input` artifact at capture).
Replay renders on that, **not** on `testset/<name>`. This is deliberate: the rendered snapshot
depends on the exact pixels uploaded, and a `testset/<name>` file can differ from what the user
actually uploaded (a re-encoded copy) — anchoring to `testset/` then renders the snapshot on the
upload but the replay on the testset file, so they silently diverge. Carrying the image removes
that whole class of bug. `image_sha256` (of `source.<ext>`) stays as an integrity check.

---

## Keying translations to re-grouped units

At replay, align re-produces units and we must attach each frozen translation to the right unit —
**without keying on anything align produces.** A unit is an align *output* (which cells grouped, and
hence its anchor / member set), so a key derived from it (the old anchor-cell key) breaks the moment
align regroups. We split the translations by what they are genuinely a function of:

- **Hint-matched units → `hint_translations`, keyed by `hint_index`.** The structured translator
  re-translates the *hint lines*; the translation of hint line *i* is a pure function of the frozen
  hint, independent of how align groups cells onto that line. At replay a unit carries the
  `hint_index` of the line it matched, and takes `hint_translations[hint_index]`. So a cell joining,
  leaving, or re-anchoring the unit changes nothing — the line, and its translation, are the same.
- **Leftover units (matched no hint line) → `leftover_translations`, keyed by a member cell id.** A
  leftover is inherently per-cell (a cell OCR'd with no hint line), so its translation is keyed by a
  cell — a frozen input. Replay attaches it by cell **membership** (the unit that contains the key
  cell), which tolerates the anchor moving or sibling cells shifting. Capture writes the anchor cell
  as the key, but any member works because the match is by membership, not by the cell still being
  the anchor.
- Order matters semantically (member order builds `source_text`, hence the translation), so
  the **snapshot comparison** uses the full **ordered** cell list, not a set.

Why `hint_index` cannot key *everything*: leftovers have no hint line (they are common — a busy
social-overlay image can have a dozen), so they would all collide on the empty index. Hence the two
maps. Residual failure modes, both real align changes the align diff already flags: a leftover key
cell becomes `ignored` (that translation unplaced), or two keys land in one unit (a merge).

### Worked example

OCR yields cells 0–8. The frozen hint has 2 lines. Align produces:

| status | cells (reading order) | translation |
|---|---|---|
| unit A | 0, 1 | hint 0 → "Hallo wereld" |
| unit B | 2, 3, 4 | hint 1 → "Totaalbedrag" |
| unit C | 8 | leftover (hint_index None) → "Prijs" |
| ignored | 5, 6, 7 | none — OCR noise / icon / unmatched fragment; original pixels kept |

```json
"hint_translations": {
  "0": { "translated_text": "Hallo wereld", "field_translations": null },
  "1": { "translated_text": "Totaalbedrag", "field_translations": null }
},
"leftover_translations": {
  "8": { "translated_text": "Prijs", "field_translations": null }
}
```
Units A and B are keyed by their `hint_index` (0, 1); the leftover unit C by its member cell (8),
attached at replay by membership. A table-row unit carries non-null `field_translations` (the
per-column `[source, translated]` pairs) that `_split_table_row` needs. Ignored cells have no entry —
no text to place. "Which cells are ignored" is itself an align decision, so the snapshot records it:
a cell flipping unit↔ignored is a regression.

---

## Schema

**fixture.json** (the replay inputs)
```json
{
  "image_sha256": "…",
  "cells": [ … ],
  "raw_hint": "…",
  "hint_translations": {
    "<hint_index>": { "translated_text": "…", "field_translations": [["src", "tr"], …] }, …
  },
  "leftover_translations": {
    "<member_cell_id>": { "translated_text": "…", "field_translations": null }, …
  },
  "request_flags": {
    "preserve_heuristic_text": true, "preserve_unchanged_text": false, "use_geometry_columns": true
  },
  "grouping_model": "…",
  "target_lang": "…"
}
```
Each translation entry stores **both** `translated_text` and `field_translations`: a table-row
unit is only split into columns by `_split_table_row` when `field_translations` is present, so
without it the receipt/menu column path is not reproduced (and untested).

Of `request_flags`, only **`preserve_heuristic_text`** changes the replay — it filters the unit
set fed to render, so the replay re-applies `_units_for_preserve_heuristic_text`. The other two
are recorded for **provenance** only: they shaped the (now frozen) translation, so they cannot
affect a replay that re-runs nothing upstream of render. The fixture stores align *inputs*,
never the units.

`target_lang` is the render's target language: it places the fixture under
`<name>/<target_lang>/` and selects the re-OCR recognizer for the snapshot/replay (mapped to the
PaddleOCR model code, with a Latin fallback for an unsupported code).

**snapshot.json** (the approved expected output)
```json
{
  "expected_units": [
    { "cells": [0,1], "member_translate": [true, true], "hint_index": 0, "level": "header",
      "alignment": "center", "font_family": "Helvetica", "font_weight": 400,
      "bullet": false, "bullet_marker": null, "block_id": 0 },
    …
  ],
  "ignored_cells": [5,6,7],
  "reocr": [ { "text": "…", "left": 0, "top": 0, "width": 0, "height": 0 }, … ]
}
```

---

## Comparison and tolerances

- **Align diff** — `expected_units` and `ignored_cells`, compared exactly and order-sensitively.
  `expected_units` carries the ordered cells, the label fields, AND the render-relevant fields align
  derives — the per-member translate flags and the font family/weight. So a render diff that stems
  from an align change is **localised here** (upstream, to the unit/field) rather than only
  surfacing as a pixel diff; a clean align diff + a render diff therefore points at `render.py`
  itself. Any composition / order / label / font / keep-drop change fails, localised to the unit.
- **Render diff** — re-OCR the replayed render at **cell level** (`merge_lines=False`) and compare
  **whole segments**. OCR is bit-stable on identical pixels — verified segment-for-segment across
  the testset (text *and* boxes match exactly on a no-op replay) — so the read-back is exact when
  nothing changed: every segment must be present with matching normalized text. A change in how the
  render groups or splits a line ("ah pizza" as one segment vs "ah" + "pizza") therefore **is**
  caught, not normalised away. Position is then checked **per segment**: each segment that matches
  by text must have its **centroid within 3px** (same machine and fonts as capture → effectively
  exact; the 3px absorbs sub-pixel AA). Loosen only if a cross-environment run is ever needed.

(An earlier version compared a **word** multiset — each segment split on whitespace — to tolerate
OCR reading a line as one box vs several. That was a workaround for a since-fixed bug where capture
and replay rendered *different* pixels, the source-divergence the self-contained source removed; on
identical pixels OCR is deterministic, so the word split was both unnecessary and able to mask a
real grouping change.)

---

## Storage

```
testset/_regression/<image>/<lang>/<variant>/
  fixture.json     # frozen inputs (cells, raw hint, translations)
  snapshot.json    # expected align output + the rendered image's re-OCR
  snapshot.png     # the approved render (inspection; not used in the diff)
  source.<ext>     # the exact canonical image the render ran on
  actual.png       # only after a failed run — the current render, for snapshot-vs-actual
```
**gitignored** — the source image and re-OCR text carry real PII.

---

## Workbench integration

Approval happens in the workbench so you **freeze exactly the result you just visually
approved** — not a fresh, possibly-different run. Everything a fixture needs is already on the
completed request, so capture re-runs nothing:

| fixture part | source on the completed request |
|---|---|
| cells | `response.ocr.cells` |
| raw hint | the grouping call's `output_text` in `response.llm_calls` |
| translations | `response.ocr.translation_units` → split into `hint_translations` (by `hint_index`) and `leftover_translations` (by member cell) |
| rendered result (for the snapshot re-OCR) | artifact `rendered` |
| identity | canonical-ingest SHA-256 of the input |

**Capturing a re-translate.** A `retranslate_image` response carries only the *new* translations +
render — not the OCR `cells`, raw grouping hint or `grouping_model` the replay needs (those live on
the run that actually did the OCR/grouping). So when the captured request is a re-translate, capture
walks up `source_request_id` to the run that did the grouping and **grafts** those inputs onto the
response before freezing; the translations, target language and request flags stay from the
re-translate. If that source run is no longer in the store, the capture fails with a clear error
rather than freezing an unreplayable empty fixture (one that would replay to the untranslated
source).

### Backend endpoints (translation-services)

1. `GET /v1/regression/status?name=<name>` —
   `{ name, in_testset, fixture_count, langs:{ <lang>: [<variant>, …] } }`, drives the UI badges. A dedicated endpoint rather than a field on every lifecycle response, so
   the hot poll path is untouched.
2. `POST /v1/regression/testset {request_id, name}` — copies the canonical input into
   `testset/<name>.<ext>`.
3. `POST /v1/regression/fixtures {request_id, name, variant?}` — writes `fixture.json` and
   computes `snapshot.json` (re-OCR of the `rendered` artifact + the align structure). The
   re-OCR runs in a worker thread so the event loop is not blocked.

### Workbench UX

| state | badge | action |
|---|---|---|
| image not in testset | "not in testset" | **Add to testset** → `POST /testset` |
| in testset, 0 fixtures | "no fixture" | **Capture fixture** → `POST /fixtures` |
| N fixtures | "N fixture(s)" + variant list | **Capture variant** → `POST /fixtures` (new variant) |

### Admin view (a separate "Regression" workflow)

Browses and manages existing fixtures by **replay** (no pipeline). It is a persistent view, so a
"Run all" survives sidebar navigation. Endpoints:

- `GET /v1/regression/fixtures` — the inventory tree (name → lang → variant + light metadata).
- `GET …/fixtures/{name}/{lang}/{variant}/{snapshot.png,actual.png,source}` — the rendered images
  and the fixture's own captured source image.
- `POST /v1/regression/run {name,lang,variant}` — replay + diff one variant →
  `{passed, diffs, has_actual, timings}`. `timings` is the per-stage replay wall-clock
  (`group_ms` = parse hint + grouping/align, `render_ms`, `reocr_ms`), shown after the variant in
  the admin tree; `has_actual` is true when the run wrote an `actual.png` (i.e. it failed).
- `POST /v1/regression/resnapshot {name,lang,variant}` — **re-baseline**: overwrite the snapshot
  from the current replay (accept a deliberate render/align change whose result is good).
- `DELETE …/fixtures/{name}[/{lang}[/{variant}]]` — cascade delete.

Per-variant actions: **Run replay**, **Accept (re-snapshot)**, **Delete**. "Run all" replays
sequentially, image by image. Capturing a genuinely new variant (a fresh VLM output) stays on the
translation-requests surface — it needs the live pipeline, which the admin view deliberately does
not touch.

---

## Phased plan

**Status: implemented.** All three steps below shipped; this is the original build plan, kept as
the record of how it was rolled out. The shipped `replay_fixture` signature is
`replay_fixture(input_path, fixture) → (actual_units, actual_ignored, rendered_png, timings)`.

**Step 1 — replay + regression run (backend only).** Prove a fixture replays deterministically
and a diff catches a real change.

- New subpackage `app/regression/`: `fixture.py` (model + load/save), `replay.py`
  (`replay_fixture(input_path, fixture)`, reusing the existing stage functions and re-applying the
  `preserve_heuristic_text` filter so the unit set matches), `snapshot.py` (capture + re-OCR),
  `compare.py` (align + render diff).
- `scripts/regress.py` — replay all fixtures, diff, print pass/fail, exit ≠ 0 on failure.
- `scripts/capture_fixture.py` — `--request-id`: the CLI precursor of the capture endpoint.
- Acceptance: capture 3 fixtures (a stable image, a bullet list, **and a receipt/menu** so the
  `_split_table_row` / `field_translations` path is exercised); run `regress.py` twice →
  deterministic pass; make a trivial render change (e.g. erase pad +2px) → it fails on the
  render diff; revert.

**Step 2 — capture endpoints** (`/regression/testset`, `/regression/fixtures`) + the
regression block in the lifecycle response, reusing step 1's capture logic.

**Step 3 — workbench buttons** (frontend) against that contract.

---

## Out of scope

- Regression-testing the **VLM grouping prompt** — covered by the separate stability sweep.
- Regression-testing **translation quality** — frozen, so untested here.
- Pixel/SSIM image diffing — rejected (font/AA differences across environments); re-OCR with
  tolerance is the portable behavioural check.
- An open, growing accept-set per image — rejected; fixtures are deterministic and curated.
