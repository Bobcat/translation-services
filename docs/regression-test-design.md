# Regression test design

How we turn the manual, one-by-one eyeballing of pipeline output into an automatic
regression test — without being defeated by the pipeline's non-determinism.

Last updated: 2026-06-22.

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
  string, and the per-unit translations. The replay-input.
- **snapshot** — the approved *expected output*: the align result (units + ignored cells) and
  the re-OCR of the rendered result.
- **approve** — record (or re-record) a snapshot + fixture.
- **regression run** — replay every fixture and diff against its snapshot.

---

## Freeze boundary

The renderer is a pure function of `(units, source image)`; grouping/align is a pure function
of `(cells, parsed hint)`. So we freeze the three non-deterministic stage outputs and re-run
everything deterministic after them:

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

## Identity

Fixtures are keyed by the **canonical-ingest SHA-256** of the source image (the bytes
`app.main` stores after canonicalisation), not the filename — robust to renames and matches a
fresh submit to an existing testset image.

---

## Keying translations to re-grouped units

At replay, align re-produces units and we must attach each frozen translation to the right
unit. A unit *is* its set of member cells (cells are partitioned across units), and each
member carries a reading `order`.

- **Key = the anchor cell** (the member with the lowest `order`). Positionally the first cell
  is the unit's anchor — the text places from there — and because cells are partitioned, that
  one cell already identifies the unit uniquely. It is the most stable key: a trailing wrap
  cell joining or leaving does not change it.
- Order matters semantically (member order builds `source_text`, hence the translation), so
  the **snapshot comparison** uses the full **ordered** cell list, not a set.

If align re-groups so the anchor moves, the key misses and that unit gets no translation —
but the align diff already fails on the composition change, so the render diff is secondary.

### Worked example

OCR yields cells 0–8. The frozen hint has 2 lines. Align produces:

| status | cells (reading order) | translation |
|---|---|---|
| unit A | 0, 1 | hint 0 → "Hallo wereld" |
| unit B | 2, 3, 4 | hint 1 → "Totaalbedrag" |
| unit C | 8 | leftover (hint_index None) → "Prijs" |
| ignored | 5, 6, 7 | none — OCR noise / icon / unmatched fragment; original pixels kept |

```json
"translations": {
  "0": { "translated_text": "Hallo wereld", "field_translations": null },
  "2": { "translated_text": "Totaalbedrag", "field_translations": null },
  "8": { "translated_text": "Prijs",        "field_translations": null }
}
```
Keyed by the anchor (lowest-order) cell of each unit. A table-row unit carries non-null
`field_translations` (the per-column `[source, translated]` pairs) that `_split_table_row`
needs. Ignored cells have no entry — no text to place. "Which cells are ignored" is itself an
align decision, so the snapshot records it: a cell flipping unit↔ignored is a regression.

---

## Schema

**fixture.json** (the replay inputs)
```json
{
  "image_sha256": "…",
  "cells": [ … ],
  "raw_hint": "…",
  "translations": {
    "<anchor_cell_id>": { "translated_text": "…", "field_translations": [["src", "tr"], …] }, …
  },
  "request_flags": {
    "preserve_heuristic_text": true, "preserve_unchanged_text": false, "use_geometry_columns": true
  },
  "grouping_model": "…"
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

**snapshot.json** (the approved expected output)
```json
{
  "expected_units": [
    { "cells": [0,1], "hint_index": 0, "level": "header",
      "alignment": "center", "bullet": false, "bullet_marker": null, "block_id": 0 },
    …
  ],
  "ignored_cells": [5,6,7],
  "reocr": [ { "text": "…", "left": 0, "top": 0, "width": 0, "height": 0 }, … ]
}
```

---

## Comparison and tolerances

- **Align diff** — `expected_units` (ordered cells + label fields) and `ignored_cells`,
  compared exactly and order-sensitively. Any composition / order / label / keep-drop change
  fails, localised to the unit.
- **Render diff** — re-OCR the replayed render at **cell level** (`merge_lines=False`), match
  regions, then require **token-recall = 1.0** and **centroid shift ≤ 3px**. Same machine and
  fonts as capture → effectively exact; the 3px absorbs sub-pixel AA. Loosen only if a
  cross-environment run is ever needed.

Cell-level re-OCR conflates a genuine placement move with OCR re-segmentation caused by a
slightly different text width; acceptable for now (the align diff carries the precise signal).

---

## Storage

```
testset/_regression/<image>/<variant>/
  fixture.json
  snapshot.json
```
**gitignored** — snapshots contain re-OCR text of testset images, which carry real PII.

---

## Workbench integration

Approval happens in the workbench so you **freeze exactly the result you just visually
approved** — not a fresh, possibly-different run. Everything a fixture needs is already on the
completed request, so capture re-runs nothing:

| fixture part | source on the completed request |
|---|---|
| cells | `response.ocr.cells` |
| raw hint | `response.ocr.raw` (`hint_raw`) / `llm_calls` |
| translations | `response.ocr.translation_units` → per-unit text, keyed by anchor cell |
| rendered result (for the snapshot re-OCR) | artifact `rendered` |
| identity | canonical-ingest SHA-256 of the input |

### Backend endpoints (translation-services)

1. `GET /v1/regression/status?name=<name>` — `{ name, in_testset, fixture_count, variants:[…] }`,
   drives the UI badges. A dedicated endpoint rather than a field on every lifecycle response, so
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

---

## Phased plan

**Step 1 — replay + regression run (backend only).** Prove a fixture replays deterministically
and a diff catches a real change.

- New subpackage `app/regression/`: `fixture.py` (model + load/save), `replay.py`
  (`replay_fixture(settings, input_path, fixture) → (units, rendered_bytes)`, reusing the
  existing stage functions and re-applying the `preserve_heuristic_text` filter so the unit set
  matches), `snapshot.py` (capture + re-OCR), `compare.py` (align + render diff).
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
