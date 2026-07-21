# Layout detector — where it could replace or reinforce our own heuristics

The page layout detector (`app/layout/`) runs on **every** image request — an image job, and
each page of a document job — and its regions are cached in the run's `grouping.json`. It
arrived late: the pipeline had already been built on OCR-cell geometry, and the detector was
adopted for the PDF work (align's column evidence, figure/chart preserve). So a good part of
the codebase reconstructs from cells what the detector states directly, and that overlap has
never been inventoried.

This document is that inventory, plus the first measurement. It is an audit, not a plan: every
row below is a candidate, and each one needs its own measurement before any code moves.

## Two properties that make it worth using more

**It is already paid for.** Detection is ~30–80 ms warm and runs behind the multi-second VLM
call; the regions ride along in `grouping.json`, so a re-entry (re-render, re-translate) has
them too. Consuming them more costs nothing extra.

**It is deterministic.** Same image, same regions — unlike the grouping VLM, whose run variance
is measured and has repeatedly forced us to back deterministic code around it.

## What we reconstruct from cells that the detector states directly

| What we work out | How we do it now | The detector's own answer | Assessment |
|---|---|---|---|
| Columns and margins | x-axis projection of every box; an empty corridor with occupied ground on both sides is a gutter (`replacement/layout/bands.py`) | one `text` region per column | **done** — rebuilt on regions, projection kept as fallback |
| "These cells are one paragraph/element" | distance heuristic (`units._near`) + hint-claim consolidation in align | each paragraph is its own `text` region with a boundary | **measured below** — 4 defects, 0 false alarms |
| "This is mathematics" | text layer: math font names (CMSY/CMMI/…) + character classes (`pdf/textlayer.py::_island_runs`) | `formula`, `formula_number` regions — **also on scanned pixels** | born-digital: cells win (ground truth). Scans: only the detector can say it |
| "This line carries a formula" | island presence on the line (`planning._body_top` gate) | a `formula` region overlapping the line | would extend the same repair to the OCR path |
| "This is a table" | field pairs from the hint + isolation/achromatic gates on the rule restore (`ground/erase.py`) | a `table` box | detector as a gate; the columns *inside* a row stay ours |
| "This ground is photographic, not flat" | pixel-variance router in erase/inpaint | `image` / `chart` boxes | free second opinion for the parked boundary router |
| "This is a heading" | the VLM's level label (varies run to run) | `paragraph_title` / `doc_title` | deterministic second vote |
| "A cell straddles a figure edge" | nothing — open task | the figure edge *is* a region edge | the detector is the first evidence here |
| Footnote / page number | marker patterns + level labels | `footnote`, `number` | small, free |

### Where cells remain the better evidence — deliberately untouched

Skew and perspective (detector boxes are axis-aligned and say nothing about angle); justified
detection (line-edge geometry, below its granularity); field/column structure *within* a table
row; reading order, token matching and dedup (text, not layout); and on born-digital pages the
text layer itself — a font name is harder evidence than a 0.6 detection score. Roughly two
thirds of the codebase sits here: OCR fixes, the tokenizer, every translation gate, font and
wrap work, colour extraction, the price/URL preserves, and all infrastructure.

## Measurement 1 — unit formation vs. region boundaries

The question: align reconstructs text blocks from loose cells (distance thresholds, hint
claims, dedup) while the detector hands those blocks over directly. **How often does a unit we
built span two regions, and is that a defect or a legitimate merge?**

Method: for every fixture carrying cached regions (image fixtures + document pages), build the
units exactly as the pipeline does, assign each member to the smallest region (score ≥ 0.6)
containing its centre, and report every unit whose members land in ≥ 2 distinct regions.

```
multi-member units with region coverage : 408
units spanning >= 2 regions             :   4  (1.0%)
```

All four are defects. None is a legitimate merge:

| Fixture | What the unit glued together |
|---|---|
| quickguide (multiblock, back) | an paragraph unit pulls in `"will see the frequency of updating in"` from a **different** text block |
| academic article (two-column) | the loose cell `"In"` — the start of the next column's paragraph — hangs off a unit in the figure-caption area |
| annual report (photo credit) | page number `"8"` glued to the footer line |
| annual report (stacked tables) | page number `"9"` glued to the footer line |

Two readings, both useful:

**99% of unit formation already agrees with the regions.** Align does its job; there is no
large body of work here to replace. The earlier assumption that "we probably built a lot that
the detector could do" does not hold for this stage.

**The 1% that disagrees is pure defect.** Low firing rate, and so far 100% precision — exactly
the profile where a guard pays off, and exactly the class of bug found by hand twice this
month (the duplicated line on the two-column article was the same unit).

### Proposed guard (not built)

In the `bands.py` pattern: when a unit's members are assignable to regions and a small minority
sits in a *different* region from the rest, that minority continues as its own leftover unit
instead of riding along. Leftovers are translated and rendered at their own position, so the
page number stays where it is and the stray cell stops contaminating the paragraph.

Failure direction: no regions, or no clear majority → nothing changes. On the design-image
class (~25% region coverage, measured) it is inert by construction.

Named caveats: four cases is a small sample, so "100% precision" is promising, not proof — the
guard must stay conservative (minority only, both sides genuinely assigned). And one scenario
absent from the fixtures but worth covering: a paragraph flowing around a figure can legitimately
occupy two text regions, so the majority region must genuinely *not* contain the odd member.

## Candidate ranking (before measurement)

1. **Islands for scanned formulas** — the flagship feature covers born-digital only; "scanned
   formulas keep today's behaviour" is a named limit in three places. `formula` regions are the
   missing evidence on the OCR path. Biggest feature win, biggest job; inline-formula scores are
   modest (0.54–0.82 measured), so it needs the care the original island phases got.
2. **Table machinery gated on `table`** — the safety win. The isolation and achromatic gates on
   the rule restore exist only because the restore misfired *outside* tables; "inside a detected
   table" would have prevented that class by construction. The parked Table 3 defect touches the
   shared field-matching core, where a table region could be the extra arbiter.
3. **Unit boundaries in align** — measured above.
4. **Ground router** — the parked boundary router's hard requirement was "zero extra VRAM or
   latency elsewhere"; the regions are already computed, so the evidence is free.
5. **Small change** — `footnote`, `number`, and `paragraph_title` as a second vote next to the
   VLM level. Too small to measure; fold in when that code is touched anyway.
