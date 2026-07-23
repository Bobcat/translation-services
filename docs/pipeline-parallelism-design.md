# Pipeline parallelism design

How `translate_pdf` can use the model pool's concurrency instead of leaving it
idle — and what to instrument first so the change can be judged.

Companion documents: `pdf-translation-design.md` (the pipeline this changes),
`pdf-benchmark-regression-design.md` (the harness that must keep passing).

Status: design proposal, 2026-07-22/23. Baseline measured, concurrency not built.
One change already made: the units carrying inline-maths tokens batch per page
instead of taking one call each, and every llm-pool call is timed.

---

## The problem

The model pool admits 4 inference requests at a time (`target_inflight`). One
user submitting one document uses 1 of those 4. `translate_pdf` walks its pages
in a loop, and each page waits for its own model calls before the next page
starts.

The runtime already runs 2 request slots, so two *requests* overlap. Within a
request nothing does.

## Baseline

`translate_pdf`, one page at a time. Three runs:

- a 15-page paper, born-digital (text layer, OCR skipped)
- a 200 dpi raster of that same paper, so it takes the OCR route on identical
  content — but it is a clean re-render of a digital PDF, not a real scan: no
  sensor noise, no skew, no compression artefacts, so OCR has it easier than it
  would in the field
- an 11-page 1974 conference paper, a genuine scan (image coverage 1.00 on every
  page) with an untrusted OCR text layer, classified `hybrid` and therefore
  routed through OCR

| stage | born-digital | raster of it | 1974 scan |
|---|---|---|---|
| grouping (VLM) | 38.4s | 37.3s | 48.8s |
| translation (LLM) | 60.2s | 46.4s | 78.4s* |
| **out-of-process** | **98.6s (88%)** | **83.8s (83%)** | **127.2s (84%)** |
| OCR | 0s | 5.1s | 3.5s |
| render | 10.6s | 10.2s | 15.6s |
| align | 1.9s | 1.8s | 4.9s |
| layout | 0.5s | 0.5s | 0.4s |
| **in-process** | **13.1s (12%)** | **17.7s (17%)** | **24.4s (16%)** |
| pages | 15 | 15 | 11 |

\* the 1974 run's stage total was 88.6s; 78.4s is the batch calls alone, the rest
per-unit calls for a page whose block structure did not come back intact.

Three things follow.

**The pipeline mostly waits.** 84-89% of the time is spent on calls to another
process. That is the headroom.

**Our own GPU work is small, on a real scan too.** OCR costs 0.34s per page on
the clean raster and 0.32s per page on the 1974 scan. The in-process GPU models
(Paddle det/rec, doclayout, LaMa) are the stages that cannot run on more than one
thread, and the harder input did not change their weight. LaMa sits inside
`render` and is not timed separately, so the GPU share is measured only for OCR;
`render` also contains CPU work.

**Page cost varies a lot.** In the born-digital run the slowest page took 11.1s,
the median 8.2s and the fastest 1.0s. Serial execution pays that spread in full.

The scanned run being faster is not evidence that OCR is cheaper than reading the
text layer. It does less: OCR flattens formulas into plain characters, the batch
carries them, and the TeX gate strips them. The same paragraph translated from
the text layer keeps `⟦M1⟧ ⟦M2⟧ ⟦M3⟧`, so the formula pixels are transplanted
back; from OCR it came out with the maths simply gone. The two columns are not
the same job.

### Measurement hygiene

Model-call time is stable. Rare single calls are not, and one of them dominates
whatever stage it lands in.

The pool answers an identical payload within 4%: 15 replays of one call ranged
5.10s to 5.30s, with no warm-up curve — the first three averaged 5.17s and the
last five 5.18s. Replaying a whole run's payloads reproduces them: 10 of 11 came
back within 4% of what they took originally.

The eleventh had taken 64.2s and came back in 7.6s. That one call was 48% of its
stage. The same shape appeared on the other document: median call 3.3s in both a
"fast" and a "slow" run, with one 51.3s outlier making the difference between
43.8s and 90.9s.

So read the per-call distribution, not the stage total. A stage total that looks
like a 50% regression is usually one stalled call and ten normal ones.

The cause is not established. Three explanations were measured and ruled out: the
model had been resident for 16 hours across every measurement, so no load or
eviction; the pool answers an identical payload within 4% with no warm-up curve;
and the payload that stalled for 64.2s replayed in 7.6s. Contention on the
multi-tenant GPU would fit the shape, but the operator reports the neighbouring
services were idle at the time.

So: rare, unexplained, and worth watching as more runs accumulate. The per-call
timings now in the log are what will catch the next one.

It does not need proving to design against. Whatever starves a call, the pipeline
has to survive it.

## Translation is not one call per page

Per page the translator makes one batched call, then one call per unit the batch
could not place. Over the born-digital run: 15 grouping calls, 13 batched
translation calls, and **50 per-unit fallback calls**. Page 1 alone made 11.

A unit carrying an island token never took that batch. The hint lines the batch
translates carry the VLM's own TeX reading of the maths, not our `⟦Mn⟧` tokens,
so the token gate would reject every one — measured earlier, they all degraded to
untranslated. Each island unit therefore got its own call: on this paper, 42 of
250 units, so 42 round trips, each resending the whole system prompt.

Island units batch fine on their **own cell text**, where the tokens are present.
That is now what happens: one numbered-list call per page over the island units,
with every per-line gate still judging each unit, and any rejected line dropping
to its own call as before. It also gives an island unit the sibling context an
isolated call cannot have.

| | before | after |
|---|---|---|
| batched calls | 13 | 13 |
| island batch | — | 9 |
| genuine fallbacks | **48** | **7** |
| single-unit (not a fallback) | 2 | 2 |
| total translation calls | 63 | 31 |

The call count halved. The stage time did not move: 62.7s before, 59.7s and 60.2s
after. Those fallbacks were short calls — the saving is round-trip overhead, not
work.

Per-call timing (added with this change) says where the time actually is:

```
13x translation_main         43.8s   ← 73% of the stage
 9x translation_island_batch 14.5s
 6x translation_hint_line     0.9s
 2x translation_single_unit   0.7s
 1x translation_batch_fallback 0.3s
```

**The 13 page-level batch calls are the cost.** The call count that looked
alarming was not. Anything aimed at cutting translation time has to target those.

The change stays, for two reasons that are not wall clock: 31 round trips instead
of 63 frees inflight slots once pages run concurrently, and island units now
translate with their neighbours in view.

Those fallbacks run one after another inside the page. Each pays a full round
trip, and they are independent of each other — one unit's translation does not
depend on another's.

So there are two independent sources of parallelism, not one:

- **across pages** — pages share nothing until the PDF is assembled.
- **within a page** — the fallback calls can be issued together.

The second one matters for single-page image requests too, which page-level
concurrency alone would not help.

## Resource classes

Each stage belongs to one of three classes, and each class needs a different
rule.

| class | stages | rule |
|---|---|---|
| out-of-process | grouping VLM, translation | issue freely; the pool owns capacity |
| in-process GPU | Paddle det/rec, doclayout, LaMa | one lock, strictly serial |
| CPU | align, wrap/fit, compositing | parallel, as far as the GIL allows |

The GPU models are not thread-safe. A single lock around them is the correctness
guarantee, and the baseline says it costs almost nothing: 4s of lock-held time
against ~110s of work. If that changes, a replica pool behind the same lock is a
local change.

The per-page chain and its classes:

```
grouping(pool) → OCR(lock) → align(CPU) → translate(pool) → render: fit(CPU) + inpaint(lock) + composite(CPU)
```

Grouping to OCR is a hard serial edge: the hint chooses the OCR model. Pages
cannot be split into "all model calls first, then all local work" — the chain
crosses between classes twice.

### After the move to dc2

Today the VLM and LLM run in another process with their own VRAM budget, so this
box's GPU carries only OCR and LaMa — measured, 4.1 GiB of the 6.2 GiB resident
is this service. On dc2 they land on one card. The three classes stay, but
"out-of-process" and "in-process GPU" stop being independent resources: page
concurrency and `target_inflight` then draw from one pool of VRAM and SM time,
and cannot be tuned separately.

`target_inflight` stays 4. `max_model_len` and the KV cache are sized for exactly
4 concurrent sequences within a 2 GiB budget; raising it would halve the context
length, which is the wrong trade for a call that carries a full page image.

## What to build

### Phase 1 — spans

The pipeline records durations per stage (`metrics` in `document.json`). For
serial execution a duration is a timeline. For concurrent pages it is not: two
durations do not say whether they overlapped.

Replace them with spans: `(page, stage, kind, t_start, t_end)` on one monotonic
clock, `t0` at request start.

`kind` is `work` or `wait`. Waiting will happen in two places — the GPU lock and
the pool queue — and a span that merges the two reads as "OCR took 4s" when it
was 3.5s of queueing. That distinction cannot be recovered afterwards, so it goes
in from the start.

Changes:

- new `app/core/timing.py` — the recorder and a `span()` context manager.
- `app/tasks/translate_image.py` — the existing `perf_counter` points become
  spans. `metrics` stays, derived from the spans, so `document.json` and the
  benchmark keep their shape.
- `app/tasks/translate_pdf.py` — one recorder per document, page identity on
  every span.
- a `timeline.json` artifact per request.

Out of scope in phase 1: concurrency, the lock, the config parameter.

### Phase 2 — page concurrency

A bounded worker pool over pages. Each page keeps its own serial chain; the mix
of stages in flight falls out of the staggering, so no scheduler decides what
runs when.

- page concurrency becomes a setting, not a constant.
- one lock around the in-process GPU stages.
- PDF assembly keeps page order regardless of completion order.
- per-page progress becomes a count of finished pages.
- `checkpoint()` and cancellation need a defined meaning with N pages in flight.

### Phase 3 — the remaining tail

The island batch already removed most of the per-unit calls. What is left per
page is a handful of hint-line and fallback calls, issued in sequence. Fanning
those out is the only work here that also speeds up a single-page image request,
where there are no other pages to overlap with — but the measured tail is now
under a second per document, so this is last, not first.

### Phase 4 — tune

With the timeline, set page concurrency against measured lock contention and
queue wait. Re-check the pool's draft-token setting on the parallel workload:
speculative decoding pays off less once the batch is full, so a value tuned
against single-stream calls is tuned against a situation that no longer exists.

## Fewer calls pulls against more parallelism

The 13 page-level batch calls could be merged across pages — fewer, larger calls.
That serialises the pages this design wants to overlap. Four pages in flight gives
four concurrent batch calls: good use of the admission limit, more calls.

Which way to lean depends on contention. For one client the pool has spare
capacity and parallelism wins. For several clients capacity is the scarce thing,
and fewer, longer calls are the better neighbour: each call carries fixed prompt
and scheduling overhead, and many short ones fragment the server's batching.

That is a reason to make page concurrency a setting rather than a constant. It is
also why the call-count reduction above is kept even though it saved no wall
clock.

## Expected gain

If model time divides by 4 and in-process work overlaps with it, the floor is
`max(model/4, in-process) + fixed overhead`:

- born-digital: `max(24.7, 13.1) + 4 ≈ 29s`, from 116s.
- scanned: `max(21.0, 17.7) + 5 ≈ 26s`, from 106s.

Both land near 4x, which is the pool's admission limit. Closing the page-cost
spread adds utilisation on top. This is arithmetic on the baseline, not a
measurement.

## Risks

- **Memory.** N pages in flight means N rasters and N renders held at once. The
  bound is what keeps that predictable.
- **Failure containment.** Firing every page's calls at once removes the
  backpressure a bounded client gives. A generous upper bound is cheap insurance
  against a stalled pool parking every thread.
- **Timeouts.** With a deep queue a call's latency is queue wait plus inference.
  A deadline that starts at admission rather than submission keeps the tail from
  failing spuriously; that belongs in the pool, and matches the `wait`/`work`
  split measured here.
- **A stalled call must not hold a slot.** Measured twice: one call took 64.2s
  where its replay took 7.6s, and another 51.3s against a 3.3s median. Serially
  that is a slow document. With N pages in flight it is worse — the call occupies
  one of the four admissions for a minute while the other pages queue behind it.
  A timeout with one retry bounds the damage to that page; without one, page
  concurrency makes a rare stall more expensive, not less.
- **Fixtures.** Pages are independent, so the PDF harness should be unaffected.
  That needs checking, not assuming.
