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

Grouping (a VLM) and translation (an LLM) run in a separate process — the *model
pool* — which serves inference requests and admits only a few at a time
(`target_inflight`). `translate_pdf` runs in this service, not in the pool; it
calls the pool for the grouping and translation of each page.

It works through a document one page at a time, and within a page it makes its
calls in sequence: grouping, then the translation batch, then the per-line calls
the batch could not place. Nothing in one document is issued to the pool
concurrently, so a document occupies at most one of the pool's slots at any moment
and the rest sit idle. Separate *requests* already overlap — the runtime runs more
than one — but within a request nothing does.

A document's pages share no data until the PDF is assembled, so their model calls
could be in flight together instead of in sequence. That idle pool capacity is the
headroom.

## Baseline

`translate_pdf`, one page at a time. Three documents, each run five times:

- a 15-page paper, born-digital (text layer, OCR skipped)
- a 200 dpi raster of that same paper — the OCR route on identical content. It is
  a clean re-render, not a real scan (no sensor noise, skew, or compression
  artefacts), so OCR has it easier than it would in the field
- an 11-page 1974 conference paper, a genuine scan (image coverage 1.00 on every
  page) with an untrusted OCR text layer, classified `hybrid` and therefore
  routed through OCR

| stage | born-digital | raster of it | 1974 scan |
|---|---|---|---|
| grouping (VLM) | 38.0 (37.8-38.6) | 38.1 (37.5-38.3) | 50.6 (50.2-52.5) |
| translation (LLM) | 71.4 (69.9-78.0) | 52.4 (51.9-56.1) | 81.2 (80.6-93.8)* |
| **out-of-process** | **109.6 (108.5-115.9), 89%** | **90.4 (89.4-94.2), 85%** | **132.5 (130.8-146.3), 84%** |
| OCR | 0 | 3.7 (3.7-4.7) | 3.5 (3.3-3.9) |
| render | 10.6 (10.4-10.7) | 10.1 (9.7-10.8) | 16.4 (16.0-17.2) |
| align | 1.9 (1.9-2.0) | 1.9 (1.8-1.9) | 4.9 (4.9-5.0) |
| layout | 0.5 (0.5-4.4) | 0.5 | 0.4 |
| **in-process** | **13.1 (12.8-17.0), 11%** | **16.2 (15.8-17.3), 15%** | **25.6 (24.8-25.9), 16%** |
| pages | 15 | 15 | 11 |

Seconds, as median (min-max) over the five runs. They cluster tightly — most stage
totals within ~6%. The spread comes from how many lines the grouping hint leaves
for the per-line tail, and from the odd slow call.

\* the upper end of the 1974 translation range (93.8s) is the one run where a
page's block structure did not come back intact and its lines fell to per-line
calls; the other four sat near 80.6s.

Three things follow.

**The pipeline mostly waits.** 84-89% of the time is spent on calls to another
process. That is the headroom.

**Our own GPU work is small, on a real scan too.** OCR costs 0.25s per page on
the clean raster and 0.31s per page on the 1974 scan. The in-process GPU models
(Paddle det/rec, doclayout, LaMa) are the stages that cannot run on more than one
thread, and the harder input did not change their weight. LaMa sits inside
`render` and is not timed separately, so the GPU share is measured only for OCR;
`render` also contains CPU work.

**Page cost varies a lot.** Across the born-digital runs the slowest page took
15.0s, the median 10.1s and the fastest 1.0s. Serial execution pays that spread in
full.

The raster run is faster, but not because OCR is cheaper. It does less: it flattens
the formulas the text-layer path carries as island tokens. The two right-hand
columns are not the same job.

## Two sources of parallelism

Per page the translator makes batched calls — one over the page's hint lines, one
over any inline-maths units — plus a per-line tail for what the batches could not
place. Over the born-digital document the 13 main batch calls took 45.1s (63% of
the translation stage) and the 9 island batches another 14.6s. The per-line tail
ran 10-19s across the five runs, varying with how many lines the grouping hint
left unplaced.

So there are two independent places to parallelise:

- **across pages** — pages share nothing until the PDF is assembled, so their
  batch calls can be in flight together. This is the main win.
- **within a page** — the tail calls do not depend on each other and can be
  issued together. This is the only source that also helps a single-page image
  request, where there are no other pages to overlap with.

## Resource classes

Each stage belongs to one of three classes, and each class needs a different
rule.

| class | stages | rule |
|---|---|---|
| out-of-process | grouping VLM, translation | issue freely; the pool owns capacity |
| in-process GPU | Paddle det/rec, doclayout, LaMa | per-engine `threading.Lock`; serial |
| CPU | align, wrap/fit, compositing | parallel, as far as the GIL allows |

The GPU engines are not thread-safe, but each already serialises itself on a
`threading.Lock` — `_PREDICT_LOCK` in layout, the per-engine locks in
`app/ocr/paddleocr.py`, `_LOCK` in inpaint. So page-parallel threads are already
safe; correctness needs no new lock. Lock-held time is small — a few seconds
against ~110s of model work. Whether to add one umbrella lock, so the engines never
overlap on the card, is a contention question for phase 4.

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

Out of scope in phase 1: concurrency and the config parameter.

### Phase 2 — page concurrency

A bounded thread pool over the `for page in profile.pages` loop in
`run_translate_pdf_pipeline` (`app/tasks/translate_pdf.py`), where each page calls
`run_translate_image_pipeline`. Threads, not asyncio: the pool calls are blocking
`httpx.post`, so threads give real overlap while each page keeps its serial chain
unchanged. The mix of stages in flight falls out of the staggering, so no scheduler
decides what runs when.

- page concurrency becomes a setting, not a constant.
- no new GPU lock — the in-process engines already self-serialise (see Resource
  classes); page threads are safe as they are.
- PDF assembly keeps page order regardless of completion order.
- per-page progress becomes a count of finished pages.
- `checkpoint()` and cancellation need a defined meaning with N pages in flight.
- each page's model calls carry a timeout and one retry, so a stalled call frees
  its shared admission slot instead of throttling the fan-out — a 64.2s call
  replayed in 7.6s, a 51.3s one against a 3.3s median.
- the timeout is measured from admission, not submission, so a deep queue does
  not trip it spuriously; phase 1's `wait`/`work` split marks admission.
- the PDF regression harness stays green; page output is per-page, so concurrency
  must not change it.

### Phase 3 — the within-page tail

The per-line tail is not small — 10-19s on the born-digital document — but it is
issued one call after another within a page. Under phase 2 one page's tail already
overlaps the next page's batch, so for a multi-page PDF page concurrency absorbs
most of it. Fanning the tail out within a page is what helps a single-page image
request, where there is no next page to overlap with — so it comes after page
concurrency, not before.

### Phase 4 — tune

With the timeline, set page concurrency against measured lock contention and
queue wait. Re-check the pool's draft-token setting on the parallel workload:
speculative decoding pays off less once the batch is full, so a value tuned
against single-stream calls is tuned against a situation that no longer exists.

## Fewer calls pulls against more parallelism

The 13 page-level batch calls could be merged across pages — fewer, larger calls.
That serialises the pages this design wants to overlap. Pages in flight give that
many concurrent batch calls: good use of the admission limit, more calls.

Which way to lean depends on contention. For one client the pool has spare
capacity and parallelism wins. For several clients capacity is the scarce thing,
and fewer, longer calls are the better neighbour: each call carries fixed prompt
and scheduling overhead, and many short ones fragment the server's batching.

That is a reason to make page concurrency a setting rather than a constant.

## Expected gain

If model time divides by 4 and in-process work overlaps with it, the floor is
`max(model/4, in-process) + fixed overhead`:

- born-digital: `max(27.4, 13.1) + 4 ≈ 31s`, from 124s.
- raster: `max(22.6, 16.2) + 5 ≈ 28s`, from 107s.

Both land near 4x, which is the pool's admission limit. Closing the page-cost
spread adds utilisation on top. This is arithmetic on the baseline, not a
measurement.
