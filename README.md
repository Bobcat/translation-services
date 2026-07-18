# translation-services

A FastAPI service for **queued image-translation jobs**. You submit a request
(an image plus a source/target language), it runs through a task pipeline — OCR,
structural grouping, translation, and re-rendering the translated text back onto
the image — and you poll for the result and its artifacts.

The service owns translation *input/output and orchestration*. It does not run
language models itself: text translation and the vision grouping hint are
delegated to a separate model pool (`llm-pool`) over HTTP, and OCR runs locally
via PaddleOCR.

## Index

- [What It Does](#what-it-does)
- [Repository Role](#repository-role)
- [Related Repositories](#related-repositories)
- [Code Map](#code-map)
- [API Surface](#api-surface)
- [Runtime Model](#runtime-model)
- [Configuration](#configuration)
- [Development](#development)
- [Tests](#tests)
- [Regression Testing](#regression-testing)
- [Deployment Notes](#deployment-notes)
- [License](#license)

## What It Does

- Accepts an uploaded image (`image/jpeg`, `image/png`, `image/webp`) and a
  source/target language, and returns a **translated image**: the original text
  is erased and the translation is re-rendered in place, matching line geometry,
  size hierarchy, font family/weight, and alignment.
- Runs the `translate_image` pipeline in stages:
  1. **OCR** — PaddleOCR detects and recognizes text cells (with merge and
     upright re-recognition passes). OCR cells stay authoritative for text and
     bounding boxes.
  2. **Grouping** — a vision-language model returns a structural hint (reading
     order, hierarchy level, font, alignment); the aligner maps it back onto the
     OCR cells and builds translation **units**. A weak hint lowers quality but
     does not fail the job.
  3. **Translation** — units are translated through `llm-pool`, with
     language-pair routing and an optional prompt from the prompt library.
  4. **Replacement** — the renderer erases the source text and draws the
     translation back onto the image footprint.
- Supports **re-translation** (`retranslate_image`): reuse a prior request's
  cached OCR/grouping and only re-run translation with a different prompt or
  target language — no OCR/VLM again.
- Stores every stage as an inspectable **artifact** per request (grouping,
  segments, translation, debug overlays, the rendered output, and the raw model
  calls).
- Exposes a small **prompt library** (CRUD) of saved translation system prompts.

## Repository Role

This repo owns:

- The HTTP API and the request lifecycle (submit → queue → run → poll).
- The job scheduler (FIFO queue, bounded concurrency, lifecycle records with
  TTL).
- The image pipeline stages (OCR, grouping/alignment, translation routing,
  re-placement rendering) and their artifacts.

It deliberately does **not** own:

- The translation or vision models — they run in `llm-pool` (see Related
  Repositories).
- Any client UI — callers are separate apps.

## Related Repositories

- **`llm-pool`** — serves the translation and vision-language models over an
  HTTP `/v1/responses` API. `translation-services` calls it for both the
  grouping hint and text translation; the model names are set in configuration.
  Without it, OCR still runs but translation fails.
- **Client apps** — any frontend that submits requests to this service's `/v1`
  API (for example a translation workbench, or a webapp's image/camera flow).

## Code Map

`app/main.py` is the composition root and ASGI entry (`app.main:app`). It wires
settings → runtime and registers the HTTP routes. Everything else is grouped by
concern:

```
app/
  main.py            # HTTP composition root + route registration
  core/              # settings, request/response schemas, helpers
  runtime/           # job execution: FIFO queue, runner loop, lifecycle store
  tasks/             # feature pipelines — one module per task (the readable flows)
    translate_image.py
    retranslate_image.py
    translate_pdf.py
  ocr/               # OCR component: paddle backend, cell merge, segment, overlay
  grouping/          # VLM hint + aligner -> translation units
  translation/       # language-pair routing, llm-pool translate, prompt library
  replacement/       # erase + re-render: geometry, fit, color, render
  pdf/               # PDF intake: census, page raster, text-layer cells, assembly
  benchmark/         # document-pair benchmark: measurement, scoring, run store
```

To see how a feature works, open `tasks/<task>.py` — it reads as a recipe that
calls named stages.

## API Surface

All routes are versioned under `/v1`.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/requests` | Submit a job. Multipart: `request_json` (a JSON object) + `image_file` (image tasks) or `document_file` (a PDF, task `translate_pdf`). Returns the lifecycle envelope with a `request_id`. |
| `GET` | `/v1/requests/{id}` | Poll the request lifecycle (`state`, `stage`, `timings`, `response`, `error`). |
| `POST` | `/v1/requests/{id}/cancel` | Request cancellation. |
| `POST` | `/v1/requests/{id}/retranslate` | Re-translate a completed request's cached units with a new prompt/target language. |
| `POST` | `/v1/requests/{id}/rerender` | Re-render a completed request's cached translations with new render flags (`render_size_mode`/`erase_fill_mode`/`width_fit_mode`/`size_metric_mode`/`size_cohort_mode`); no new translation. |
| `GET` | `/v1/requests/{id}/artifacts/{name}` | Fetch a stored artifact (e.g. `rendered.png`, `translation.json`). |
| `GET` `POST` | `/v1/prompts` | List / create saved prompts. |
| `GET` `PUT` `DELETE` | `/v1/prompts/{id}` | Read / update / delete a saved prompt. |
| `GET` | `/v1/completions` | Poll completed-request events (`events` + `next_seq` cursor). |
| `GET` | `/v1/status` | Service status / health. |
| `GET` | `/v1/benchmark/results` | Stored document-benchmark runs (the comparison matrix). |
| `GET` | `/v1/benchmark/testset` | PDF testset documents available as benchmark sources. |
| `POST` | `/v1/benchmark/run` | Measure + score a pair: `{request_id}` of a completed `translate_pdf` run, or an uploaded pair + system label. |
| `GET` | `/v1/benchmark/runs/{doc}/{system}` | Detail (per-page scores) of the latest stored run. |
| `GET` | `/v1/benchmark/runs/{doc}/{system}/{run}/overlay/{side}/{page}` | Region-overlay render of a measured page. |

`request_json` fields: `task` (`translate_image` \| `retranslate_image` \| `rerender_image` \| `translate_pdf`),
`source_lang_code` (**required for routing**), `target_lang_code`, optional
`translator_model`, `translator_mode` (`generic` \| `translategemma`),
`grouping_model`, `translation_prompt` / `translation_prompt_id`,
`source_request_id` (for re-translate), `metadata`, `request_id`.

Lifecycle `state`: `queued` → `running` → `completed` \| `failed` \|
`cancelled` (with `cancel_requested` in between).

Submit example:

```bash
curl -sS -X POST http://127.0.0.1:8030/v1/requests \
  -F 'request_json={"task":"translate_image","source_lang_code":"en","target_lang_code":"nl"};type=application/json' \
  -F 'image_file=@input.jpg;type=image/jpeg'
```

## Runtime Model

- Requests are admitted to a **FIFO queue** (bounded by `scheduler.queue_limit`)
  and executed by a fixed number of concurrent **runner slots**
  (`scheduler.runner_slots`). uvicorn runs without `--reload`, so code edits
  require a restart.
- Each request gets a directory under the configured `work_root` holding the
  uploaded input and every stage artifact (grouping, segments, translation,
  debug overlays, `rendered.png` / `output.png`, and the raw `llm_calls/`).
- Terminal records are retained in memory with a per-state **TTL**
  (`scheduler.records_ttl_s`) and capped at `scheduler.records_max`.
- External dependency: `llm-pool` at `llm_pool.base_url` for the grouping hint
  and translation calls.

## Configuration

Settings load from `config/settings.json`, with an optional `config/local.json`
overlaying per-host overrides (deep-merged, gitignored). The settings file path
can be overridden with the `TRANSLATION_SERVICES_SETTINGS_PATH` environment
variable.

Key sections (see `app/core/config.py` for all defaults):

- `service` — `host`, `port` (default `8030`), `log_level`, `work_root`,
  `prompts_root`.
- `scheduler` — `runner_slots`, `queue_limit`, `records_max`, `records_ttl_s`.
- `llm_pool` — `base_url`, `translator_model`, `translator_mode`,
  `grouping_model`, `request_timeout_s`. The model fields are names the pool
  resolves; set them to models the pool actually serves.
- `ocr` — `backend`, `language` (empty = route the recognizer per image from the
  grouping hint), `min_confidence`, `device` (`cpu` / `gpu:0`), `ocr_version`,
  detection limits, and optional explicit `det_model` / `rec_model`.
- `inpaint` — `model_path` (TorchScript checkpoint for the model-based erase
  fill, used by `erase_fill_mode="inpaint"`; GPU-only), `pixel_budget_px` (hard
  cap on the crop area fed to the model, bounds its VRAM use).

`translator_mode`: `generic` uses a target-language prompt and auto-detects the
source; `translategemma` uses the dedicated model and needs `source_lang_code`.

## Development

Requires Python 3.11+.

```bash
python -m venv .venv
.venv/bin/pip install -e .
.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8030
```

PaddleOCR / PaddleX are heavy dependencies; the first OCR run downloads model
weights. OCR can run on CPU or GPU via `ocr.device`. Translation requires a
reachable `llm-pool`.

## Tests

```bash
.venv/bin/python -m pytest tests/
```

`tests/` covers the API surface (`test_api.py`), grouping/alignment
(`test_grouping.py`), re-placement rendering (`test_replacement.py`), and
translation routing (`test_translate.py`).

## Regression Testing

The testset grows as new cases are added, and a fix aimed at one image can change
the grouping, alignment or render of others. Checking every image by hand after
each change doesn't scale — so it gets skipped, and a regression slips through
unnoticed.

Automating it isn't as simple as re-rendering and comparing, because the pipeline
isn't fully deterministic: for the same image, two runs usually agree but can
differ slightly — a different element alignment, or an equally valid translation.
A plain re-render would flag that normal variation as a regression.

The fix is to **freeze the non-deterministic stages and re-run the deterministic
ones:**

- Freeze the grouping **hint** (from the VLM), the OCR **cells**, and the
  **translations**.
- Re-run `parse_grouping_output` → `group_cells_into_units` (align) →
  `render_translated_image` on them.

With the code unchanged, that chain produces the same result every time, so any
diff is a real code regression. The test pins the code, not the models. It's also
fast: a replay makes no VLM or translation calls per image.

The render is compared by reading the image back with OCR — **re-OCR** — and
checking the recognised text and box positions, not the raw pixels. That makes it
a *behavioural* check, not a pixel-exact one. It's reliable because OCR is
**bit-stable**: identical pixels re-OCR to identical text and boxes, so the
comparison adds no noise. And it catches real changes — a no-op replay
passes, a one-line render change (e.g. +2px erase pad) fails, and reverting it
passes again.

### What it tests

| stage | in a regression run |
|---|---|
| VLM grouping hint | **frozen** (raw string) — `hint_parser` still re-runs |
| OCR cells | **frozen** (OCR is deterministic) |
| alignment (`grouping/align.py`) | **re-runs** — tested |
| translation | **frozen** — one approved variant, attached to the re-grouped units |
| render (`replacement/render.py`) | **re-runs** — tested |

Alignment and render are the deterministic stages where the bugs live; the hint
parser runs too. Translation *quality* stays frozen, so it's out of scope here.

### How a result is compared

- **Alignment** — compared exactly, field by field: each unit's ordered cells,
  which cells were ignored, and the render-relevant fields the aligner sets
  (per-member translate flag, font family/weight, level, alignment). A failure
  points at the exact unit and field.
- **Render** — the re-OCR of the replayed image is checked two ways:
    - **text** — every segment must be present with matching text. OCR is exact
      on identical pixels, so a change in how a line is split or grouped is caught.
    - **position** — each matched segment must land within **3px**.

  Same machine and fonts make this effectively exact; the 3px only absorbs
  sub-pixel anti-aliasing.

### Approve and replay

A fixture freezes the exact result you approved in the workbench. After that:

- **Replay** one fixture or all of them; each gets a pass/fail against its
  snapshot, with per-stage timings (align, render, re-OCR).
- **Re-baseline** when a change to align or render is intended and correct.

Each fixture carries its own source image, so a replay never drifts against an
external file. Fixtures are organised per target language, and a language can hold
several variants (`v1`, `v2`, …) — each freezing a different valid hint or
translation, which aligns and renders differently. They're curated cases, not a
growing accept-set.

Full design — freeze boundary, schema, comparison rules, the measured variance
study and the regression API endpoints:
**[docs/regression-test-design.md](docs/regression-test-design.md)**.

## Deployment Notes

A systemd **user** unit and start script live under `deploy/systemd/` (the start
script reads the port from `service.port` or `DEFAULT_PORT`). See
`deploy/systemd/README.md` for install commands and the per-host venv override.

**Render fonts are a prerequisite** and are not vendored: the renderer loads
metric-compatible Latin faces (and Noto Sans KR for Korean) from a per-host
fonts directory; Han/Kana fonts are fetched lazily by PaddleX. A missing font
degrades gracefully (Latin falls back; CJK/Korean render as tofu), so fonts must
be provisioned per host for correct output. Exact filenames and install steps
are in `deploy/systemd/README.md`.

## License

No license file is present in this repository.
