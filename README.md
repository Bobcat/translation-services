# translation-services

A FastAPI service for **queued translation jobs**. You submit a request (an
image, later also plain text or a PDF), it runs through a task pipeline, and you
poll for the result and its artifacts.

It owns translation *input/output and orchestration*; it does not run language
models itself — text translation is delegated to **`llm-pool`** over HTTP, and
OCR runs locally via **PaddleOCR**.

## Tasks

| Task | In → out | Status |
|---|---|---|
| `translate_image` | image → translated image | in development (grouping + translation wired; rendering next) |
| `translate_text` | text → translated text | planned (reuses routing + translation core) |
| `translate_pdf` | pdf → translated pdf | later |

A request carries a `task`; each task is a pipeline composed of shared stages.

## Architecture (composition root)

`app/main.py` is the composition root and ASGI entry (`app.main:app`). It wires
settings → runtime and exposes the API. Everything else is grouped by concern:

```
app/
  main.py            # HTTP composition root
  core/              # settings, request/response schemas, helpers
  runtime/           # job execution: FIFO queue, runner loop, lifecycle store
  tasks/             # feature pipelines — one module per task (the readable flows)
    translate_image.py
  ocr/               # OCR component: paddle backend, merge, overlay
  grouping/          # grouping component: VLM hint + aligner -> translation units
  translation/       # translation component: language-pair routing + llm-pool translate
```

To see how a feature works, open `tasks/<task>.py` — it reads as a recipe that
calls named stages.

## The translate_image pipeline

Vocabulary: a **cell** is one OCR box (text + bbox + confidence). A **translation
unit** is a group of cells that are translated together, tagged **`flow`** (lines
of one continuous text, e.g. a wrapped heading or paragraph) or **`field`** (a
standalone label or value, e.g. a price or a table cell).

| # | Stage | What | Status |
|---|---|---|---|
| 1 | **Ingest** | store canonical input (EXIF-transposed) | ✅ done (`app/main.py`) |
| 2 | **OCR** | PaddleOCR → cells (detection resolution via `text_det_limit_side_len`); the recognition model is routed per image on the #5 hint's script (Han/Kana → multilingual server pair, else the en recognizer), so the hint is requested first | ✅ done (`ocr/`) |
| 3 | **Orientation rescue** | re-recognise low-confidence cells by cropping + rotating (fixes garbled rotated text) | ⏳ planned |
| 4 | **Coverage gate** | compare the VLM transcription (#5) to the cells; escalate OCR (higher `text_det_limit_side_len`) when much is missing | 🟡 byproduct of #5 available; gate itself parked |
| 5 | **Grouping** (VLM-based) | a VLM groups text that belongs together (a heading, paragraph or item) so each coherent piece is translated as one unit; an aligner maps the groups onto the OCR cells | ✅ done (`grouping/`) |
| 6 | **Routing** | single direct A→B path: the configured translator model + mode (seam for richer routing later) | ✅ done (`translation/routing.py`) |
| 7 | **Translation** | call `llm-pool` (`/v1/responses`) per unit | ✅ done (`translation/translate.py`) |
| 8 | **Re-placement** | render translated text back into the image ([design directions](docs/re-placement.md)) | 🟡 Tier-1 done (model-free simple replace) |

OCR cells stay **authoritative** for text + bbox; the VLM is only a grouping
*hint*, so a weak/incomplete VLM lowers quality but does not fail the job. That
same VLM transcription doubles as the coverage reference for #4.

Translation (#7) is **one `llm-pool` call per unit**, so dense images (receipts,
menus) are costly. Batching several units into one call — if the translator
handles it well without cross-talk — is a future optimization, not yet tried.

## Current status

Phase: **Tier-1 re-placement.** `translate_image` runs ingest → VLM hint (also
routes the OCR model) → OCR → grouping alignment → per-unit routing + translation
→ re-placement, and returns the OCR cells, the
translation units (each with `translated_text`, `kind`, members), a grouping debug
overlay, and the **`rendered`** artifact — the translated image (Tier-1 model-free
simple replace; see [docs/re-placement.md](docs/re-placement.md) for the upgrade
path). Translations also live in the response JSON (`ocr.translation_units`,
`metadata.full_translated_text`).

## Topology

Runs on **dc1** (RTX 5070 Ti, port 8030). OCR runs locally on the GPU
(PaddleOCR / PP-OCRv5, paddlepaddle-gpu cu130, ~2.5 GB).
Translation is delegated to **`llm-pool` on dc2** via the existing SSH tunnel
(`llm-pool-dc2-tunnel`, `127.0.0.1:8011`), where the larger translation models
live.

## API

| Method | Path | |
|---|---|---|
| POST | `/v1/requests` | submit a job (`request_json` form field + `image_file`) |
| GET | `/v1/requests/{id}` | lifecycle + result |
| POST | `/v1/requests/{id}/cancel` | cancel |
| GET | `/v1/requests/{id}/artifacts/{name}` | fetch an artifact (image/json) |
| GET | `/v1/completions` | recent completion events |
| GET | `/v1/status` | queue / runner status |

`request_json` fields: `task`, `source_lang_code`, `target_lang_code`, optional
`translator_model` / `translator_mode`, optional `grouping_model`, optional `request_id`.

```bash
curl -sS -X POST http://127.0.0.1:8030/v1/requests \
  -F 'request_json={"task":"translate_image","source_lang_code":"en","target_lang_code":"nl"};type=application/json' \
  -F 'image_file=@input.jpg;type=image/jpeg'
```

## Configuration

`config/settings.json` (version-controlled) + optional `config/local.json`
(per-machine, gitignored, deep-merged). Key sections: `service`, `scheduler`
(`runner_slots`, `queue_limit`), `ocr` (`backend`, `language`, `device`,
`ocr_version`, `min_confidence`, `text_det_limit_side_len`, `text_det_limit_type`),
`llm_pool` (`base_url`, `translator_model`, `translator_mode`, `grouping_model`).

Translation is a single A→B path: `translator_model` is called in `translator_mode`
(`generic` = a target-language prompt, auto-detects the source; `translategemma` =
the dedicated model, needs `source_lang_code`). Per-pair/content routing was removed
and will be reintroduced in `translation/routing.py` if testing shows the need.

## Development

```bash
python3 -m venv .venv && .venv/bin/python -m pip install -e .
.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8030
.venv/bin/python -m pytest tests/test_api.py
```

Deploy: systemd user unit in `deploy/systemd/` (symlinked into
`~/.config/systemd/user/translation-services.service`).
