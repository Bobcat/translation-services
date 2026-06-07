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

Vocabulary: a **cell** is one OCR box (text + bbox + confidence); a **translation
unit** is a group of cells; a **route** is the OCR mode (`scene`).

| # | Stage | What | Status |
|---|---|---|---|
| 1 | **Ingest** | store canonical input (EXIF-transposed) | ✅ done (`app/main.py`) |
| 2 | **OCR** | PaddleOCR `scene` route → cells (detection resolution via `text_det_limit_side_len`) | ✅ done (`ocr/`; adaptive `auto` planned) |
| 3 | **Orientation rescue** | re-recognise low-confidence cells by cropping + rotating (fixes garbled rotated text) | ⏳ planned |
| 4 | **Coverage gate** | compare the VLM transcription (#5) to the cells; escalate OCR (higher `text_det_limit_side_len`) when much is missing | 🟡 byproduct of #5 available; gate itself parked |
| 5 | **Grouping** (VLM-based) | VLM transcribes + groups the image; aligned back onto the OCR cells → translation units (`flow`/`field`) in reading order | ✅ done (`grouping/`) |
| 6 | **Routing** | single direct A→B path: the configured translator model + mode (seam for richer routing later) | ✅ done (`translation/routing.py`) |
| 7 | **Translation** | call `llm-pool` (`/v1/responses`) per unit | ✅ done (`translation/translate.py`) |
| 8 | **Re-placement** | render translated text back into the image | ⏳ to build |

OCR cells stay **authoritative** for text + bbox; the VLM is only a grouping
*hint*, so a weak/incomplete VLM lowers quality but does not fail the job. That
same VLM transcription doubles as the coverage reference for #4.

## Current status

Phase: **translation wired; rendering next.** `translate_image` runs ingest → OCR
→ VLM grouping → per-unit routing + translation, and returns the OCR cells, the
translation units (each with `translated_text`, `kind`, members) and a grouping
debug overlay. Re-placement (#8) is not done yet, so the **output image is still
the debug overlay**; the translations live in the response JSON
(`ocr.translation_units`, `metadata.full_translated_text`).

## Topology

Runs on **dc1** (RTX 5070 Ti, port 8030). OCR runs locally on the GPU
(PaddleOCR / PP-OCRv5, paddlepaddle-gpu cu130, ~2.5 GB for the scene route).
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

`request_json` fields: `task`, `source_lang_code`, `target_lang_code`,
`ocr_route` (`scene` only — the `document` route was removed; field kept for the
planned `auto` route and is a removal candidate), `ocr_unwarp` (bool), optional
`translator_model` / `translator_mode`, optional `grouping_model`, optional `request_id`.

```bash
curl -sS -X POST http://127.0.0.1:8030/v1/requests \
  -F 'request_json={"task":"translate_image","source_lang_code":"en","target_lang_code":"nl","ocr_route":"scene"};type=application/json' \
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
