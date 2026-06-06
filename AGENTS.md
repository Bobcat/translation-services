# AGENTS.md

Working notes for agents. Read `README.md` first — it describes the service, the
`app/` layout (composition root), the translate_image pipeline stages, and the
current status.

## Navigation

- `app/main.py` — composition root + ASGI entry (`app.main:app`); HTTP routes.
- `app/core/` — settings, schemas, helpers.
- `app/runtime/` — job execution: FIFO queue, runner loop, lifecycle store.
- `app/tasks/` — one module per task; the readable pipeline for a feature.
- `app/ocr/` — OCR component (paddle backend, merge, overlay, doc-unwarp).
- `app/translation/` — translation component (language-pair routing).

New pipeline stages land in their component (`ocr/`, `translation/`) and are
wired into the task in `tasks/`.

## Working style

- Keep changes small and phased; state what changes and why.
- Leave unrelated code untouched. No opportunistic refactors.
- Do not add fallback/compatibility paths unless explicitly requested; do not
  leave dead code "just in case".
- Single vocabulary: the queued unit is a **request**; do not reintroduce
  "image-pool"/"pool"/"job" synonyms.
- Do not run `git commit`/`git push` unless explicitly requested.

## Verification

- `.venv/bin/python -m pytest tests/test_api.py` after changes.
- Restart `translation-services.service` (dc1) when testing live; smoke a
  request against `/v1/requests` and check `/v1/status`.
