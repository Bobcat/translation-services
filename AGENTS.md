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
- Do not run `git commit`/`git push` unless explicitly requested.

### Generalize, don't curve-fit (OCR / grouping / replacement)

A fix prompted by one test image must hold for unseen images, not fit that one.
For every such change, state in the response whether it generalizes and why:

- **Principle, not pixels.** The fix must rest on a geometric/structural fact
  true of the image class (a tilted line's axis bbox spills past the ink; a
  photographed line fans by perspective), not on values read off one image.
- **No-op where it shouldn't apply.** Prefer changes that reduce to the old
  behaviour on the cases they don't target (oriented sampling == axis sampling
  at angle ≈ 0; a baseline fit == 0° on flat text). A strict no-op can't regress.
- **Safe fallback** when inputs are too thin to support the new path (one-word
  line → fall back to the quad angle), never a guess.
- **Nothing image-specific** hard-coded: no literal colour, no constant tuned to
  make one fixture look right.
- **Validate on another image** before claiming generality: re-render a second
  tilted/photographed fixture (e.g. `afstand-houden`, `menukaart`) and confirm no
  regression — argument alone is not enough.
- **Name the honest limit.** Say what the change does NOT fix (flat fill still
  scars on textured backgrounds — the LaMa/Tier-2 case; a straight tile can't
  follow a curved surface), so the boundary is explicit.

## Verification

- `.venv/bin/python -m pytest tests/test_api.py` after changes.
- **Always restart the service after changing service code** so the user can test:
  `systemctl --user restart translation-services.service` (dc1). uvicorn runs
  without `--reload`, so edits are not picked up until restart. Then smoke a
  request against `/v1/requests` and check `/v1/status`.
