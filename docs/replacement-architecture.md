# `app/replacement/` — package map

Stage #8 of the pipeline: render the translated text back into the image. This document is
the navigation map — read it to find the code behind a render detail, and to see the
top-level flow. It also tracks the in-progress restructure from one 1400-line `render.py`
into concern modules.

## The one public surface

Everything outside the package imports exactly one name: `render_translated_image` (via the
package `__init__`). The three pipeline tasks and the regression replay use only that; nothing
external reaches into a submodule. So the internals can be reorganised freely — the only things
that move with an internal change are `render.py`'s own imports and the tests.

## Reading the tree (symptom → module)

```
app/replacement/
  __init__.py        exports render_translated_image        (public surface)
  render.py          COMPOSITION ROOT — the pipeline flow; docstring is the map
  jobs.py            _Job — a placed render job (shared vocabulary)
  pixels.py          _INK_DELTA — "a pixel is ink" threshold (shared)
  geometry.py        polygon geometry + _plane_corners, _ANGLE_DEADZONE_DEG (shared)

  layout/            how translation units become placed jobs on the page
    groups.py          cluster consecutive units into render-groups
    planning.py        _plan_group — the group→jobs planner + its plane/member helpers
    tables.py          split a receipt row into per-column cells + field matching
    markers.py         bullets & enumerators: keep/redraw the marker, inset the text
    sweep.py           stray-ink cleanup on flat, angle-snapped images
    compositing.py     warp a text tile onto its plane and paste it

  text/              how the translated text is shaped
    angle.py           text-line tilt: the document angle field, baseline fit, flatness
    size.py            source size / band metric (how tall a line renders)
    wrap.py            break & condense a group's text onto its planes
    fit.py             font / script / text-width primitives (load_font, wrap_lines, ...)

  ground/            how the original is erased and the background filled
    color.py           sample background / foreground colour of a region
    erase.py           flat-vs-model ground router, the model erase mask, residue recovery
    inpaint.py         LaMa runtime (the Tier-2 model fill)
```

| you see, in the render… | look in |
|---|---|
| a line too big / too small | `text/size.py` |
| text tilted or not parallel to a band | `text/angle.py` |
| text broke to odd lines / over-condensed | `text/wrap.py` (font itself: `text/fit.py`) |
| a receipt column split or shifted wrong | `layout/tables.py` |
| a wrong / doubled / mis-inset bullet | `layout/markers.py` |
| a flat grey patch or leftover ink | `ground/erase.py` (+ `layout/sweep.py`) |
| a tile placed or warped wrong | `layout/compositing.py` |
| the wrong background colour painted | `ground/color.py` |

## Straggler placements (decided)

- `_image_is_flat` → `text/angle.py`. Flat-vs-tilted is fundamentally the text-angle question;
  it gates both the angle field and the size band.
- `_clean_right_extension` (width_fit "extend") → `layout/planning.py`. It widens a plane using
  pixel evidence before fitting — a plane-width planning decision, tightly bound to `_plan_group`.
- `_reproduced_in` → `layout/planning.py`. Member-selection helper (is a non-translate member's
  source actually in the translation, so it should be erased/redrawn) — the planner's machinery.

## Granularity calls

- `ground/erase.py` stays one file: router, model mask and residue read as one cohesive layer
  and were just committed together; split only if it grows.
- `text/` keeps primitives (`fit.py`) separate from group-to-plane fitting (`wrap.py`): "font
  wrong" and "line broke weird" are different questions.

## Shared layer (must move first to avoid import cycles)

Some helpers/constants are used by more than one concern, so they belong to the shared leaves,
not to any one concern (otherwise a concern would import the planner and cycle):

- `_Job` → `jobs.py`.
- `_INK_DELTA` → `pixels.py` (done).
- `_plane_corners`, `_ANGLE_DEADZONE_DEG` → `geometry.py`.
- `_line_clusters`, `_baseline_angle` live in `text/angle.py` but are also used by
  `layout/planning.py` — the dependency runs planning → text (downward), never the reverse.

## Execution order (each step: pure move + update test imports + tests/sweep green)

- [x] 1. `jobs.py` (`_Job`) — unblocks every concern's type reference.
- [x] 2. Shared geometry: `_plane_corners`, `_ANGLE_DEADZONE_DEG` into `geometry.py`.
- [x] 3. `text/` — `angle.py`, `size.py`, `wrap.py` (+ `layout/groups.py`, pulled early because
       the angle field depends on it). Moving the existing `fit.py` into `text/` is still to do.
- [x] 4. `ground/` — `color.py`, `erase.py`, `inpaint.py` moved in; `fit.py` moved to `text/`.
- [x] 5. `layout/` — `tables.py`, `markers.py`, `sweep.py`, `compositing.py`, `planning.py`.
- [x] 6. `render.py` is the composition root (~160 lines); its docstring carries the map.

Done — the tree above is the actual layout. `render.py` holds only `render_translated_image`
(the pipeline); every concern is a named module reached from a render symptom.

`_plan_group` itself (≈280 lines) moves whole in step 5; splitting its internals is a separate,
non-mechanical task, not part of this restructure.
