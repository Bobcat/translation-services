from __future__ import annotations

from typing import Any
from typing import Literal

from pydantic import BaseModel
from pydantic import Field


TaskName = Literal["translate_image", "retranslate_image", "rerender_image", "translate_pdf"]
RequestState = Literal["queued", "running", "completed", "failed", "cancelled", "cancel_requested"]
TranslatorMode = Literal["translategemma", "generic"]


class RequestPayload(BaseModel):
    # Records key on the raw id but directories on ``safe_token(id)`` — an unconstrained id
    # lets two distinct requests collide on one work dir (pruning one deletes the other's
    # artifacts). Constrain at the edge to exactly what safe_token passes through unchanged.
    request_id: str | None = Field(default=None, max_length=120, pattern=r"^[A-Za-z0-9._-]+$")
    task: TaskName
    source_lang_code: str | None = None
    target_lang_code: str | None = None
    translator_model: str | None = None
    translator_mode: TranslatorMode | None = None
    grouping_model: str | None = None
    # Structured-route prompt selection (see app/translation/prompts). Precedence:
    # ``translation_prompt`` (an ad-hoc raw system prompt) > ``translation_prompt_id`` (a
    # saved library prompt) > the pipeline default. ``source_request_id`` points the
    # ``retranslate_image`` task at a prior completed run whose cached grouping/units it
    # reuses (no VLM/OCR/grouping again).
    translation_prompt: str | None = None
    translation_prompt_id: str | None = None
    source_request_id: str | None = None
    # Render the OCR + grouping debug overlays (each a full-image PNG, ~0.5-1s). Off by default so
    # a non-debug caller (e.g. the asr camera app, which only fetches the rendered translation)
    # skips that work; the workbench sets it true to populate its overlay artifact dropdown.
    debug_overlays: bool = False
    # Preserve text selected by local heuristics (prices, URL-only fields, codes) as original pixels.
    preserve_heuristic_text: bool = True
    # Preserve text whose translated output is effectively identical to the source.
    preserve_unchanged_text: bool = False
    # Feed translation the geometry-adjusted hints (a `|` injected where OCR cell gaps show a column
    # the VLM missed) instead of the raw VLM hints, so a `label  value` row renders per column.
    use_geometry_columns: bool = True
    # Keep text inside detected image/chart regions as original pixels (a screenshot's UI labels,
    # a plot's axes). False translates and renders it too — for a "figure" that is really a table
    # screenshot whose cells you want translated.
    preserve_image_regions: bool = True
    # How a render group's single font size is chosen from its lines' measured heights: "min"
    # never overflows the smallest line's band; "median" resists one under-measured (lowercase)
    # line dragging the whole block down. Default "median" since 2026-07-06: evaluated across
    # the testset — clearly better renders, no regressions seen.
    render_size_mode: Literal["min", "median"] = "median"
    # How erased source text is filled. "flat" paints each erased line with its sampled
    # background colour; "inpaint" is the hybrid model-based fill — flat paint on designed
    # flat ground, model reconstruction where the ground varies
    # (app/replacement/ground/inpaint.py). Default "inpaint" since 2026-07-09: the ground
    # router keeps designed-flat ground on the flat paint, so the model only runs where it
    # visibly wins (photo texture, gradients). Requires a GPU + checkpoint — on a box
    # without either, requests must pass "flat" explicitly.
    erase_fill_mode: Literal["flat", "inpaint"] = "inpaint"
    # How a translation wider than its original line is fitted. "footprint" keeps it inside
    # the original line's width (condense in x, then shrink pt); "extend" first widens the
    # usable width into VERIFIED clean background right of the line (never over other text,
    # ink or a surface change), so a short list item's longer translation keeps its size.
    # "extend_to_margin" is that same growth with one added ceiling: the right margin of the
    # text band the line sits in (a document's own margin, and each column's gutter on a
    # multi-column page). Plain "extend" knows only the image edge, which is the right frame
    # for a photo or sign — where a line SHOULD be free to grow into empty design space —
    # and the wrong one for a document, where it grows across the gutter or into the margin.
    width_fit_mode: Literal["footprint", "extend", "extend_to_margin"] = "footprint"
    # Where a line's SOURCE SIZE comes from. "extent" (default) sizes from the OCR
    # polygon's full ink extent; "band" clamps each line to its strong ink band scaled
    # by the document's own extent/band norm, so sparse tall glyphs (parentheses — as a
    # marker or mid-text — brackets) cannot inflate a line's size past its siblings.
    # One-sided: "band" only ever shrinks an outlier, and weak ink evidence keeps the
    # extent behaviour.
    # "fill" sizes each line so its rendered ink is as tall as the source line's ink (a direct
    # per-line pixel match, self-calibrating on the mapped face), fixing the ~10-20% undershoot
    # where translated body text reads smaller/airier than the print. Flat, non-CJK groups only.
    size_metric_mode: Literal["extent", "band", "fill"] = "extent"
    # Cross-element size uniformity from the VLM's per-element font-size (pt) label. "off"
    # sizes each element from its own OCR true-height. "vlm" groups elements the VLM gave one
    # pt into a size cohort and — when their OCR heights AGREE (the VLM's equal claim holds) —
    # snaps the whole cohort to its OCR median, so a list the VLM judged one size renders
    # uniform (and a short item sizes up and re-wraps over its lines instead of collapsing to
    # one tiny line). A cohort whose OCR heights disagree keeps per-element sizing. Default
    # "vlm" since 2026-07-11: evaluated across the testset — visibly calmer renders, no
    # regressions observed; the OCR-agreement gate bounds any snap to the spread already there.
    size_cohort_mode: Literal["off", "vlm"] = "vlm"
    # translate_pdf only: feed born-digital pages the PDF text layer as cells
    # (exact text + style, no OCR). False forces OCR on every page — the A/B
    # lever for measuring the text-layer path against the raster path.
    use_pdf_text_layer: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class RequestLifecycle(BaseModel):
    request_id: str
    state: RequestState
    task: TaskName
    queue_position: int | None = None
    submitted_at_utc: str
    started_at_utc: str | None = None
    finished_at_utc: str | None = None
    stage: str | None = None
    # Per-page progress of a running translate_pdf document; None for image tasks.
    pages_done: int | None = None
    pages_total: int | None = None
    timings: dict[str, float] = Field(default_factory=dict)
    response: dict[str, Any] | None = None
    error: dict[str, Any] | None = None


class RequestSubmitEnvelope(RequestLifecycle):
    pass


class CompletionEvent(BaseModel):
    seq: int
    event: str
    request_id: str
    state: RequestState
    task: TaskName
    submitted_at_utc: str
    finished_at_utc: str | None = None
    response: dict[str, Any] | None = None
    error: dict[str, Any] | None = None


class CompletionsEnvelope(BaseModel):
    events: list[CompletionEvent] = Field(default_factory=list)
    next_seq: int = Field(default=0, ge=0)
    # Per-process id: seq numbers restart at 1 on a service restart, so a poller must reset its
    # cursor whenever this changes — without it a stale large cursor waits forever.
    instance_id: str = ""
