from __future__ import annotations

from typing import Any
from typing import Literal

from pydantic import BaseModel
from pydantic import Field


TaskName = Literal["translate_image", "retranslate_image", "rerender_image"]
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
    # How a render group's single font size is chosen from its lines' measured heights: "min"
    # never overflows the smallest line's band; "median" resists one under-measured (lowercase)
    # line dragging the whole block down. Default "median" since 2026-07-06: evaluated across
    # the testset — clearly better renders, no regressions seen.
    render_size_mode: Literal["min", "median"] = "median"
    # How erased source text is filled. "flat" paints each erased line with its sampled
    # background colour; "inpaint" is the hybrid model-based fill — flat paint on designed
    # flat ground, model reconstruction where the ground varies (app/replacement/inpaint.py).
    erase_fill_mode: Literal["flat", "inpaint"] = "flat"
    # How a translation wider than its original line is fitted. "footprint" keeps it inside
    # the original line's width (condense in x, then shrink pt); "extend" first widens the
    # usable width into VERIFIED clean background right of the line (never over other text,
    # ink or a surface change), so a short list item's longer translation keeps its size.
    width_fit_mode: Literal["footprint", "extend"] = "footprint"
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
