from __future__ import annotations

from typing import Any
from typing import Literal

from pydantic import BaseModel
from pydantic import Field


TaskName = Literal["translate_image", "retranslate_image"]
RequestState = Literal["queued", "running", "completed", "failed", "cancelled", "cancel_requested"]
TranslatorMode = Literal["translategemma", "generic"]


class RequestPayload(BaseModel):
    request_id: str | None = None
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
