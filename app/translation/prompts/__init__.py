"""Domain-agnostic translation prompt library.

translation-services owns the prompts it applies. A prompt is a ``{{var}}`` template
pair (system + user); the pipeline resolves which prompt to use and renders it with
the call's variables. Resolution precedence for the structured route:

    explicit raw prompt string  >  prompt id  >  the pipeline's default id
"""
from __future__ import annotations

from app.translation.prompts.store import PromptConflictError
from app.translation.prompts.store import PromptNotFoundError
from app.translation.prompts.store import PromptStore
from app.translation.prompts.store import PromptStoreError
from app.translation.prompts.store import PromptValidationError
from app.translation.prompts.store import get_prompt_store
from app.translation.prompts.store import normalize_prompt_id
from app.translation.prompts.store import store_for
from app.translation.prompts.templates import DEFAULT_USER_TEMPLATE
from app.translation.prompts.templates import IMAGE_DEFAULT_ID
from app.translation.prompts.templates import PromptEntry
from app.translation.prompts.templates import render_template


def resolve_structured_prompt(
    store: PromptStore,
    *,
    raw_prompt: str | None,
    prompt_id: str | None,
    default_id: str,
) -> PromptEntry:
    """Pick the prompt for the structured route: an ad-hoc raw string wins (used as the
    system template, with the default user template), else the named id, else the
    pipeline default id."""
    raw = str(raw_prompt or "").strip()
    if raw:
        return PromptEntry(id="(adhoc)", system=raw, user=DEFAULT_USER_TEMPLATE)
    named = str(prompt_id or "").strip()
    return store.get(named or default_id)


__all__ = [
    "DEFAULT_USER_TEMPLATE",
    "IMAGE_DEFAULT_ID",
    "PromptConflictError",
    "PromptEntry",
    "PromptNotFoundError",
    "PromptStore",
    "PromptStoreError",
    "PromptValidationError",
    "get_prompt_store",
    "normalize_prompt_id",
    "render_template",
    "resolve_structured_prompt",
    "store_for",
]
