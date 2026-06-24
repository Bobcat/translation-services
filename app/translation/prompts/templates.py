"""Prompt templates: a domain-agnostic ``{{var}}`` model for translation prompts.

A prompt entry is a ``system`` template plus a ``user`` template. Both carry
``{{variable}}`` placeholders that the pipeline fills at apply-time with the
variables that call has (e.g. ``target_lang``, ``category``, ``source_window``).
A template only references the variables it needs; unreferenced ones are ignored.

The built-in defaults live here in code so a prompt id always resolves even with
an empty user store; the file store (``store.py``) can override any id by saving a
file with the same id.
"""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field


# The structured route's user message is, by default, just the ``###``-joined source
# blocks the pipeline builds — kept identical to the pre-template behaviour. A prompt
# author may frame them, but must keep the response structure intact for alignment.
DEFAULT_USER_TEMPLATE = "{{source_window}}"

# The image translate pipeline resolves to this id when the request names no prompt.
IMAGE_DEFAULT_ID = "translate_image_default"


@dataclass(frozen=True)
class PromptEntry:
    id: str
    system: str
    user: str = DEFAULT_USER_TEMPLATE
    # Free-form labels for filtering the flat list per use case/workflow (e.g. a domain).
    tags: list[str] = field(default_factory=list)
    builtin: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "system": self.system,
            "user": self.user,
            "tags": list(self.tags),
            "builtin": self.builtin,
        }


def render_template(text: str, variables: dict[str, str]) -> str:
    """Substitute ``{{key}}`` placeholders. Unknown placeholders are left untouched;
    variables not referenced by the template are ignored."""
    rendered = str(text or "")
    for key, value in variables.items():
        rendered = rendered.replace("{{" + key + "}}", str(value))
    return rendered


# Mirrors the previous ``_structured_system_prompt`` output, with the runtime values
# ({{category}}, {{target_lang}}) as placeholders instead of f-string interpolation.
BUILTIN_PROMPTS: dict[str, PromptEntry] = {
    IMAGE_DEFAULT_ID: PromptEntry(
        id=IMAGE_DEFAULT_ID,
        system=(
            "You are a translation engine for text on an image.\n"
            "Translate every word of every unit into {{target_lang}}, even words already in another "
            "language. When a line repeats the same message in several languages, translate each "
            "occurrence into {{target_lang}}, even if that produces the same {{target_lang}} word "
            "twice (for example, three words meaning 'welcome' in three languages become the "
            "{{target_lang}} word for welcome, three times). Keep only proper names (places, brands).\n"
            "Preserve the newlines and the '|' delimiters.\n"
            "{{category_instructions}}"
            "Output only the translations."
        ),
        user=(
            "# INPUT CATEGORY\n"
            "Category: **{{category}}**\n"
            "\n"
            "# TRANSLATION UNITS\n"
            "{{source_window}}"
        ),
        tags=["image", "default"],
        builtin=True,
    ),
}
