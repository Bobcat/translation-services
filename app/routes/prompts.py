"""``/v1/prompts*`` — CRUD over the saved translation system prompts."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from pydantic import BaseModel
from pydantic import Field

from app.core.config import AppSettings
from app.routes.common import error_response as _error
from app.runtime.service import RequestRuntime
from app.translation.prompts import PromptConflictError
from app.translation.prompts import PromptEntry
from app.translation.prompts import PromptNotFoundError
from app.translation.prompts import PromptValidationError
from app.translation.prompts import store_for
from app.translation.prompts.templates import DEFAULT_USER_TEMPLATE


class PromptBody(BaseModel):
    system: str
    user: str = DEFAULT_USER_TEMPLATE
    tags: list[str] = Field(default_factory=list)

    def to_entry(self, prompt_id: str) -> PromptEntry:
        return PromptEntry(
            id=prompt_id,
            system=self.system,
            user=self.user or DEFAULT_USER_TEMPLATE,
            tags=list(self.tags),
        )


class PromptCreateBody(PromptBody):
    id: str


def register(app: FastAPI, *, settings: AppSettings, runtime: RequestRuntime) -> None:
    prompt_store = store_for(settings.service.prompts_root)

    @app.get("/v1/prompts")
    async def list_prompts() -> JSONResponse:
        return JSONResponse(status_code=200, content={"prompts": [e.to_dict() for e in prompt_store.list()]})

    @app.post("/v1/prompts")
    async def create_prompt(body: PromptCreateBody) -> JSONResponse:
        try:
            entry = prompt_store.create(body.to_entry(body.id))
        except (PromptConflictError, PromptValidationError, PromptNotFoundError) as exc:
            return _prompt_error(exc)
        return JSONResponse(status_code=200, content=entry.to_dict())

    @app.get("/v1/prompts/{prompt_id:path}")
    async def get_prompt(prompt_id: str) -> JSONResponse:
        try:
            entry = prompt_store.get(prompt_id)
        except (PromptValidationError, PromptNotFoundError) as exc:
            return _prompt_error(exc)
        return JSONResponse(status_code=200, content=entry.to_dict())

    @app.put("/v1/prompts/{prompt_id:path}")
    async def update_prompt(prompt_id: str, body: PromptBody) -> JSONResponse:
        try:
            entry = prompt_store.update(prompt_id, body.to_entry(prompt_id))
        except (PromptValidationError, PromptNotFoundError) as exc:
            return _prompt_error(exc)
        return JSONResponse(status_code=200, content=entry.to_dict())

    @app.delete("/v1/prompts/{prompt_id:path}")
    async def delete_prompt(prompt_id: str) -> JSONResponse:
        try:
            prompt_store.delete(prompt_id)
        except (PromptValidationError, PromptNotFoundError) as exc:
            return _prompt_error(exc)
        return JSONResponse(status_code=200, content={"id": prompt_id, "deleted": True})


def _prompt_error(exc: Exception) -> JSONResponse:
    if isinstance(exc, PromptNotFoundError):
        return _error(404, code="PROMPT_NOT_FOUND", message=str(exc), retryable=False)
    if isinstance(exc, PromptConflictError):
        return _error(409, code="PROMPT_CONFLICT", message=str(exc), retryable=False)
    return _error(400, code="PROMPT_INVALID", message=str(exc), retryable=False)
