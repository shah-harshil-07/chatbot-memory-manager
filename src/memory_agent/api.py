"""HTTP API.  Run:  uvicorn memory_agent.api:app --port 8000   (docs at /docs)

The server is stateless: chat history is sent by the client; long-term memory lives in Qdrant.
Memory writes run as a background task after the reply, so users don't wait for them.
"""
import logging
import secrets
from typing import Annotated, TypeAlias
from contextlib import asynccontextmanager

import openai
from pydantic import BaseModel, Field
from fastapi.responses import JSONResponse
from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Request

from .config import get_settings
from .logging_config import configure_logging
from .container import Container, build_container
from .models import ChatMessage, MaintenanceReport, Memory, MemoryOperation, ScoredMemory

log = logging.getLogger(__name__)


# ------------------------------------------------------------------ schemas
class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    history: list[ChatMessage] = Field(default_factory=list, max_length=100)
    sync_write: bool = False  # True = wait for memory writes and return them (debugging)


class ChatResponse(BaseModel):
    reply: str
    recalled: list[ScoredMemory]
    operations: list[MemoryOperation] | None = None


class RememberRequest(BaseModel):
    text: str = Field(min_length=3, max_length=2000)


# --------------------------------------------------------------------- app
def get_container(request: Request) -> Container:
    return request.app.state.container


C: TypeAlias = Annotated[Container, Depends(get_container)]


def create_app(container: Container | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if container is None:
            settings = get_settings()
            configure_logging(settings.log_level, settings.log_json)
            app.state.container = build_container(settings)
        else:
            app.state.container = container
        yield

    app = FastAPI(title="Memory Agent", version="0.1.0", lifespan=lifespan)

    def require_api_key(request: Request, x_api_key: Annotated[str | None, Header()] = None) -> None:
        expected = request.app.state.container.settings.app_api_key
        if expected and not (x_api_key and secrets.compare_digest(x_api_key, expected.get_secret_value())):
            raise HTTPException(401, "Invalid or missing X-API-Key")

    auth = [Depends(require_api_key)]

    # -------------------------------------------------------------- errors
    @app.exception_handler(openai.RateLimitError)
    async def _rate_limited(_: Request, _exc: openai.RateLimitError):
        log.warning("openai_rate_limited")
        return JSONResponse({"detail": "LLM provider is rate limiting; retry shortly."},
                            status_code=503, headers={"Retry-After": "10"})

    @app.exception_handler(openai.APIError)
    async def _llm_error(_: Request, exc: openai.APIError):
        log.error("openai_error", extra={"ctx": {"error": str(exc)}})
        return JSONResponse({"detail": "LLM provider error."}, status_code=502)

    # ----------------------------------------------------------- endpoints
    @app.get("/health")
    def health(container: C):
        ok = container.store.ping()
        return JSONResponse({"status": "ok" if ok else "degraded", "qdrant": ok},
                            status_code=200 if ok else 503)

    @app.post("/v1/users/{user_id}/chat", response_model=ChatResponse, dependencies=auth)
    def chat(user_id: str, req: ChatRequest, container: C, background: BackgroundTasks):
        reply, recalled = container.agent.respond(user_id, req.message, req.history)
        if req.sync_write:
            return ChatResponse(reply=reply, recalled=recalled,
                                operations=container.agent.write(user_id, req.message, reply))
        background.add_task(_safe_write, container, user_id, req.message, reply)
        return ChatResponse(reply=reply, recalled=recalled)

    @app.get("/v1/users/{user_id}/memories", response_model=list[Memory], dependencies=auth)
    def list_memories(user_id: str, container: C, include_inactive: bool = False):
        return container.store.list(user_id, active_only=not include_inactive)

    @app.post("/v1/users/{user_id}/memories", response_model=list[MemoryOperation], dependencies=auth)
    def remember(user_id: str, req: RememberRequest, container: C):
        return container.manager.remember_text(user_id, req.text)

    @app.delete("/v1/users/{user_id}/memories/{memory_id}", status_code=204, dependencies=auth)
    def forget(user_id: str, memory_id: str, container: C):
        mem = container.store.get(memory_id)
        if mem is None or mem.user_id != user_id:  # never reveal other users' ids
            raise HTTPException(404, "Memory not found")
        container.store.delete([memory_id])

    @app.post("/v1/users/{user_id}/maintenance/consolidate", response_model=MaintenanceReport,
              dependencies=auth)
    def consolidate(user_id: str, container: C, dry_run: bool = True):
        return container.maintenance.consolidate(user_id, dry_run=dry_run)

    @app.post("/v1/users/{user_id}/maintenance/purge", dependencies=auth)
    def purge(user_id: str, container: C, older_than_days: int | None = None):
        return {"deleted": container.maintenance.purge(user_id, older_than_days)}
    return app


def _safe_write(container: Container, user_id: str, message: str, reply: str) -> None:
    try:
        container.agent.write(user_id, message, reply)
    except Exception:  # noqa: BLE001 — a failed memory write must never break chat
        log.exception("background_memory_write_failed", extra={"ctx": {"user_id": user_id}})


app = create_app()
