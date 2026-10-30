import asyncio
import json
import logging
import math
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app import agent
from app.blog import mcp_healthy, ollama_healthy, resolve_user
from app.config import get_settings
from app.ratelimit import RateLimiter
from app.schemas import ChatRequest, Error, Health

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

settings = get_settings()
limiter = RateLimiter(settings.rate_limit_per_minute)
chat_slots = asyncio.Semaphore(settings.max_concurrent_chats)
bearer = HTTPBearer(auto_error=False)


@asynccontextmanager
async def lifespan(_: FastAPI):
    log.info("Langfuse tracing %s", "enabled" if settings.langfuse_enabled else "disabled (no keys)")
    yield
    if settings.langfuse_enabled:
        from langfuse import get_client

        get_client().flush()


app = FastAPI(title="Inkwell Blog Agent API", version="0.1.0", lifespan=lifespan)


@app.get("/agent/health", response_model=Health, responses={503: {"model": Health}})
async def health():
    ollama, mcp = await asyncio.gather(ollama_healthy(), mcp_healthy())
    body = Health(
        status="ok" if ollama and mcp else "degraded",
        ollama="ok" if ollama else "error",
        mcp="ok" if mcp else "error",
        model=settings.ollama_model,
        langfuse="enabled" if settings.langfuse_enabled else "disabled",
    )
    return JSONResponse(body.model_dump(), status_code=200 if body.status == "ok" else 503)


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post(
    "/agent/v1/chat",
    responses={
        200: {"content": {"text/event-stream": {}}, "description": "Event stream"},
        401: {"model": Error},
        429: {"model": Error},
    },
)
async def chat(body: ChatRequest, request: Request, creds: HTTPAuthorizationCredentials | None = Depends(bearer)):
    # nginx sets X-Real-IP; fall back to the socket peer for direct calls.
    client_ip = request.headers.get("x-real-ip") or (request.client.host if request.client else "unknown")
    if (wait := limiter.check(client_ip)) is not None:
        return JSONResponse(
            {"detail": "Too many questions, please wait a moment"},
            status_code=429,
            headers={"Retry-After": str(math.ceil(wait))},
        )
    token = creds.credentials if creds else None
    user = await resolve_user(token)

    async def events():
        async with chat_slots:
            try:
                async for event, data in agent.stream_chat(body.messages, user, token):
                    yield sse(event, data)
            except Exception:
                log.exception("chat run failed")
                yield sse("error", {"detail": "The assistant failed to answer. Please try again."})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
