"""Calls to blog-api: resolve who is chatting, and health checks."""

from dataclasses import dataclass

import httpx
from fastapi import HTTPException, status

from app.config import get_settings


@dataclass(frozen=True)
class BlogUser:
    username: str
    first_name: str
    is_admin: bool


async def resolve_user(token: str | None) -> BlogUser | None:
    """None for anonymous visitors; 401 if a token was sent but blog-api rejects it."""
    if not token:
        return None
    async with httpx.AsyncClient(base_url=get_settings().blog_api_url, timeout=5) as client:
        try:
            r = await client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
        except httpx.HTTPError:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Blog API is unavailable")
    if r.status_code == 401:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or expired token")
    r.raise_for_status()
    body = r.json()
    return BlogUser(body["username"], body["first_name"], body["is_admin"])


async def mcp_healthy() -> bool:
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.post(
                get_settings().mcp_url,
                json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
                headers={"Accept": "application/json, text/event-stream"},
            )
        return r.status_code == 200 and "tools" in r.json().get("result", {})
    except (httpx.HTTPError, ValueError):
        return False


async def ollama_healthy() -> bool:
    settings = get_settings()
    try:
        async with httpx.AsyncClient(base_url=settings.ollama_base_url, timeout=5) as client:
            r = await client.get("/api/tags")
        return any(m["name"] == settings.ollama_model for m in r.json().get("models", []))
    except (httpx.HTTPError, ValueError, KeyError):
        return False
