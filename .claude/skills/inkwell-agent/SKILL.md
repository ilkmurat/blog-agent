---
name: inkwell-agent
description: Work on the Inkwell blog-agent (LangChain agent on Ollama qwen3 using blog-api MCP tools, SSE chat API, Langfuse tracing). Use when changing the chat assistant's behaviour, prompt, tool access by role, streaming, models, rate limits, or when debugging chat answers or Langfuse traces.
---

# Inkwell blog-agent

FastAPI service behind nginx `/agent/`. Per request: resolve the user (blog-api `/api/v1/auth/me`), load MCP tools
from blog-api `/mcp/` (forwarding the user's token), hide tools by role, run a LangChain `create_agent` on
`ChatOllama`, translate the LangGraph stream into SSE events. See `README.md` for API, models and Langfuse.

## Run and verify
```bash
cd ../blog-app
docker compose up -d --build agent                  # code is baked into the image: rebuild after changes
docker compose exec -T agent pytest -q -p no:warnings
curl -s localhost:8001/agent/health
```
Unit tests use fakes (no LLM). **Behaviour changes must also be checked against the real model**: ask the same
question several times (≥3) per role — visitor, member (register a throwaway user, delete it afterwards), admin
(`murat`/`murat`) — and count grounded, clean answers. Parse the SSE stream (`event:`/`data:` lines). Mind the
rate limit (10/min per IP → 429): wait for the window rather than spoofing `X-Real-IP`. Check traces in Langfuse
(http://localhost:3001; v4 query API: `GET /api/public/v2/observations` with the project keys).

## Rules
- Tool access by name prefix in `visible_tools`: `admin_` → admins, `hr_` → logged-in users, others → everyone.
  The MCP server enforces the same rules; never rely on the prompt for access control.
- New MCP tool in blog-api → add prompt guidance here if needed and a label in blog-app `pages/Chat.tsx`.
- Only the latest question is a chat message; earlier turns go into the system prompt (`format_history`).
- Middleware order matters: `[GroundingCheck(), TextToolCallRepair()]` — repair runs on every raw model reply,
  including GroundingCheck's retry. Nudge texts must not leak into answers (they say "do not mention this check").
- The first model turn of a question is emitted from its final message (it may be retried); later turns stream live.
- HR answers: only from `hr_search` passages, cite *(document, s. page)*, say "not covered" otherwise.
- Never invent URLs; only `/post/<slug>` links from tool results.
- Keep `mcp<2` and `langchain-mcp-adapters` in step (adapters require mcp 1.x).

## Gotchas
- qwen3 thinks by default; `OLLAMA_REASONING=false` (passed as `reasoning=` to `ChatOllama`) keeps answers fast.
- Local models: tool calls as text, answering from memory, CJK drift after tools in multi-turn — handled by the
  guards; re-measure them when changing the model.
- `.env` is read at container creation: `docker compose up -d --force-recreate agent` after editing it.
- Langfuse v4 ("events only"): the old `/api/public/traces` endpoint is gone; SDK v5 sends OpenTelemetry spans.
- In zsh never use `path` as a variable name; don't `source` `langfuse/.env` (it sets `COMPOSE_PROJECT_NAME`).
