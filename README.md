# Inkwell Blog Agent

Chat assistant for the Inkwell blog. A **LangChain** agent running on a local **Ollama** model answers questions
("Design kategorisinde hangi yazılar var?", "En çok hangisi beğenildi?", "Kaç kullanıcı var?") using the blog's
**MCP** tools served by [blog-api](https://github.com/ilkmurat/blog-api) at `/mcp/`. Optional **Langfuse** tracing.
The UI is the `/chat` page of [blog-app](https://github.com/ilkmurat/blog-app).

```
browser /chat ──SSE──► nginx /agent/ ──► blog-agent (FastAPI + LangChain) ──► Ollama (host, qwen2.5:14b)
                                              │                   └──────────► Langfuse (optional)
                                              └── MCP (streamable HTTP) ──► blog-api /mcp/ ──► Postgres, Qdrant
```

## Run
Ollama runs **on the host** (Docker on macOS has no GPU access):

```bash
ollama serve &              # if not already running
ollama pull qwen2.5:14b     # once
cd ../blog-app && docker compose up -d --build   # starts web, api, agent, postgres, qdrant
open http://localhost:3000/chat
```

Health: `curl localhost:8001/agent/health` → `{"status":"ok","ollama":"ok","mcp":"ok",...}`

## API
Contract: [`contract/openapi.yaml`](contract/openapi.yaml).

`POST /agent/v1/chat` with `{"messages": [{"role": "user", "content": "..."}]}` (stateless; send the history,
last message from the user, max 20). Responds with Server-Sent Events: `tool_start`, `tool_end`, `token`, `done`,
`error`. An optional blog bearer token identifies the user; **admins** additionally get `admin_` tools (user details).
Everyone else only sees public tools and aggregate counts — enforced both here and by the MCP server.

## Configuration (`.env`, see `.env.example`)
| Variable | Default |
|---|---|
| `OLLAMA_MODEL` | `qwen2.5:14b` |
| `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL` | unset → tracing disabled |
| `RATE_LIMIT_PER_MINUTE` | `10` per client IP |
| `MAX_CONCURRENT_CHATS` | `2` (others wait) |

With Langfuse keys set, every chat is traced as `blog-chat` with the user (`anonymous` or username) and tags
`blog-chat`, `public`/`admin`: model calls (tokens, latency), tool calls with arguments and results.

## Langfuse (self-hosted)
[`langfuse/`](langfuse/) runs Langfuse v4 locally: the official compose file unchanged
(`docker-compose.yml`) plus `docker-compose.override.yml` (UI on port **3001**, its Postgres not published, since
the blog uses 3000/5432).

```bash
./langfuse/setup.sh                       # once: random secrets in langfuse/.env, keys written to ./.env
docker compose -f langfuse/docker-compose.yml -f langfuse/docker-compose.override.yml --env-file langfuse/.env up -d
cd ../blog-app && docker compose up -d --force-recreate agent   # pick up the keys
open http://localhost:3001                # login printed by setup.sh (admin@inkwell.local)
```
Telemetry to Langfuse is disabled. Both `.env` files are git-ignored. Needs roughly 4–6 GB RAM.

## Reliability notes (qwen2.5:14b on Ollama)
Measured issues with this model and the guards that handle them (`app/agent.py`):
- **Tool calls written as plain JSON text** → `TextToolCallRepair` middleware turns them into real tool calls.
- **Follow-up questions answered from memory (invented posts)** → `GroundingCheck` retries a first answer that used no tool.
- **Drift into Chinese / repetition after tool calls in multi-turn chats** (reproducible with raw Ollama calls) →
  earlier turns are passed as context in the system prompt instead of chat messages (0/8 → 8/8 clean in tests);
  garbled answers are retried and a live stream is cut if it starts to degenerate.

A stronger model (e.g. `qwen3:14b`) can be tried by changing `OLLAMA_MODEL`.

## Tests
```bash
docker compose exec agent pytest -q     # (from ../blog-app) unit tests with a fake agent; no LLM needed
```
