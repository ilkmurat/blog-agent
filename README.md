# Inkwell Blog Agent

AI chat assistant of the **Inkwell** platform (the `/chat` screen of [blog-app](https://github.com/ilkmurat/blog-app)).
A **LangChain** agent on a local **Ollama `qwen3:14b`** model answers questions about the blog and — for logged-in
users — about the company's HR documents, using the **MCP** tools of [blog-api](https://github.com/ilkmurat/blog-api).
Every run can be traced in **Langfuse**.

```
browser /chat ──SSE──► nginx /agent/ ──► blog-agent (FastAPI + LangChain) ──► Ollama on the host (qwen3:14b)
                                              │                       └──────► Langfuse :3001 (optional)
                                              └── MCP (Streamable HTTP) ──► blog-api /mcp/ ──► PostgreSQL, Qdrant
```

## What it can answer
| Asked by | Examples | Tools (from blog-api MCP) |
|---|---|---|
| everyone | "Design kategorisinde hangi yazılar var?", "En çok hangisi beğenildi?", "Kaç kullanıcı var?", "Which posts talk about focus?" | `list_posts`, `top_liked_posts`, `search_posts`, `get_post`, `list_categories`, `list_tags`, `site_stats`, `user_stats` |
| logged-in users | "Yıllık iznim kaç gün?", "Masrafları kaç gün içinde bildirmeliyim?", "Hangi HR dokümanları var?" | + `hr_search`, `hr_list_documents` — answers cite *(document, s. page)*; "not covered" when the documents don't say |
| admins | "Son 30 günde kimler kayıt oldu? E-postalarıyla listele" | + `admin_list_users` |

Visitors asking HR questions are asked to log in. Tool visibility is filtered here **and** enforced by the MCP server.

## Models
| Purpose | Model | Notes |
|---|---|---|
| Chat + tool calling | **`qwen3:14b`** on Ollama (host) | `OLLAMA_REASONING=false` turns qwen3's thinking phase off for faster answers |
| Retrieval (in blog-api) | [intfloat/multilingual-e5-large](https://huggingface.co/intfloat/multilingual-e5-large) for HR, [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) for blog posts | via fastembed ONNX builds from Hugging Face ([Qdrant/multilingual-e5-large-onnx](https://huggingface.co/Qdrant/multilingual-e5-large-onnx), [Qdrant/bge-small-en-v1.5-onnx-Q](https://huggingface.co/Qdrant/bge-small-en-v1.5-onnx-Q)) |

Ollama runs natively on the Mac (Docker on macOS has no GPU/Metal access); the container reaches it at
`host.docker.internal:11434`.

## Run
```bash
ollama serve &                   # keep running
ollama pull qwen3:14b            # once, ~9 GB
cd ../blog-app && docker compose up -d --build     # web, api, agent, postgres, qdrant
curl localhost:8001/agent/health # {"status":"ok","ollama":"ok","mcp":"ok","model":"qwen3:14b","langfuse":"enabled|disabled"}
open http://localhost:3000/chat
```
Switch model: set `OLLAMA_MODEL` in `.env`, then `cd ../blog-app && docker compose up -d --force-recreate agent`.

## API
Contract: [`contract/openapi.yaml`](contract/openapi.yaml)

- `GET /agent/health` — Ollama (model present) and MCP reachability.
- `POST /agent/v1/chat` — body `{"messages": [{"role": "user"|"assistant", "content": "…"}]}` (stateless; max 20
  messages, last from the user, ≤ 4000 chars each). Optional `Authorization: Bearer <blog token>`.
  Response: Server-Sent Events `tool_start {name,args}`, `tool_end {name,ok}`, `token {text}`, `done {}`,
  `error {detail}`. `401` invalid token, `422` invalid body, `429` rate limit (with `Retry-After`).

## Langfuse (observability)
Self-hosted Langfuse v4 lives in [`langfuse/`](langfuse/): the official compose file unchanged plus an override
(UI on **3001**, its Postgres not published) and `setup.sh` (random secrets, project keys, login user).

```bash
./langfuse/setup.sh       # once: writes langfuse/.env and LANGFUSE_* keys into ./.env, prints the login
docker compose -f langfuse/docker-compose.yml -f langfuse/docker-compose.override.yml --env-file langfuse/.env up -d
cd ../blog-app && docker compose up -d --force-recreate agent
open http://localhost:3001  # login: admin@inkwell.local / password in langfuse/.env (LANGFUSE_INIT_USER_PASSWORD)
```
Each chat is a `blog-chat` trace with user (`anonymous` or username), tags (`public`/`admin`), model calls
(tokens, latency) and tool calls with arguments and results (e.g. retrieved HR passages). Telemetry to Langfuse
is disabled; ~4–6 GB RAM. Stop: same compose command with `down`. Langfuse Cloud works too: put its keys and
`LANGFUSE_BASE_URL` into `.env`.

## Configuration (`.env`, see `.env.example`)
| Variable | Default |
|---|---|
| `OLLAMA_MODEL` | `qwen3:14b` |
| `OLLAMA_REASONING` | `false` |
| `OLLAMA_NUM_CTX`, `TEMPERATURE` | `8192`, `0.2` |
| `BLOG_API_URL`, `MCP_URL` | set by compose: `http://api:8000`, `http://api:8000/mcp/` |
| `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL` | unset → tracing disabled |
| `RATE_LIMIT_PER_MINUTE` | `10` per client IP (nginx `X-Real-IP`) |
| `MAX_CONCURRENT_CHATS` | `2` (others wait) |

Port 8001 is bound to `127.0.0.1`; browsers go through nginx.

## Reliability guards (`app/agent.py`)
Measured with local models (qwen2.5:14b, now qwen3:14b) and kept on:
- **`TextToolCallRepair`** — tool calls written as plain JSON text become real tool calls.
- **`GroundingCheck`** — if the first answer to a question used no tool, the model is asked once more; the retry
  is kept only if it calls a tool. Garbled answers (e.g. drift into Chinese) are regenerated.
- **History in the system prompt** — earlier turns are passed as context, not as chat messages (multi-turn chats
  degenerated otherwise: 0/8 → 8/8 clean in tests).
- **Stream guard** — a live answer is cut off if it starts to degenerate.

## Tests
```bash
cd ../blog-app && docker compose exec agent pytest -q   # 31 tests; fake agent/LLM, no Ollama needed
```

## Project layout
```
app/
  main.py        FastAPI app: /agent/health, /agent/v1/chat (SSE), rate limit, concurrency cap
  agent.py       system prompt, tool visibility by role, middleware, LangGraph stream → chat events
  blog.py        resolve the user via blog-api /auth/me, health checks
  config.py, schemas.py, ratelimit.py
contract/        openapi.yaml
langfuse/        self-hosted Langfuse (compose, override, setup.sh)
tests/
```
