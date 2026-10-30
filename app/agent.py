"""LangChain agent over the blog's MCP tools, streamed as (event, data) pairs."""

import json
import logging
import re
import uuid
from collections.abc import AsyncIterator, Sequence
from datetime import date
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama

from app.blog import BlogUser
from app.config import get_settings
from app.schemas import ChatMessage

log = logging.getLogger(__name__)

ADMIN_PREFIX = "admin_"
MEMBER_PREFIX = "hr_"  # internal HR documents: logged-in users only

SYSTEM_PROMPT = """You are the assistant of "Inkwell". You answer questions about (1) the Inkwell blog (web development,
design, productivity, DevOps) and (2) for logged-in users, the company's internal HR documents.
Today is {today}. You are talking to {who}.

Rules:
- Always answer in the same language as the user's last message (Turkish or English).
- For any fact about the blog (posts, categories, tags, likes, users, numbers) call a tool first.
  Never invent posts, titles, links or numbers. If the tools return nothing, say so.
- Earlier answers in this conversation are not a source of truth: call tools again for every new question.
- Link posts with Markdown using the url field, e.g. [Title](/post/slug). Never invent URLs: HR documents have no
  links, so mention them by title only.
- Keep answers short and well formatted; use bullet lists for several items.
- User data: you may share aggregate counts (user_stats, site_stats) with anyone. {user_data_rule}
- HR questions (company policies, leave, working hours, benefits, expenses, procedures): {hr_rule}
- If a question is unrelated to both the blog and HR, answer briefly and steer back.{history}"""

HISTORY_BLOCK = """

Conversation so far (context for follow-up questions only; take facts from tools, not from here):
{turns}"""
HISTORY_TURNS = 6
HISTORY_CHARS = 600


def format_history(messages: Sequence[ChatMessage]) -> str:
    """Earlier turns as plain text. qwen2.5 on Ollama degenerates (e.g. drifts into Chinese) after tool calls when
    earlier turns are sent as chat messages, but stays clean when the same context is part of the system prompt."""
    earlier = list(messages[:-1])[-HISTORY_TURNS:]
    if not earlier:
        return ""
    lines = []
    for m in earlier:
        text = m.content if len(m.content) <= HISTORY_CHARS else m.content[:HISTORY_CHARS] + "…"
        lines.append(f"{'User' if m.role == 'user' else 'Assistant'}: {text}")
    return HISTORY_BLOCK.format(turns="\n".join(lines))


HR_RULE_MEMBER = (
    "call hr_search with the question (keep the user's language) and answer ONLY from the returned passages. "
    "Cite each fact as (Document title, s. page). If the passages do not answer the question, say that the HR "
    "documents do not cover it. Never answer company HR policy from general knowledge. "
    "Use hr_list_documents when asked which HR documents or topics exist."
)
HR_RULE_ANONYMOUS = (
    "HR documents are only available to logged-in users. Do not answer HR policy questions; "
    "ask the visitor to log in or sign up first."
)


def system_prompt(user: BlogUser | None, history: str = "") -> str:
    if user is None:
        who, rule = "an anonymous visitor", "Never reveal names, usernames or emails of users; you have no tool for that."
    elif user.is_admin:
        who = f"{user.first_name or user.username}, a site administrator"
        rule = "This user is an admin: you may use admin_ tools and show user details."
    else:
        who = f"{user.first_name or user.username}, a logged-in member"
        rule = "Never reveal names, usernames or emails of other users; you have no tool for that."
    hr_rule = HR_RULE_ANONYMOUS if user is None else HR_RULE_MEMBER
    return SYSTEM_PROMPT.format(
        today=date.today().isoformat(), who=who, user_data_rule=rule, hr_rule=hr_rule, history=history
    )


def visible_tools(tools: Sequence[BaseTool], user: BlogUser | None) -> list[BaseTool]:
    """Admin tools only for admins, HR tools only for logged-in users (the MCP server enforces this too)."""
    def allowed(name: str) -> bool:
        if name.startswith(ADMIN_PREFIX):
            return user is not None and user.is_admin
        if name.startswith(MEMBER_PREFIX):
            return user is not None
        return True

    return [t for t in tools if allowed(t.name)]


def to_langchain(messages: Sequence[ChatMessage]) -> list[BaseMessage]:
    """Only the latest question is sent as a chat message; earlier turns go to the system prompt."""
    return [HumanMessage(messages[-1].content)]


async def load_tools(token: str | None) -> list[BaseTool]:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    client = MultiServerMCPClient(
        {"blog": {"transport": "streamable_http", "url": get_settings().mcp_url, "headers": headers}}
    )
    return await client.get_tools()


_FENCE = re.compile(r"^(?:```(?:json)?|<tool_call>)\s*|\s*(?:```|</tool_call>)$")


def parse_text_tool_calls(content: str, tool_names: set[str]) -> list[dict[str, Any]]:
    """Tool calls that a model wrote as plain JSON text, e.g. {"name": "site_stats", "arguments": {}}.

    Only exact JSON naming an available tool counts; anything else is a normal answer.
    """
    text = _FENCE.sub("", content.strip()).strip()
    if not text.startswith(("{", "[")):
        return []
    try:
        data = json.loads(text)
    except ValueError:
        return []
    calls = []
    for item in data if isinstance(data, list) else [data]:
        if not isinstance(item, dict) or item.get("name") not in tool_names:
            return []
        args = item.get("arguments", item.get("parameters", {}))
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                return []
        if not isinstance(args, dict):
            return []
        calls.append({"name": item["name"], "args": args, "id": f"call_{uuid.uuid4().hex[:12]}", "type": "tool_call"})
    return calls


class TextToolCallRepair(AgentMiddleware):
    """Local models (e.g. qwen2.5 on Ollama) sometimes emit a tool call as text; turn it into a real call."""

    def _repair(self, request: ModelRequest, response: ModelResponse) -> ModelResponse:
        names = {getattr(t, "name", None) or t.get("name") for t in request.tools}
        result = []
        for message in response.result:
            if isinstance(message, AIMessage) and not message.tool_calls and isinstance(message.content, str):
                if calls := parse_text_tool_calls(message.content, names):
                    log.info("Repaired text tool call(s): %s", [c["name"] for c in calls])
                    message = AIMessage(content="", tool_calls=calls, id=message.id)
            result.append(message)
        return ModelResponse(result=result, structured_response=response.structured_response)

    def wrap_model_call(self, request, handler):
        return self._repair(request, handler(request))

    async def awrap_model_call(self, request, handler):
        return self._repair(request, await handler(request))


_CJK = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]")


def degenerate(text: str, question: str) -> bool:
    """CJK characters in an answer to a non-CJK question: the local model drifted off (a known qwen2.5 failure)."""
    return bool(_CJK.search(text)) and not _CJK.search(question)


GARBLED_NUDGE = (
    "[system check] The previous draft was garbled. Write the answer to my original question again, in the same "
    "language as that question, using the tools. Do not mention this check."
)

GROUNDING_NUDGE = (
    "[system check] No tool was called for my original question. If it concerns the blog (posts, categories, tags, "
    "likes, users, numbers) or HR topics (leave, benefits, expenses, policies, procedures), call the matching tool "
    "now. If no tool applies, reply with just: OK"
)


def _since_last_question(messages: Sequence[BaseMessage]) -> Sequence[BaseMessage]:
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            return messages[i + 1 :]
    return messages


def _question(messages: Sequence[BaseMessage]) -> str:
    return next((str(m.content) for m in reversed(messages) if isinstance(m, HumanMessage)), "")


class GroundingCheck(AgentMiddleware):
    """Retries a model turn once when the answer may be unusable:
    - garbled output (CJK drift): the retried answer replaces it;
    - the first answer to a question used no tool (small models answer follow-ups from memory and invent posts):
      if the retry calls a tool it replaces the answer, otherwise the original answer stands (no tool applied)."""

    def _nudge(self, request: ModelRequest, response: ModelResponse) -> tuple[AIMessage, str] | None:
        answer = next((m for m in reversed(response.result) if isinstance(m, AIMessage)), None)
        if answer is None or answer.tool_calls:
            return None
        if isinstance(answer.content, str) and degenerate(answer.content, _question(request.messages)):
            return answer, GARBLED_NUDGE
        if request.tools and not any(isinstance(m, ToolMessage) for m in _since_last_question(request.messages)):
            return answer, GROUNDING_NUDGE
        return None

    def _retry_request(self, request: ModelRequest, answer: AIMessage, nudge: str) -> ModelRequest:
        return request.override(messages=[*request.messages, answer, HumanMessage(nudge)])

    @staticmethod
    def _pick(original: ModelResponse, retried: ModelResponse, nudge: str) -> ModelResponse:
        if nudge is GARBLED_NUDGE:
            return retried
        calls_tool = any(isinstance(m, AIMessage) and m.tool_calls for m in retried.result)
        return retried if calls_tool else original

    def wrap_model_call(self, request, handler):
        response = handler(request)
        if (retry := self._nudge(request, response)) is not None:
            response = self._pick(response, handler(self._retry_request(request, *retry)), retry[1])
        return response

    async def awrap_model_call(self, request, handler):
        response = await handler(request)
        if (retry := self._nudge(request, response)) is not None:
            log.info("Retrying model turn: %s", "garbled" if retry[1] is GARBLED_NUDGE else "no tool used")
            response = self._pick(response, await handler(self._retry_request(request, *retry)), retry[1])
        return response


def build_agent(tools: Sequence[BaseTool], user: BlogUser | None, history: str = ""):
    settings = get_settings()
    model = ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=settings.temperature,
        num_ctx=settings.ollama_num_ctx,
        reasoning=settings.ollama_reasoning,
    )
    return create_agent(model, tools, system_prompt=system_prompt(user, history), middleware=[GroundingCheck(), TextToolCallRepair()])


def run_config(user: BlogUser | None) -> dict[str, Any]:
    if not get_settings().langfuse_enabled:
        return {}
    from langfuse.langchain import CallbackHandler

    return {
        "run_name": "blog-chat",
        "callbacks": [CallbackHandler()],
        "metadata": {
            "langfuse_user_id": user.username if user else "anonymous",
            "langfuse_tags": ["blog-chat", "admin" if user and user.is_admin else "public"],
        },
    }


async def to_events(
    stream: AsyncIterator[tuple[str, Any]], question: str = ""
) -> AsyncIterator[tuple[str, dict[str, Any]]]:
    """Translate LangGraph (mode, payload) items into chat events.

    - The first model turn of a question is not streamed: middleware may repair or retry it, so its final message
      (from the "updates" stream) is what the user sees.
    - Later turns (after tools ran) stream live, except text that starts like JSON, which is held until the turn
      ends: a tool call written as text is dropped, anything else is released as normal answer text.
      If a live turn starts to degenerate, the rest of that turn is cut off.
    """
    held = ""
    streaming = False  # this model turn already streams plain text
    tools_ran = False
    cut = False  # the live turn degenerated; drop the rest of it
    async for mode, payload in stream:
        if mode == "messages":
            chunk, meta = payload
            if not (isinstance(chunk, AIMessageChunk) and meta.get("langgraph_node") == "model" and chunk.content):
                continue
            if not tools_ran or cut:
                continue
            text = chunk.content if isinstance(chunk.content, str) else json.dumps(chunk.content)
            if degenerate(text, question):
                log.warning("Answer degenerated mid-stream; cutting it off")
                clean = _CJK.split(text, 1)[0]
                if streaming and clean:
                    yield "token", {"text": clean}
                cut, held = True, ""
                continue
            if streaming:
                yield "token", {"text": text}
                continue
            held += text
            start = held.lstrip()
            if start and not start.startswith(("{", "[", "`", "<")):
                streaming = True
                yield "token", {"text": held}
                held = ""
            continue
        for node, update in payload.items():  # "updates": complete node outputs
            for message in (update or {}).get("messages", []):
                if node == "model" and isinstance(message, AIMessage):
                    if message.tool_calls:
                        held = ""
                        for call in message.tool_calls:
                            yield "tool_start", {"name": call["name"], "args": call["args"]}
                    elif not tools_ran:
                        if isinstance(message.content, str) and message.content:
                            yield "token", {"text": message.content}
                    elif held:
                        yield "token", {"text": held}
                    held = ""
                    streaming = cut = False
                elif isinstance(message, ToolMessage):
                    tools_ran = True
                    yield "tool_end", {"name": message.name, "ok": message.status != "error"}
    yield "done", {}


async def stream_chat(
    messages: Sequence[ChatMessage], user: BlogUser | None, token: str | None
) -> AsyncIterator[tuple[str, dict[str, Any]]]:
    tools = visible_tools(await load_tools(token if user else None), user)
    agent = build_agent(tools, user, format_history(messages))
    stream = agent.astream(
        {"messages": to_langchain(messages)},
        config=run_config(user),
        stream_mode=["messages", "updates"],
    )
    async for event in to_events(stream, messages[-1].content):
        yield event
