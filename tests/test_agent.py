import asyncio

from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langchain_core.tools import tool

from app.agent import GARBLED_NUDGE, GROUNDING_NUDGE, GroundingCheck, degenerate, parse_text_tool_calls, to_events

NAMES = {"site_stats", "admin_list_users", "list_posts"}
MODEL = {"langgraph_node": "model"}


def test_parses_plain_json_tool_call():
    [call] = parse_text_tool_calls('{\n  "name": "admin_list_users",\n  "arguments": {}\n}', NAMES)
    assert call["name"] == "admin_list_users" and call["args"] == {} and call["id"].startswith("call_")


def test_parses_fenced_list_and_string_args():
    calls = parse_text_tool_calls(
        '```json\n[{"name": "list_posts", "arguments": "{\\"category\\": \\"design\\"}"}, {"name": "site_stats"}]\n```',
        NAMES,
    )
    assert [(c["name"], c["args"]) for c in calls] == [("list_posts", {"category": "design"}), ("site_stats", {})]
    assert parse_text_tool_calls('<tool_call>{"name": "site_stats", "arguments": {}}</tool_call>', NAMES)


def test_ignores_normal_answers_and_unknown_tools():
    assert parse_text_tool_calls("Toplam 5 yazı var.", NAMES) == []
    assert parse_text_tool_calls('{"name": "drop_tables", "arguments": {}}', NAMES) == []
    assert parse_text_tool_calls('{"name": "site_stats", "arguments": [1]}', NAMES) == []
    assert parse_text_tool_calls('{"title": "a JSON answer"}', NAMES) == []


async def _gen(items):
    for item in items:
        yield item


def events(items, question=""):
    async def collect():
        return [e async for e in to_events(_gen(items), question)]

    return asyncio.run(collect())


def tokens(evts):
    return "".join(d["text"] for e, d in evts if e == "token")


CALL = {"name": "site_stats", "args": {}, "id": "c1", "type": "tool_call"}
TOOL_ROUND = [
    ("updates", {"model": {"messages": [AIMessage("", tool_calls=[CALL])]}}),
    ("updates", {"tools": {"messages": [ToolMessage("{}", tool_call_id="c1", name="site_stats")]}}),
]


def test_answer_after_tools_streams_live():
    evts = events([
        *TOOL_ROUND,
        ("messages", (AIMessageChunk("Mer"), MODEL)),
        ("messages", (AIMessageChunk("haba"), MODEL)),
        ("updates", {"model": {"messages": [AIMessage("Merhaba")]}}),
    ])
    assert evts[2:] == [("token", {"text": "Mer"}), ("token", {"text": "haba"}), ("done", {})]


def test_first_turn_uses_final_message_not_draft_chunks():
    # The draft (invented answer) is replaced by the grounded retry inside the model node.
    evts = events([
        ("messages", (AIMessageChunk("Uydurma yazılar..."), MODEL)),
        ("messages", (AIMessageChunk("Merhaba!"), MODEL)),
        ("updates", {"model": {"messages": [AIMessage("Merhaba!")]}}),
    ])
    assert evts == [("token", {"text": "Merhaba!"}), ("done", {})]


def test_text_tool_call_is_not_shown_to_the_user():
    call = {"name": "site_stats", "args": {}, "id": "c1", "type": "tool_call"}
    evts = events([
        ("messages", (AIMessageChunk('{"name": "site_stats",'), MODEL)),
        ("messages", (AIMessageChunk(' "arguments": {}}'), MODEL)),
        ("updates", {"model": {"messages": [AIMessage("", tool_calls=[call])]}}),  # repaired by middleware
        ("updates", {"tools": {"messages": [ToolMessage("{}", tool_call_id="c1", name="site_stats")]}}),
        ("messages", (AIMessageChunk("5 yazı var."), MODEL)),
        ("updates", {"model": {"messages": [AIMessage("5 yazı var.")]}}),
    ])
    assert [e for e, _ in evts] == ["tool_start", "tool_end", "token", "done"]
    assert tokens(evts) == "5 yazı var."


def test_json_looking_answer_is_released_at_turn_end():
    evts = events([
        *TOOL_ROUND,
        ("messages", (AIMessageChunk('{"note": 1}'), MODEL)),
        ("updates", {"model": {"messages": [AIMessage('{"note": 1}')]}}),
    ])
    assert tokens(evts) == '{"note": 1}'


def test_ignores_chunks_from_other_nodes():
    evts = events([("messages", (AIMessageChunk("x"), {"langgraph_node": "tools"}))])
    assert evts == [("done", {})]


@tool
def site_stats() -> str:
    """stats"""
    return ""


class FakeHandler:
    def __init__(self, *answers):
        self.answers = list(answers)
        self.requests = []

    async def __call__(self, request):
        self.requests.append(request)
        return ModelResponse(result=[self.answers.pop(0)])


def run_check(messages, *answers):
    handler = FakeHandler(*answers)
    request = ModelRequest(model=None, messages=messages, tools=[site_stats])
    response = asyncio.run(GroundingCheck().awrap_model_call(request, handler))
    return response.result[-1], handler.requests


def test_grounding_retries_answer_without_tools():
    grounded = AIMessage("", tool_calls=[CALL])
    final, requests = run_check([HumanMessage("Design yazıları?")], AIMessage("Uydurma"), grounded)
    assert final is grounded and len(requests) == 2
    assert requests[1].messages[-1].content == GROUNDING_NUDGE


def test_grounding_keeps_original_answer_when_no_tool_applies():
    # e.g. a visitor's HR question: "please log in" is right; the retry must not replace it.
    original = AIMessage("Lütfen giriş yapın.")
    final, requests = run_check([HumanMessage("Yıllık iznim kaç gün?")], original, AIMessage("OK"))
    assert final is original and len(requests) == 2


def test_grounding_accepts_tool_calls_and_answers_after_tools():
    call = AIMessage("", tool_calls=[CALL])
    final, requests = run_check([HumanMessage("q")], call)
    assert final is call and len(requests) == 1
    after_tools = [HumanMessage("q"), call, ToolMessage("{}", tool_call_id="c1", name="site_stats")]
    final, requests = run_check(after_tools, AIMessage("5 yazı var"))
    assert final.content == "5 yazı var" and len(requests) == 1


def test_grounding_checks_only_the_latest_question():
    # Tools ran for an earlier question, not for the new follow-up -> still retried.
    history = [HumanMessage("q1"), AIMessage("", tool_calls=[CALL]), ToolMessage("{}", tool_call_id="c1", name="site_stats"),
               AIMessage("a1"), HumanMessage("q2")]
    _, requests = run_check(history, AIMessage("from memory"), AIMessage("", tool_calls=[CALL]))
    assert len(requests) == 2


def test_degenerate_detection():
    assert degenerate("2 beğenil士您的助手", "Design kategorisinde neler var?")
    assert not degenerate("Merhaba, 5 yazı var.", "kaç yazı var?")
    assert not degenerate("有5篇文章", "有多少篇文章?")  # a Chinese question may get a Chinese answer


def test_live_stream_is_cut_when_it_degenerates():
    evts = events([
        *TOOL_ROUND,
        ("messages", (AIMessageChunk("Design: [CSS Grid](/post/x) - 2 beğenil"), MODEL)),
        ("messages", (AIMessageChunk("士您的助手"), MODEL)),
        ("messages", (AIMessageChunk("more garbage"), MODEL)),
        ("updates", {"model": {"messages": [AIMessage("...")]}}),
    ], question="Design kategorisinde neler var?")
    assert tokens(evts) == "Design: [CSS Grid](/post/x) - 2 beğenil"


def test_grounding_retries_garbled_answer_after_tools():
    call = AIMessage("", tool_calls=[CALL])
    msgs = [HumanMessage("Design?"), call, ToolMessage("{}", tool_call_id="c1", name="site_stats")]
    final, requests = run_check(msgs, AIMessage("答案"), AIMessage("Design: 1 yazı"))
    assert final.content == "Design: 1 yazı" and requests[1].messages[-1].content == GARBLED_NUDGE
