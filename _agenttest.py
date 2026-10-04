"""Integration check for the agent plumbing, with a fake model (no API calls).

Verifies the things the Streamlit rerun model depends on:
  1. two agents (different modes / temperatures) share ONE checkpointer
  2. memory survives switching agent mid-conversation (same thread_id)
  3. ``agent.update_state`` injects the hidden "Remember my CV" message
  4. ``stream_turn`` emits tool_call / tool_result / final events
  5. multimodal content blocks are accepted by the message schema
"""

from __future__ import annotations

import base64
import io
import sys
from typing import Any, List

os_env = None  # noqa: N816
import os  # noqa: E402

os.environ["GOOGLE_API_KEY"] = "fake-key-for-offline-tests"
os.environ.pop("TAVILY_API_KEY", None)

from langchain.agents import create_agent  # noqa: E402
from langchain.messages import HumanMessage  # noqa: E402
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage as LC_HumanMessage  # noqa: E402
from langchain.tools import tool  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

import agent_factory  # noqa: E402
from utils import extract_text  # noqa: E402

FAILURES: List[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"[{'PASS' if condition else 'FAIL'}] {name}" + (f" -> {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


class FakeToolModel(FakeMessagesListChatModel):
    """Minimal fake chat model that supports tool binding."""

    def bind_tools(self, tools: Any, **kwargs: Any) -> "FakeToolModel":  # noqa: D102
        return self


@tool
def remember(fact: str) -> str:
    """Store a fact."""
    return f"stored: {fact}"


def scripted_model(script: List[AIMessage]) -> FakeToolModel:
    return FakeToolModel(responses=iter(script))


# --------------------------------------------------------------------------- #
print("\n== 1/2/3: shared checkpointer + memory across agents ==")
saver = InMemorySaver()
thread = "thread-demo"

agent_a = create_agent(scripted_model([AIMessage(content="hi from mode A")]), tools=[remember], system_prompt="A", checkpointer=saver)
agent_b = create_agent(
    scripted_model([AIMessage(content="hi from mode B")]),
    tools=[remember],
    system_prompt="B",
    checkpointer=saver,
)
config = {"configurable": {"thread_id": thread}}

agent_a.invoke({"messages": [HumanMessage(content="remember: my name is Jane")]}, config)

state_b = agent_b.get_state(config)
msgs_b = state_b.values.get("messages", []) if state_b.values else []
check("agent B can read memory written by agent A", any("my name is Jane" in extract_text(m.content) for m in msgs_b), str([type(m).__name__ for m in msgs_b]))

# Hidden CV injection through update_state (the trick app.py uses)
agent_b.update_state(config, {"messages": [LC_HumanMessage(content="Remember my CV: Jane Doe, Python, FastAPI")]})
state_b2 = agent_b.get_state(config)
check(
    "update_state injects the hidden CV message",
    any("Remember my CV" in extract_text(m.content) for m in (state_b2.values or {}).get("messages", [])),
)

# Fresh thread must be empty but must not raise
fresh = agent_b.get_state({"configurable": {"thread_id": "brand-new"}})
check("fresh thread is empty, no exception", not fresh.values)

# --------------------------------------------------------------------------- #
print("\n== 4: stream_turn events ==")
script = [
    AIMessage(content="", tool_calls=[{"name": "remember", "args": {"fact": "target company = Acme"}, "id": "call-1"}]),
    AIMessage(content="Saved. Ask me anything else."),
]
saver2 = InMemorySaver()
agent_c = create_agent(scripted_model(script), tools=[remember], system_prompt="C", checkpointer=saver2)

events = list(agent_factory.stream_turn(agent_c, LC_HumanMessage(content="save this"), {"configurable": {"thread_id": "t-stream"}}))
kinds = [e["kind"] for e in events]
check("emits tool_call then tool_result then final", kinds == ["tool_call", "tool_result", "final"], str(kinds))

call_event = events[0]
check("tool_call event carries args", call_event["args_preview"].find("Acme") != -1, call_event["args_preview"])
check("tool_call event builds a ToolCallRecord", call_event["record"].name == "remember" and call_event["record"].result_preview == "running…")

result_event = events[1]
check("tool_result carries latency", isinstance(result_event["latency_s"], float))
check("tool_result carries the raw payload", "stored: target company = Acme" in result_event["raw"])
check("final event carries the answer", events[2]["text"] == "Saved. Ask me anything else.")
check("final event counts the steps", events[2]["steps"] == 1)

# --------------------------------------------------------------------------- #
print("\n== 5: multimodal content blocks ==")
png_b64 = base64.b64encode(b"\x89PNG\r\n\x1a\nfake").decode()
wav_bytes = io.BytesIO()
import wave  # noqa: E402

with wave.open(wav_bytes, "wb") as fh:
    fh.setnchannels(1)
    fh.setsampwidth(2)
    fh.setframerate(16000)
    fh.writeframes(b"\x00\x00" * 1600)
wav_b64 = base64.b64encode(wav_bytes.getvalue()).decode()

image_msg = LC_HumanMessage(
    content=[
        {"type": "text", "text": "What is in this screenshot?"},
        {"type": "image", "base64": png_b64, "mime_type": "image/png"},
    ]
)
audio_msg = LC_HumanMessage(content=[{"type": "audio", "base64": wav_b64, "mime_type": "audio/wav"}])
check("image blocks validate", len(image_msg.content) == 2 and image_msg.content[1]["mime_type"] == "image/png")
check("audio block validates", audio_msg.content[0]["type"] == "audio")
check("extract_text keeps only the text block", extract_text(image_msg.content) == "What is in this screenshot?")

# The fake model accepts multimodal content without complaining
saver3 = InMemorySaver()
agent_d = create_agent(scripted_model([AIMessage(content="a screenshot")]), tools=[], system_prompt="D", checkpointer=saver3)
out = agent_d.invoke({"messages": [image_msg]}, {"configurable": {"thread_id": "t-mm"}})
check("multimodal message survives the graph", "a screenshot" in extract_text(out["messages"][-1].content))

# --------------------------------------------------------------------------- #
print("\n== 6: agent_factory helpers ==")
check("make_config nests thread_id", agent_factory.make_config("abc") == {"configurable": {"thread_id": "abc"}})
check("get_secret reads the environment", agent_factory.get_secret("GOOGLE_API_KEY") == "fake-key-for-offline-tests")
check("get_secret returns None when unset", agent_factory.get_secret("DEFINITELY_NOT_SET_1234") is None)
status = agent_factory.key_status()
check("key_status reports google=True tavily=False", status["GOOGLE_API_KEY"] is True and status["TAVILY_API_KEY"] is False, str(status))
check("resolve_model_name falls back to a default", bool(agent_factory.resolve_model_name("Fast")))
check("resolve_model_name honours the env override", bool(agent_factory.resolve_model_name("Deep")))

print("\n" + ("ALL CHECKS PASSED" if not FAILURES else f"{len(FAILURES)} FAILURES: {FAILURES}"))
sys.exit(1 if FAILURES else 0)