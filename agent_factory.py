"""Cached construction of models, agents and the shared checkpointer.

Why this module exists
----------------------
Streamlit re-executes the whole script on **every** interaction. If we rebuilt the
LLM, the tools and (above all) the ``InMemorySaver`` on each rerun, the conversation
memory would be wiped 20 times per minute.

The factory therefore guarantees three things:

1. **One** ``InMemorySaver`` for the whole process (``@st.cache_resource``).
2. One compiled agent per ``(model, mode, temperature)`` combination, kept in a dict
   inside a cached function.
3. Because every agent is compiled with the *same* checkpointer and we always use the
   same ``thread_id``, **switching model or mode mid-conversation keeps the memory**.
   That is the demo point of notebook 1.1 (foundational models) and 1.3 (memory).
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Tuple

import streamlit as st
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver

from prompts import get_prompt
from tools import ALL_TOOLS
from utils import extract_text, truncate

# --------------------------------------------------------------------------- #
# Constants / defaults
# --------------------------------------------------------------------------- #

DEFAULT_FAST_MODEL = "gemini-3.6-flash"
DEFAULT_DEEP_MODEL = "gemini-flash-latest"

#: sidebar choices -> (radio label, env var holding the model id)
MODEL_OPTIONS: Dict[str, Tuple[str, str]] = {
    "Fast": ("GEMINI_FAST_MODEL", DEFAULT_FAST_MODEL),
    "Deep": ("GEMINI_DEEP_MODEL", DEFAULT_DEEP_MODEL),
}


# --------------------------------------------------------------------------- #
# Secrets / env
# --------------------------------------------------------------------------- #


def load_local_env() -> None:
    """Load a local ``.env`` file into ``os.environ`` (no-op on cloud deploys).

    Existing environment variables win, so a Streamlit secret never gets
    overwritten by a stray local file.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - dependency missing
        return
    load_dotenv(override=False)


def get_secret(name: str) -> Optional[str]:
    """Read a key from ``os.environ`` first, then from Streamlit secrets."""
    value = os.environ.get(name)
    if value and value.strip():
        return value.strip()

    try:  # st.secrets raises when no secrets file exists
        value = st.secrets.get(name)  # type: ignore[union-attr]
        if value and str(value).strip() and str(value) != "****":
            return str(value).strip()
    except Exception:  # noqa: BLE001 - no secrets.toml / StreamlitSecretNotFoundError
        return None
    return None


def google_api_key() -> Optional[str]:
    """The Gemini API key or ``None``."""
    return get_secret("GOOGLE_API_KEY")


def tavily_api_key() -> Optional[str]:
    """The Tavily API key or ``None``."""
    return get_secret("TAVILY_API_KEY")


def inject_secrets_into_env() -> None:
    """Copy Streamlit secrets into ``os.environ``.

    Third-party clients (Tavily) read the environment directly, so on Streamlit
    Community Cloud / Spaces we mirror the secrets first.
    """
    for key in ("GOOGLE_API_KEY", "TAVILY_API_KEY"):
        value = get_secret(key)
        if value:
            os.environ[key] = value


def resolve_model_name(choice: str) -> str:
    """Map the sidebar ``Fast``/``Deep`` choice onto a concrete Gemini model id."""
    env_var, fallback = MODEL_OPTIONS.get(choice, MODEL_OPTIONS["Fast"])
    return (os.environ.get(env_var) or fallback).strip() or fallback


def key_status() -> Dict[str, bool]:
    """Which keys are currently available (for the sidebar indicators)."""
    return {"GOOGLE_API_KEY": bool(google_api_key()), "TAVILY_API_KEY": bool(tavily_api_key())}


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #


@st.cache_resource(show_spinner=False)
def get_model(model_name: str, temperature: float = 0.3) -> Any:
    """Build (and cache) a ``ChatGoogleGenerativeAI`` instance.

    ``show_spinner=False`` + a per-model cache means switching models in the sidebar
    does not rebuild the client on every rerun.
    """
    load_local_env()
    key = google_api_key()
    if not key:
        raise RuntimeError("GOOGLE_API_KEY is missing.")
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=key,
        temperature=float(temperature),
        max_output_tokens=8192,
    )


@st.cache_resource(show_spinner=False)
def get_scoring_model() -> Any:
    """The cheap model used for the nested ``score_answer`` structured-output call."""
    return get_model(resolve_model_name("Fast"), 0.2)


# --------------------------------------------------------------------------- #
# Checkpointer - shared by every agent on purpose
# --------------------------------------------------------------------------- #


@st.cache_resource(show_spinner=False)
def get_checkpointer() -> InMemorySaver:
    """The **single** in-memory checkpointer shared by every agent (notebook 1.3).

    Because it is cached as a resource it survives Streamlit reruns. It does *not*
    survive a server restart - swap it for ``SqliteSaver`` / ``PostgresSaver`` for
    persistence (see the README).
    """
    return InMemorySaver()


# --------------------------------------------------------------------------- #
# Agents
# --------------------------------------------------------------------------- #


@st.cache_resource(show_spinner=False)
def _get_agent_cached(model_name: str, mode: str, temperature: float, tools_enabled: bool) -> Any:
    """Compile one agent and memoise it in a process-wide dict.

    ``_AGENT_CACHE`` lives in this module (outside the function body), so the cached
    callable always hits the same dict across reruns while still being keyed by its
    arguments.
    """
    load_local_env()
    inject_secrets_into_env()

    key = (model_name, mode, round(float(temperature), 2), bool(tools_enabled))
    agent = _AGENT_CACHE.get(key)
    if agent is not None:
        return agent

    model = get_model(model_name, temperature)
    agent = create_agent(
        model=model,
        tools=ALL_TOOLS if tools_enabled else [],
        system_prompt=get_prompt(mode),
        checkpointer=get_checkpointer(),
    )
    _AGENT_CACHE[key] = agent
    return agent


#: {(model, mode, temperature, tools_enabled): compiled_agent}
_AGENT_CACHE: Dict[Tuple[str, str, float, bool], Any] = {}


def get_agent(model_name: str, mode: str, temperature: float, tools_enabled: bool = True) -> Any:
    """Public entry point used by ``app.py`` to fetch (or build) the agent."""
    return _get_agent_cached(model_name, mode, float(temperature), bool(tools_enabled))


def make_config(thread_id: str) -> Dict[str, Any]:
    """LangGraph invocation config for one conversation thread."""
    return {"configurable": {"thread_id": thread_id}}


def agent_cache_info() -> Dict[str, Any]:
    """Diagnostics for the glass-box panel."""
    return {"cached_agents": list(_AGENT_CACHE.keys()), "checkpointer": type(get_checkpointer()).__name__}


# --------------------------------------------------------------------------- #
# Streaming turn - kept here (not in app.py) so it is testable without Streamlit
# --------------------------------------------------------------------------- #


def stream_turn(agent: Any, message: Any, config: Dict[str, Any]) -> Any:
    """Stream one agent turn, yielding render-ready events.

    Yields dictionaries instead of touching Streamlit, so ``app.py`` only has to
    paint them and this function can be unit tested with a fake model:

    * ``{"kind": "tool_call",   "record": ToolCallRecord, "args_preview": str}``
    * ``{"kind": "tool_result", "record": ToolCallRecord, "preview": str}``
    * ``{"kind": "final",       "text": str}``

    The final event carries the last plain-text AI message seen while streaming; the
    caller can always fall back to ``agent.get_state(config)`` if it is missing.
    """
    import time  # local import: only needed while streaming

    from utils import ToolCallRecord

    pending: Dict[str, Dict[str, Any]] = {}
    step = 0
    final_text = ""

    for chunk in agent.stream({"messages": [message]}, config, stream_mode="updates"):
        for _node, update in (chunk or {}).items():
            if not isinstance(update, dict):
                continue
            for msg in update.get("messages", []):
                mtype = type(msg).__name__

                if mtype == "AIMessage":
                    # An AI message that only *requests* tools is not the answer yet;
                    # the one without tool calls at the end of the run is.
                    tool_calls = getattr(msg, "tool_calls", None) or []
                    if not tool_calls:
                        text_block = extract_text(getattr(msg, "content", ""))
                        if text_block:
                            final_text = text_block
                        continue
                    for call in tool_calls:
                        step += 1
                        call_id = str(call.get("id"))
                        pending[call_id] = {
                            "name": call.get("name", "?"),
                            "args": call.get("args", {}) or {},
                            "started": time.perf_counter(),
                            "step": step,
                        }
                        yield {
                            "kind": "tool_call",
                            "step": step,
                            "name": str(call.get("name", "?")),
                            "args_preview": json.dumps(call.get("args", {}) or {}, ensure_ascii=False),
                            "record": ToolCallRecord(
                                name=str(call.get("name", "?")),
                                args=dict(call.get("args", {}) or {}),
                                result_preview="running…",
                                step=step,
                            ),
                        }
                    continue

                if mtype == "ToolMessage":
                    call_id = str(getattr(msg, "tool_call_id", "") or "")
                    info = pending.pop(call_id, None)
                    latency = (time.perf_counter() - info["started"]) if info else None
                    name = str(getattr(msg, "name", None) or (info["name"] if info else "?"))
                    raw = extract_text(getattr(msg, "content", ""))
                    yield {
                        "kind": "tool_result",
                        "step": info["step"] if info else step,
                        "name": name,
                        "latency_s": latency,
                        "raw": raw,
                        "record": ToolCallRecord(
                            name=name,
                            args=info["args"] if info else {},
                            result_preview=truncate(raw, 3000),
                            ok=not (name == "web_search" and '"error"' in raw),
                            latency_s=latency,
                            step=info["step"] if info else step,
                        ),
                    }

                else:
                    # Any other node output (e.g. a custom middleware message).
                    text_block = extract_text(getattr(msg, "content", ""))
                    if text_block and type(msg).__name__ != "ToolMessage":
                        final_text = text_block

    yield {"kind": "final", "text": final_text, "steps": step}