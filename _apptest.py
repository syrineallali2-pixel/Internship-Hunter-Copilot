"""Streamlit AppTest smoke test: the app must render and never crash the script.

Run with: python _apptest.py

This suite is hermetic: it must behave identically whether or not you have a real
`.env` next to app.py. `agent_factory.load_local_env()` calls `dotenv.load_dotenv`
at import time, which would otherwise re-populate os.environ *after* we cleared it
and make the "no API key" cases pass only on machines without a .env. So we neuter
load_dotenv before the app script is ever executed.
"""

from __future__ import annotations

import os
import sys

os.environ.pop("GOOGLE_API_KEY", None)
os.environ.pop("TAVILY_API_KEY", None)

# Neutralise .env loading. `load_local_env` does `from dotenv import load_dotenv`
# *inside* the function, so this attribute lookup happens at call time and the stub
# below is what the app actually gets.
try:
    import dotenv

    dotenv.load_dotenv = lambda *a, **k: False  # type: ignore[assignment]
    _DOTENV_STUBBED = True
except Exception:  # pragma: no cover - dotenv missing
    _DOTENV_STUBBED = False

from streamlit.testing.v1 import AppTest  # noqa: E402

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    safe = str(detail).encode("ascii", "replace").decode("ascii")
    print(f"[{'PASS' if condition else 'FAIL'}] {name}" + (f" -> {safe}" if safe else ""))
    if not condition:
        FAILURES.append(name)


def texts(elements) -> str:
    return " ".join(str(getattr(e, "value", e)) for e in elements)


print("\n== A: no API key -> setup screen, no exception ==")
check("dotenv loading is stubbed (suite is hermetic)", _DOTENV_STUBBED)
at = AppTest.from_file("app.py", default_timeout=90).run()
check("app runs", not at.exception, str(at.exception))
check("shows a warning about the missing key", bool(at.warning) or bool(at.error), texts(list(at.warning) + list(at.error)))

print("\n== B: sidebar + layout with a (fake) key ==")
os.environ["GOOGLE_API_KEY"] = "fake-key-for-offline-tests"
at = AppTest.from_file("app.py", default_timeout=90).run()
check("app runs with a key present", not at.exception, str(at.exception)[:400])

sidebar_md = texts(at.sidebar.markdown)
sidebar_buttons = [str(b.label) for b in at.sidebar.button]
check("sidebar has the mode buttons", any("Career Coach" in label for label in sidebar_buttons), str(sidebar_buttons))
check("sidebar has the model switcher", any("Fast" in label for label in at.sidebar.radio[0].options))
check("sidebar has the temperature slider", bool(at.sidebar.slider), str([s.label for s in at.sidebar.slider]))
check("sidebar has a CV textarea", any("CV text" == str(ta.label) for ta in at.sidebar.text_area))
check("sidebar has a CV uploader", bool(at.sidebar.file_uploader))
check("sidebar has New session", any("New session" in label for label in sidebar_buttons), str(sidebar_buttons))
check("sidebar has Recall test", any("Recall test" in label for label in sidebar_buttons))
check("4 glass-box tabs", len(at.tabs) == 4, str([t.label for t in at.tabs]))

main_buttons = [str(b.label) for b in at.button]
check("5 quick actions", sum(1 for label in main_buttons if label and any(k in label for k in ("Paste job posting", "Analyse", "Research", "mock interview", "Tailor"))) == 5, str(main_buttons))
check("chat input rendered", bool(at.chat_input), str([getattr(c, "placeholder", "") for c in at.chat_input]))
check("glass box shows the system prompt", any("Career Coach" in str(ta.value) for ta in at.text_area), "")

print("\n== C: reruns are clean (the Streamlit trap) ==")
for _ in range(3):
    at.run()
check("3 consecutive reruns do not crash", not at.exception, str(at.exception)[:400])
thread = at.session_state["session"].thread_id
check("thread_id is stable across reruns", at.session_state["session"].thread_id == thread, thread[:8])

print("\n== D: mode buttons switch the mode without breaking ==")
recruiter_btn = next(b for b in at.sidebar.button if "Recruiter" in str(b.label))
at = recruiter_btn.click().run()
check("no exception after mode switch", not at.exception, str(at.exception)[:400])
check("mode changed to recruiter", at.session_state.get("mode") == "recruiter", str(at.session_state.get("mode")))
check("thread_id unchanged by mode switch", at.session_state["session"].thread_id == thread)

print("\n== E: model switch keeps the thread too ==")
at.session_state["model_choice"] = "Deep"
at.run()
check("no exception after model switch", not at.exception, str(at.exception)[:400])
check("thread_id unchanged by model switch", at.session_state["session"].thread_id == thread)

print("\n== F: New session rotates the thread and keeps the CV ==")
at.session_state["session"].cv_text = "Jane Doe, Python, FastAPI"
at.run()
new_btn = next(b for b in at.sidebar.button if "New session" in str(b.label))
at = new_btn.click().run()
rotated = at.session_state["session"].thread_id
check("no exception after New session", not at.exception, str(at.exception)[:400])
check("thread_id rotated", rotated != thread, f"{thread[:8]} -> {rotated[:8]}")
check("CV survived the reset", at.session_state["session"].cv_text.startswith("Jane Doe"))
check("chat history cleared", at.session_state["session"].chat == [])

print("\n== G: a queued turn with a fake key fails gracefully ==")
at.session_state["pending"] = ("Research Stripe engineering blog tech stack", None, None)
at = at.run()
check("script does not crash", not at.exception, str(at.exception)[:400])
msgs = texts(list(at.error) + list(at.warning) + list(at.markdown))
check(
    "friendly error instead of a traceback",
    any(k in msgs for k in ("Gemini", "API key", "GOOGLE_API_KEY", "Tavily", "401", "403", "Unauthenticated")),
    msgs[:200],
)

print("\n== H: multimodal queueing does not crash ==")
at.session_state["pending"] = ("Read this screenshot", [("QUJD", "image/png")], ("REVG", "audio/wav"))
at = at.run()
check("script does not crash on multimodal input", not at.exception, str(at.exception)[:400])

print("\n== I: a tool_result with no latency must not blow up the turn ==")
# Regression: stream_turn can yield latency_s=None when a ToolMessage arrives with no
# matching pending tool_call. Formatting that None as "%.1f" used to raise TypeError
# inside st.status, which surfaced as a bogus "Failed" banner on a good run.
import agent_factory  # noqa: E402
from utils import SessionData  # noqa: E402

class _OrphanToolAgent:
    """Fake agent that emits a ToolMessage with an unmatched tool_call_id."""

    def stream(self, payload, config, stream_mode="updates"):
        from langchain.messages import AIMessage, ToolMessage

        yield {"tools": {"messages": [ToolMessage(content='{"ok": 1}', tool_call_id="ghost", name="web_search")]}}
        yield {"model": {"messages": [AIMessage(content="All done.")]}}

    def get_state(self, config):  # pragma: no cover - unused
        raise AssertionError("should not be reached")


events = list(agent_factory.stream_turn(_OrphanToolAgent(), "hi", {"configurable": {"thread_id": "t"}}))
tool_results = [e for e in events if e["kind"] == "tool_result"]
check("orphan ToolMessage still yields a tool_result event", len(tool_results) == 1, str([e["kind"] for e in events]))
check("its latency is None rather than a crash", tool_results and tool_results[0]["latency_s"] is None)
check("final answer still captured", events[-1]["kind"] == "final" and events[-1]["text"] == "All done.", str(events[-1]))

print("\n== J: chip rendering escapes HTML in skill names ==")
from utils import extract_text  # noqa: E402,F401

sess = SessionData()
sess.fit = {
    "fit_score": 80,
    "matched": ["<img src=x onerror=alert(1)>", "Python"],
    "missing": [],
    "nice_to_have": [],
    "nice_to_have_missing": [],
}
at2 = AppTest.from_file("app.py", default_timeout=90).run()
at2.session_state["session"].fit = sess.fit
at2 = at2.run()
check("script does not crash rendering a hostile skill name", not at2.exception, str(at2.exception)[:400])
raw_html = " ".join(str(getattr(m, "value", "")) for m in at2.markdown)
check("raw <img> tag is not emitted unescaped", "<img src=x onerror" not in raw_html, raw_html[:200])

print("\n" + ("ALL CHECKS PASSED" if not FAILURES else f"{len(FAILURES)} FAILURES: {FAILURES}"))
sys.exit(1 if FAILURES else 0)