"""Adversarial probe: exercise the paths the main suites do not.

Run with: python _deepprobe.py

Goal is to *find bugs*, not to confirm what already works, so most of these drive
degenerate input: empty strings, None, malformed JSON, unicode, very long text,
missing session fields, hostile HTML, and repeated state mutation across reruns.
"""

from __future__ import annotations

import os
import sys

os.environ.pop("GOOGLE_API_KEY", None)
os.environ.pop("TAVILY_API_KEY", None)
try:
    import dotenv

    dotenv.load_dotenv = lambda *a, **k: False  # type: ignore[assignment]
except Exception:  # pragma: no cover
    pass

from streamlit.testing.v1 import AppTest  # noqa: E402

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    safe_name = str(name).encode("ascii", "replace").decode("ascii")
    safe = str(detail).encode("ascii", "replace").decode("ascii")[:220]
    print(f"[{'PASS' if condition else 'FAIL'}] {safe_name}" + (f" -> {safe}" if safe else ""))
    if not condition:
        FAILURES.append(name)


def texts(elements) -> str:
    return " ".join(str(getattr(e, "value", e)) for e in elements)


# --------------------------------------------------------------------------- #
print("\n== 1. utils against degenerate input ==")
import utils  # noqa: E402

for label, value in [
    ("None", None),
    ("empty string", ""),
    ("int", 42),
    ("list of nothing", []),
    ("list of unknown blocks", [{"type": "weird"}, {"no": "type"}]),
    ("text block missing text", [{"type": "text"}]),
    ("non-string text", [{"type": "text", "text": 99}]),
    ("deeply nested", [{"type": "text", "text": [{"type": "text", "text": "x"}]}]),
]:
    try:
        out = utils.extract_text(value)
        check(f"extract_text survives {label}", isinstance(out, str), repr(out)[:100])
    except Exception as exc:  # noqa: BLE001
        check(f"extract_text survives {label}", False, f"{type(exc).__name__}: {exc}")

check("truncate survives None", utils.truncate(None) == "")
check("truncate with limit 0", isinstance(utils.truncate("hello", 0), str))
check("truncate with negative limit", isinstance(utils.truncate("hello", -5), str))

for label, raw in [
    ("None", None),
    ("empty", ""),
    ("bare array", "[]"),
    ("json in prose", 'Here you go: {"a": 1} done'),
    ("truncated json", '{"a": 1,'),
    ("json array not object", "[1, 2, 3]"),
    ("html entities", "&lt;script&gt;"),
    ("null bytes", "a\x00b"),
]:
    try:
        utils.safe_json_loads(raw)
        check(f"safe_json_loads survives {label}", True)
    except Exception as exc:  # noqa: BLE001
        check(f"safe_json_loads survives {label}", False, f"{type(exc).__name__}: {exc}")

for label, exc in [
    ("rate limit", Exception("429 rate limit exceeded")),
    ("bad key", Exception("API key not valid")),
    ("timeout", Exception("ReadTimeout: timed out")),
    ("weird type", Exception({"dict": "exception"})),
    ("empty message", Exception("")),
    ("None message", Exception(None)),
]:
    try:
        title, hint = utils.classify_error(exc)
        check(f"classify_error handles {label}", isinstance(title, str) and isinstance(hint, str), f"{title} | {hint}")
    except Exception as inner:  # noqa: BLE001
        check(f"classify_error handles {label}", False, f"{type(inner).__name__}: {inner}")

# --------------------------------------------------------------------------- #
print("\n== 2. report built from an empty session ==")
try:
    empty_report = utils.build_session_report(
        utils.SessionData(), model_name="m", mode_label="Career Coach", temperature=0.3, search_enabled=False
    )
    check("empty session -> non-empty report", bool(empty_report.strip()))
    check("empty report has the title", "# Internship Hunter Copilot" in empty_report)
    check("empty report has no 'None' leakage", "None" not in empty_report, empty_report[:200])
except Exception as exc:  # noqa: BLE001
    check("build_session_report on empty session", False, f"{type(exc).__name__}: {exc}")

print("\n== 3. report with hostile + unicode content ==")
hostile = utils.SessionData()
hostile.role = "<script>alert(1)</script>"
hostile.company = "Ünïcödé 株式会社 🎯"
hostile.research_summary = "```python\nprint('fence')\n```"
hostile.add_source("evil", "javascript:alert(1)")
try:
    rep = utils.build_session_report(
        hostile, model_name="m", mode_label="Recruiter", temperature=0.0, search_enabled=True
    )
    check("hostile session report builds", bool(rep))
    check("unicode survives the report", "株式会社" in rep)
except Exception as exc:  # noqa: BLE001
    check("hostile session report builds", False, f"{type(exc).__name__}: {exc}")

# --------------------------------------------------------------------------- #
print("\n== 4. cv_gap_analyzer degenerate postings ==")
import tools  # noqa: E402

cases = {
    "empty posting": "",
    "whitespace posting": "   \n\n\t ",
    "posting with no skills": "We are a nice company looking for a go-getter.",
    "posting with only nice-to-have": "Nice to have: Kubernetes, Kafka, GraphQL.",
    "posting with unicode": "Nous recherchons un ingénieur backend 🚀 Python, Django",
    "gigantic posting": ("Python Docker AWS GCP Kubernetes Kafka React Node " * 500),
    "posting with regex metachars": r"C++ .NET C# F# <script> $^&*",
}
for label, posting in cases.items():
    try:
        out = tools.cv_gap_analyzer.invoke({"cv_text": "Python and Docker", "job_text": posting})
        score = out.get("fit_score")
        check(f"cv_gap_analyzer: {label}", isinstance(score, int) and 0 <= score <= 100, f"score={score}")
    except Exception as exc:  # noqa: BLE001
        check(f"cv_gap_analyzer: {label}", False, f"{type(exc).__name__}: {exc}")

try:
    none_args = tools.cv_gap_analyzer.invoke({})
    check(
        "cv_gap_analyzer tolerates omitted args (docstring promises 'leave empty')",
        isinstance(none_args, dict) and "error" in none_args,
        str(none_args)[:120],
    )
except Exception as exc:  # noqa: BLE001
    check("cv_gap_analyzer with no args at all", False, f"{type(exc).__name__}: {exc}")

print("\n== 5. other tools with junk input ==")
for label, payload in [
    ("whitespace query", {"query": "   "}),
    ("empty query", {"query": ""}),
    ("very long query", {"query": "x" * 5000}),
    ("unicode query", {"query": "реальность 🎯"}),
]:
    try:
        tools.web_search.invoke(payload)
        check(f"web_search survives {label}", True)
    except Exception as exc:  # noqa: BLE001
        check(f"web_search survives {label}", False, f"{type(exc).__name__}: {exc}")

try:
    res = tools.save_note.invoke({"key": "", "value": ""})
    check("save_note with empty key/value", isinstance(res, dict), str(res)[:120])
except Exception as exc:  # noqa: BLE001
    check("save_note with empty key/value", False, f"{type(exc).__name__}: {exc}")

try:
    res = tools.score_answer.invoke({"question": "", "answer": "", "rubric": ""})
    check("score_answer with all-empty input", 1 <= int(res.get("score", 0)) <= 10, str(res.get("score")))
except Exception as exc:  # noqa: BLE001
    check("score_answer with all-empty input", False, f"{type(exc).__name__}: {exc}")

# --------------------------------------------------------------------------- #
print("\n== 6. UI: every glass-box tab renders with content ==")
os.environ["GOOGLE_API_KEY"] = "fake-key-for-offline-tests"
at = AppTest.from_file("app.py", default_timeout=90).run()
check("app boots", not at.exception, str(at.exception)[:300])
check("4 tabs", len(at.tabs) == 4, str([t.label for t in at.tabs]))
for i, tab in enumerate(at.tabs):
    try:
        # Empty states use st.info / st.warning, not st.markdown, so count every
        # element type before deciding a tab is blank.
        blob = " ".join(
            str(getattr(e, "value", e))
            for e in list(tab.markdown) + list(tab.info) + list(tab.warning) + list(tab.caption)
        )
        check(f"tab {i} ({tab.label}) renders something", len(blob.strip()) > 15, blob[:70])
    except Exception as exc:  # noqa: BLE001
        check(f"tab {i} renders content", False, f"{type(exc).__name__}: {exc}")

print("\n== 7. UI: sidebar toggles and sliders ==")
try:
    if at.sidebar.toggle:
        t = at.sidebar.toggle[0]
        t.set_value(not t.value).run()
        check("toggling the tools checkbox does not crash", not at.exception, str(at.exception)[:300])
        t.set_value(t.value).run()
        check("toggling back does not crash", not at.exception, str(at.exception)[:300])
    else:
        check("tools toggle exists", False, "no sidebar toggle found")
except Exception as exc:  # noqa: BLE001
    check("tools toggle", False, f"{type(exc).__name__}: {exc}")

try:
    slider = at.sidebar.slider[0]
    for value in (0.0, 1.0, 0.95):
        slider.set_value(value).run()
        check(f"temperature={value} rerun is clean", not at.exception, str(at.exception)[:200])
except Exception as exc:  # noqa: BLE001
    check("temperature slider", False, f"{type(exc).__name__}: {exc}")

print("\n== 8. UI: recall test + save CV buttons ==")
for needle in ("Recall test", "Save CV to memory"):
    try:
        btn = next((b for b in at.sidebar.button if needle in str(b.label)), None)
        if btn is None:
            check(f"{needle} button exists", False)
            continue
        out = btn.click().run()
        check(f"{needle} click does not crash", not out.exception, str(out.exception)[:300])
        if needle == "Recall test":
            check("Recall test says something when there is no CV", texts(out.markdown) or texts(out.info))
    except Exception as exc:  # noqa: BLE001
        check(f"{needle} click", False, f"{type(exc).__name__}: {exc}")

print("\n== 9. UI: CV textarea round-trip and unicode ==")
try:
    ta = next((t for t in at.sidebar.text_area if "CV" in str(t.label)), None)
    if ta is None:
        check("CV textarea exists", False)
    else:
        # Typing must NOT silently reach the agent - that is the documented design,
        # and it must warn the user instead of pretending to be saved.
        ta.set_value("Jane Doe 🚀 Python, FastAPI, ünïcödé").run()
        sess = at.session_state["session"]
        check("typed CV is not auto-injected into session", (sess.cv_text or "") == "", repr(sess.cv_text)[:80])
        check("typed-but-unsaved CV warns the user", bool(at.sidebar.warning), texts(at.sidebar.warning)[:120])

        save_btn = next((b for b in at.sidebar.button if "Save CV" in str(b.label)), None)
        if save_btn is None:
            check("Save CV button exists", False)
        else:
            at = save_btn.click().run()
            saved = at.session_state["session"].cv_text
            check("Save CV moves the draft into session", "FastAPI" in (saved or ""), repr(saved)[:90])
            check("unicode survives the save", "ünïcödé" in (saved or "") and "🚀" in (saved or ""))
            sidebar_warnings = texts(at.sidebar.warning)
            check("CV-unsaved warning clears after saving", "Unsaved CV" not in sidebar_warnings, sidebar_warnings[:100])
            at.run()
            check("saved CV survives a rerun", "FastAPI" in (at.session_state["session"].cv_text or ""))
            ta.set_value("").run()
            check("clearing the draft does not crash", not at.exception, str(at.exception)[:200])
except Exception as exc:  # noqa: BLE001
    check("CV textarea round-trip", False, f"{type(exc).__name__}: {exc}")

print("\n== 9b. UI: hostile web-search results are escaped ==")
try:
    at.session_state["session"].sources = [
        {"title": "<img src=x onerror=alert(1)>", "url": "javascript:alert(2)"},
        {"title": "ok & fine", "url": "https://example.com/a"},
    ]
    out = at.run()
    check("script does not crash on hostile source metadata", not out.exception, str(out.exception)[:300])
    tab_src = next((t for t in out.tabs if "Sources" in str(t.label)), None)
    rendered = " ".join(
        str(getattr(e, "value", "")) for e in list(tab_src.markdown) + list(tab_src.caption)
    ) if tab_src else ""
    check("hostile title is escaped, not emitted raw", "<img src=x onerror" not in rendered, rendered[:150])
    check("legitimate source still renders", "example.com" in rendered, rendered[:150])
except Exception as exc:  # noqa: BLE001
    check("hostile sources", False, f"{type(exc).__name__}: {exc}")

print("\n== 10. UI: pathological chat input ==")
for label, payload in [
    ("very long text", "x" * 20000),
    ("unicode + emoji", "héllo 🎯 こんにちは مرحبا"),
    ("markdown injection", "# heading\n\n```python\nimport os\n```\n\n> quote"),
    ("html injection", "<img src=x onerror=alert(1)>"),
    ("empty string", ""),
]:
    try:
        at.session_state["pending"] = (payload, None, None)
        out = at.run()
        check(f"chat input survives {label}", not out.exception, str(out.exception)[:300])
    except Exception as exc:  # noqa: BLE001
        check(f"chat input survives {label}", False, f"{type(exc).__name__}: {exc}")

print("\n== 11. UI: repeated New session (state must not drift) ==")
try:
    ids = []
    for _ in range(3):
        btn = next(b for b in at.sidebar.button if "New session" in str(b.label))
        at = btn.click().run()
        if at.exception:
            check("repeated New session stays clean", False, str(at.exception)[:300])
            break
        ids.append(at.session_state["session"].thread_id)
    check("3x New session stays clean", not at.exception, str(at.exception)[:200])
    check("every New session gives a fresh thread", len(set(ids)) == len(ids), str([i[:8] for i in ids]))
except Exception as exc:  # noqa: BLE001
    check("repeated New session", False, f"{type(exc).__name__}: {exc}")

print("\n" + ("ALL CHECKS PASSED" if not FAILURES else f"{len(FAILURES)} FAILURES:"))
for f in FAILURES:
    print(f"   - {f}")
sys.exit(1 if FAILURES else 0)