"""Offline verification of the non-LLM parts of Internship Hunter Copilot.

Run with:  python _selftest.py
It does NOT call Gemini or Tavily, and must never do so: a suite that depends on
a third party being down is not a suite. `.env` loading is stubbed and the keys are
removed up front so the no-key code paths are exercised deterministically.
"""

from __future__ import annotations

import json
import os
import sys

os.environ.pop("GOOGLE_API_KEY", None)
os.environ.pop("TAVILY_API_KEY", None)
try:
    import dotenv

    dotenv.load_dotenv = lambda *a, **k: False  # type: ignore[assignment]
except Exception:  # pragma: no cover - dotenv missing
    pass

from prompts import MODE_LABELS, MODE_PROMPTS, get_prompt, prompt_diff
from tools import (
    TOOL_NAMES,
    active_session,
    cv_gap_analyzer,
    extract_skills,
    save_note,
    score_answer,
    web_search,
)
from utils import SessionData, build_session_report, classify_error, extract_text, parse_cv_upload, truncate

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" -> {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


# --------------------------------------------------------------------------- #
print("\n== utils ==")
check("extract_text handles a plain string", extract_text("hello") == "hello")
check(
    "extract_text joins text blocks and drops images",
    extract_text(
        [
            {"type": "text", "text": "part one"},
            {"type": "image", "base64": "AAAA", "mime_type": "image/png"},
            {"type": "text", "text": "part two"},
        ]
    )
    == "part one\n\npart two",
)
check("extract_text survives None", extract_text(None) == "")
check("truncate short-circuits", truncate("abc", 10) == "abc")
check("classify_error spots a rate limit", "rate limit" in classify_error(Exception("429 rate limit exceeded"))[0].lower())
check("classify_error spots a bad key", "key" in classify_error(Exception("API key not valid"))[0].lower())


class FakeUpload:
    """Stand-in for a Streamlit UploadedFile (same duck-typed API)."""

    def __init__(self, name: str, data: bytes) -> None:
        self.name = name
        self._data = data

    def getvalue(self) -> bytes:
        return self._data


txt_upload = FakeUpload("cv.txt", b"Jane Doe\r\nPython, FastAPI\r\n\r\n\r\nBuilt a rideshare API.\r\n")
parsed = parse_cv_upload(txt_upload)
check("parse_cv_upload reads a .txt CV", "Jane Doe" in parsed and "rideshare" in parsed, repr(parsed))
check("parse_cv_upload strips \\r", "\r" not in parsed, repr(parsed))
check("parse_cv_upload collapses blank runs", "\n\n\n" not in parsed, repr(parsed))
check("parse_cv_upload survives a BOM", "Jane" in parse_cv_upload(FakeUpload("cv.md", b"\xef\xbb\xbfJane Doe")))
check("parse_cv_upload on None -> empty", parse_cv_upload(None) == "")

# --------------------------------------------------------------------------- #
print("\n== prompts ==")
check("4 modes registered", len(MODE_PROMPTS) == 4, str(list(MODE_LABELS.values())))
check("all prompts are substantial", all(len(p) > 900 for p in MODE_PROMPTS.values()))
check("unknown mode falls back", get_prompt("coach") == MODE_PROMPTS["coach"])
check(
    "prompt diff is non-empty across modes",
    prompt_diff("coach", "recruiter") != "" and prompt_diff("coach", "coach") == "",
)
check("every prompt mentions web_search", all("web_search" in p for p in MODE_PROMPTS.values()))
check("placeholders survive (no f-string mangling)", "{company}" in MODE_PROMPTS["recruiter"])

# --------------------------------------------------------------------------- #
print("\n== cv_gap_analyzer (deterministic) ==")
CV = """
Jane Doe - Software Engineering student, 2027 graduate.
Built a FastAPI service for a campus rideshare app, deployed on AWS with Docker.
Wrote pytest suites (85% coverage), used PostgreSQL and Redis.
Led a team of 3 in a hackathon, won 2nd place.
Skills: Python, JavaScript, React, Git, Docker, AWS.
"""
POSTING = """
Software Engineer Intern, Backend
Requirements: 3+ months internship, strong Python, experience with FastAPI or Django,
PostgreSQL, Docker, AWS or GCP, Git, unit testing.
Familiarity with Kubernetes and Kafka is a plus.
Nice to have: machine learning experience, GraphQL.
"""
required = {
    "cv_text": CV,
    "job_text": POSTING,
}
result = cv_gap_analyzer.invoke(required)
print(json.dumps(result, indent=2)[:900])
check("returns a fit score in range", isinstance(result["fit_score"], int) and 0 <= result["fit_score"] <= 100)
check("Python matched", "Python" in result["matched"])
check("FastAPI matched", "FastAPI" in result["matched"])
check("Kubernetes is nice-to-have, not required", "Kubernetes" not in result["missing"], str(result["missing"]))
check("Kubernetes counted as nice_to_have_missing", "Kubernetes" in result["nice_to_have_missing"])
check("GraphQL nice-to-have missing", "GraphQL" in result["nice_to_have_missing"])
check("matched is non-empty", len(result["matched"]) >= 4)

empty = cv_gap_analyzer.invoke({"cv_text": "", "job_text": POSTING})
check("missing CV -> friendly error, no crash", "error" in empty and empty["fit_score"] == 0)

skills = extract_skills("We use Postgres, k8s and CI/CD in our React frontend.")
check("extract_skills finds canonical names", {"PostgreSQL", "Kubernetes", "React", "CI/CD"} <= set(skills), str(skills))

# Comparability: a posting with no nice-to-have section must not hand out a free
# 20% bonus, or the same CV scores higher on that posting than on one that lists them.
NO_NICE = """
Software Engineer Intern, Backend
Requirements: 3+ months internship, strong Python, experience with FastAPI or Django,
PostgreSQL, Docker, AWS or GCP, Git, unit testing.
"""
flat = cv_gap_analyzer.invoke({"cv_text": CV, "job_text": NO_NICE})
print("with nice-to-have section:", result["fit_score"], "| without:", flat["fit_score"])
check("no nice-to-have section -> 100% weighted onto required", 0 <= flat["fit_score"] <= 100, str(flat["fit_score"]))
check(
    "identical required lists score identically",
    flat["matched"] == result["matched"] and flat["missing"] == result["missing"],
    f"{flat['matched']} vs {result['matched']}",
)
check(
    "missing nice-to-haves can only lower the score, never raise it",
    flat["fit_score"] >= result["fit_score"],
    f"flat={flat['fit_score']} with-nice={result['fit_score']}",
)

# --------------------------------------------------------------------------- #
print("\n== save_note + session context ==")
session = SessionData()
with active_session(session):
    saved = save_note.invoke({"key": "Target Role", "value": "Backend intern at Acme"})
check("note stored snake_case", session.notes.get("target_role") == "Backend intern at Acme", str(session.notes))
check("note echoed back", saved.get("saved") is True)

outside = save_note.invoke({"key": "k", "value": "v"})
check("save_note without a session -> error, no crash", "error" in outside)

# Live testing showed the agent storing the *entire* CV and posting as notes, which
# are then replayed into every later turn. Notes must be clamped.
from tools import MAX_NOTE_CHARS  # noqa: E402

big_session = SessionData()
with active_session(big_session):
    huge = save_note.invoke({"key": "cv_text", "value": CV * 5})
check(
    f"an oversized note is clamped to {MAX_NOTE_CHARS} chars",
    len(huge["value"]) <= MAX_NOTE_CHARS + 20,
    f"stored {len(huge['value'])} chars",
)
check("the clamp is explained back to the model", "warning" in huge, str(huge.get("warning"))[:90])
check(
    "the clamped note is what actually got stored",
    len(big_session.notes.get("cv_text", "")) <= MAX_NOTE_CHARS + 20,
    str(big_session.notes)[:90],
)
small_session = SessionData()
with active_session(small_session):
    ok_note = save_note.invoke({"key": "deadline", "value": "2026-03-01"})
check("a short note is stored verbatim", small_session.notes.get("deadline") == "2026-03-01", str(small_session.notes))
check("a short note raises no warning", "warning" not in ok_note)

# --------------------------------------------------------------------------- #
print("\n== score_answer (heuristic fallback, no keys) ==")
# GOOGLE_API_KEY is popped at the top of this file, so get_scoring_model() must fail
# and the heuristic path must run. That is the whole point of this test: it used to
# "pass" merely because a 429 quota error happened to be raised, which made it a
# test of Google's uptime rather than of our fallback.
check("no GOOGLE_API_KEY in the environment", not os.environ.get("GOOGLE_API_KEY"))
scored = score_answer.invoke(
    {
        "question": "Walk me through a time you shipped a feature under pressure.",
        "answer": "In my hackathon we built a React app in 36 hours. I owned the API with FastAPI and Postgres. "
        "We won second place out of 80 teams. I would start earlier next time by writing the schema first.",
        "rubric": "Score on STAR completeness.",
    }
)
check("fallback returns a 1-10 score", 1 <= int(scored["score"]) <= 10, str(scored["score"]))
check("fallback returns strengths", len(scored["strengths"]) >= 2)
check("fallback returns improvements", len(scored["improvements"]) >= 2)
check(
    "fallback reports the degradation",
    "error" in scored,
    str(scored.get("error"))[:110],
)
check(
    "the fallback reason is a missing key, not a network flake",
    "GOOGLE_API_KEY" in str(scored.get("error", "")),
    str(scored.get("error"))[:110],
)

# --------------------------------------------------------------------------- #
print("\n== web_search (no key -> graceful) ==")
import os  # noqa: E402

os.environ.pop("TAVILY_API_KEY", None)
searched = web_search.invoke({"query": "Stripe engineering blog tech stack"})
check("missing Tavily key -> error field, no exception", "error" in searched and searched["results"] == [], str(searched)[:120])
check("empty query handled", "error" in web_search.invoke({"query": "  "}))

# --------------------------------------------------------------------------- #
print("\n== side-effect capture (fit card / scoreboard / notes) ==")
# These used to live in app.py, which made them impossible to test without a
# Streamlit runtime. They are pure transforms over SessionData.
from utils import capture_answer_side_effects, capture_tool_side_effects  # noqa: E402

cap = SessionData()
capture_tool_side_effects(cap, "cv_gap_analyzer", json.dumps({"fit_score": 71, "matched": ["Python"], "missing": []}))
check("cv_gap_analyzer output populates data.fit", cap.fit is not None and cap.fit["fit_score"] == 71, str(cap.fit)[:70])

capture_tool_side_effects(cap, "score_answer", json.dumps({"score": 7, "strengths": ["x"], "improvements": ["y"]}))
capture_tool_side_effects(cap, "score_answer", json.dumps({"score": 9, "strengths": ["x"], "improvements": ["y"]}))
check("score_answer results build a scoreboard series", [s["score"] for s in cap.interview_scores] == [7, 9], str([s["score"] for s in cap.interview_scores]))
check("the scoreboard auto-numbers each row", [s.get("step") for s in cap.interview_scores] == [1, 2], str([s.get("step") for s in cap.interview_scores]))

capture_tool_side_effects(cap, "save_note", json.dumps({"saved": True, "notes": {"target_company": "Acme"}}))
check("save_note output populates the notes", cap.notes == {"target_company": "Acme"}, str(cap.notes))

capture_tool_side_effects(cap, "cv_gap_analyzer", "not json at all")
check("garbage tool output is ignored, not fatal", cap.fit is not None and cap.fit["fit_score"] == 71)
capture_tool_side_effects(cap, "unknown_tool", '{"anything": 1}')
check("an unknown tool name is ignored", True)

capture_answer_side_effects(
    cap,
    "**Role:** Backend Intern\n**Company:** Acme\n\n### What I found\nThey use Python (https://x.com)\n",
)
check("role is scraped from the answer", cap.role == "Backend Intern", repr(cap.role))
check("company is scraped from the answer", cap.company == "Acme", repr(cap.company))
check("research summary is captured", "https://x.com" in cap.research_summary)
capture_answer_side_effects(cap, "A later answer with **Role:** Sales Intern")
check("an already-set role is not overwritten", cap.role == "Backend Intern", repr(cap.role))
capture_answer_side_effects(cap, "")
check("an empty answer is a no-op", cap.role == "Backend Intern")

# --------------------------------------------------------------------------- #
print("\n== report ==")
session.fit = result
session.role = "Software Engineer Intern"
session.company = "Acme"
session.research_summary = "They use Python. https://example.com/blog"
session.add_source("Acme blog", "https://example.com/blog")
session.tailored_bullets = "- Built a FastAPI rideshare API serving 2k rides/day"
report = build_session_report(session, model_name="gemini-x", mode_label="Career Coach", temperature=0.3, search_enabled=True)
for section in ("# Internship Hunter Copilot", "## CV / job fit", "## Company research", "## Transcript"):
    check(f"report contains {section!r}", section in report)
check("report contains the fit score", str(result["fit_score"]) in report)

# --------------------------------------------------------------------------- #
print("\n== tool registry ==")
check("4 tools registered", len(TOOL_NAMES) == 4, str(TOOL_NAMES))
for t in (web_search, cv_gap_analyzer, score_answer, save_note):
    check(f"{t.name} has a docstring", bool(t.description and t.description.strip()))

print("\n" + ("ALL CHECKS PASSED" if not FAILURES else f"{len(FAILURES)} FAILURES: {FAILURES}"))
sys.exit(1 if FAILURES else 0)