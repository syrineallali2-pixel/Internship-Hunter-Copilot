"""LIVE smoke test - makes REAL, BILLED calls to Gemini and Tavily.

    python _livesmoke.py            # model + Tavily + one full agent turn
    python _livesmoke.py --model-only   # just verify the key and model id

Unlike the other three suites this one spends money. It is deliberately staged so
the cheapest possible failure happens first: a missing key or a wrong model id
costs nothing, and only then do we run an agent turn.
"""

from __future__ import annotations

import json
import sys
import time

import agent_factory
from agent_factory import (
    get_agent,
    key_status,
    make_config,
    resolve_model_name,
    stream_turn,
    tavily_api_key,
)
from prompts import MODE_LABELS, MODE_PROMPTS
from tools import active_session
from utils import SessionData, capture_answer_side_effects, capture_tool_side_effects, classify_error, extract_text
from langchain.messages import HumanMessage

MODEL_ONLY = "--model-only" in sys.argv


def banner(text: str) -> None:
    print("\n" + "=" * 68)
    print(text)
    print("=" * 68)


def ok(text: str) -> None:
    print(f"  [OK]   {text}")


def bad(text: str) -> None:
    print(f"  [FAIL] {text}")


agent_factory.load_local_env()

# --------------------------------------------------------------------------- #
banner("1. Keys")
status = key_status()
for name, present in status.items():
    (ok if present else bad)(f"{name} {'present' if present else 'MISSING'}")
if not status["GOOGLE_API_KEY"]:
    print("\nGOOGLE_API_KEY is required. Put it in .env next to app.py, then retry.")
    sys.exit(1)
ok(f"Tavily {'available' if status['TAVILY_API_KEY'] else 'absent -> research will degrade gracefully'}")
print("\n  model ids in use:")
for choice in ("Fast", "Deep"):
    print(f"    {choice:<5} -> {resolve_model_name(choice)}")

# --------------------------------------------------------------------------- #
banner("2. Minimal Gemini call (cheapest proof the key + model id work)")
fast_model = resolve_model_name("Fast")
started = time.perf_counter()
try:
    reply = agent_factory.get_model(fast_model, 0.0).invoke(
        [HumanMessage(content="Reply with exactly: PONG")]
    )
    took = time.perf_counter() - started
    text = extract_text(reply.content).strip()
    if text:
        ok(f"{fast_model} replied {text!r} in {took:.1f}s")
    else:
        bad(f"{fast_model} replied with empty content in {took:.1f}s")
except Exception as exc:  # noqa: BLE001
    took = time.perf_counter() - started
    title, hint = classify_error(exc)
    bad(f"{fast_model} FAILED after {took:.1f}s -> {title}")
    print(f"         {hint}")
    print(f"         raw: {str(exc)[:400]}")
    print("\n  If this says the model is not found, your key cannot reach that id.")
    print("  Set GEMINI_FAST_MODEL in .env to a model you have access to.")
    sys.exit(1)

if MODEL_ONLY:
    print("\n--model-only: stopping here.")
    sys.exit(0)

# --------------------------------------------------------------------------- #
banner("3. Minimal Tavily call")
if tavily_api_key():
    from tools import web_search

    started = time.perf_counter()
    res = web_search.invoke({"query": "Cloudflare engineering blog technology stack"})
    took = time.perf_counter() - started
    if res.get("error"):
        bad(f"web_search returned an error after {took:.1f}s -> {res['error'][:200]}")
    else:
        hits = res.get("results", [])
        ok(f"web_search returned {len(hits)} result(s) in {took:.1f}s")
        for h in hits[:3]:
            print(f"         - {h.get('title', '')[:70]}")
else:
    bad("no TAVILY_API_KEY - skipping (the agent will say research is disabled)")

# --------------------------------------------------------------------------- #
banner("4. One full agent turn (Career Coach, tools on)")
CV = """
Jane Doe - Software Engineering student, 2027 graduate.
Built a FastAPI service for a campus rideshare app, deployed on AWS with Docker.
Wrote pytest suites (85% coverage) and used PostgreSQL and Redis.
Led a team of 3 in a hackathon and won 2nd place of 80 teams.
Skills: Python, JavaScript, React, Git, Docker, AWS.
"""
POSTING = """
Software Engineer Intern, Backend
Requirements: 3+ months internship, strong Python, experience with FastAPI or Django,
PostgreSQL, Docker, AWS or GCP, Git, unit testing.
Familiarity with Kubernetes and Kafka is a plus.
Nice to have: machine learning experience, GraphQL.
"""

data = SessionData(thread_id="live-smoke-0001")
data.cv_text = CV.strip()
data.job_text = POSTING.strip()
data.role = "Software Engineer Intern"
data.company = "Cloudflare"

prompt = (
    "I saved my CV in memory. Here is the job posting I want to apply to:\n\n"
    f"--- JOB POSTING ---\n{POSTING.strip()}\n--- END POSTING ---\n\n"
    "Run cv_gap_analyzer on my saved CV against this posting and quote the fit score "
    "and the missing skills. Then do ONE web_search for Cloudflare's engineering blog, "
    "and in 4 short bullet points tell me the single thing I should fix first. "
    "Keep the whole answer under 150 words."
)
print(f"  prompt: {len(prompt)} chars, {len(CV) + len(POSTING)} chars of context")
print(f"  system prompt: {len(MODE_PROMPTS['coach'])} chars ({MODE_LABELS['coach']})")

started = time.perf_counter()
tool_names: list[str] = []
final_text = ""
error: BaseException | None = None
try:
    with active_session(data):
        agent = get_agent(fast_model, "coach", 0.3, True)
        config = make_config(data.thread_id)

        # Mirror app.save_cv_to_memory(): the CV must also land in the *agent's*
        # message memory, or the model has never seen it and correctly asks for it.
        agent.update_state(
            config,
            {
                "messages": [
                    HumanMessage(
                        content=(
                            "Remember my CV for this whole session. It is my only source of truth about my "
                            "experience - never invent experience that is not in it.\n\n"
                            f"--- MY CV ---\n{CV.strip()}\n--- END CV ---"
                        )
                    )
                ]
            },
        )
        print("  CV injected into agent memory (same call the Save button makes)\n")

        for event in stream_turn(agent, HumanMessage(content=prompt), config):
            if event["kind"] == "tool_call":
                tool_names.append(event["name"])
                print(f"  -> tool call   [{event['step']}] {event['name']}")
                if event["name"] == "cv_gap_analyzer":
                    # The tool is deterministic for a *given* input, but the model picks
                    # the input, so print what it actually passed to explain any variance.
                    args = json.loads(event["args_preview"]) if event.get("args_preview") else {}
                    print(
                        f"       cv_text={len(args.get('cv_text', ''))} chars, "
                        f"job_text={len(args.get('job_text', ''))} chars"
                    )
            elif event["kind"] == "tool_result":
                rec = event["record"]
                lat = f"{event['latency_s']:.1f}s" if event["latency_s"] is not None else "?"
                print(f"  <- tool result [{rec.step}] {rec.name} in {lat} ok={rec.ok}")
                # Same hook app.run_agent_turn uses to fill the Job-Fit card, the
                # interview scoreboard and the report from structured tool output.
                capture_tool_side_effects(data, rec.name, event["raw"])
            elif event["kind"] == "final":
                final_text = event["text"]
                capture_answer_side_effects(data, final_text)
except Exception as exc:  # noqa: BLE001
    error = exc
took = time.perf_counter() - started

print(f"\n  wall clock: {took:.1f}s")
if error is not None:
    title, hint = classify_error(error)
    bad(f"the turn raised -> {title}: {hint}")

print(f"\n  tools called: {tool_names or 'NONE'}")
print(f"  sources collected: {len(data.sources)}")
for s in data.sources[:4]:
    print(f"    - {s['url'][:88]}")
print(f"  notes saved: {data.notes}")
print(f"  fit captured: {'yes, score=' + str(data.fit['fit_score']) if data.fit else 'no'}")
print(f"  final answer: {len(final_text)} chars")

print("\n" + "-" * 68)
print(final_text.strip()[:1200] if final_text.strip() else "(no final answer)")
print("-" * 68)

# --------------------------------------------------------------------------- #
banner("5. Memory survived the turn?")
try:
    state = agent.get_state(make_config(data.thread_id))
    msgs = (state.values or {}).get("messages", []) if state.values else []
    print(f"  checkpointed messages: {len(msgs)} -> {[type(m).__name__ for m in msgs]}")
    (ok if len(msgs) >= 4 else bad)("memory is accumulating (hidden CV + user + tool + answer)")
    print("  ask a follow-up in the app and it should remember this exchange")
except Exception as exc:  # noqa: BLE001
    bad(f"could not read checkpoint -> {type(exc).__name__}: {exc}")

print("\nDone. This run made real billable calls.\n")