"""Utility helpers for Internship Hunter Copilot.

Everything in here is pure Python (no Streamlit, no LangChain agent imports) so
the helpers can be unit tested and reused by the tools, the agent factory and the
UI layer alike.

Main responsibilities
---------------------
* ``extract_text``      -> normalise LangChain message ``content`` (str | list of blocks)
* base64 helpers        -> prepare image / audio / PDF payloads for multimodal Gemini
* CV parsing            -> plain text or PDF (via ``pypdf``)
* session bookkeeping   -> :class:`SessionData`, the per-browser-session state object
* report export         -> Markdown "Session Report" download
"""

from __future__ import annotations

import base64
import binascii
import io
import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable

# --------------------------------------------------------------------------- #
# Text helpers
# --------------------------------------------------------------------------- #

_WHITESPACE_RE = re.compile(r"[ \t\x00]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


def extract_text(content: Any) -> str:
    """Flatten a LangChain message ``content`` value into plain text.

    Gemini (and the v1 content-block standard) may return ``content`` either as a
    plain string or as a list of typed blocks::

        "hello"
        [{"type": "text", "text": "hello"}]
        [{"type": "text", "text": "hi"}, {"type": "image", "base64": "..."}]

    This helper joins every ``text`` block, ignores binary blocks and tolerates
    unknown shapes so the UI never crashes on an unexpected payload.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        # A single block passed directly.
        if "text" in content and isinstance(content["text"], str):
            return content["text"]
        return ""
    if isinstance(content, (list, tuple)):
        parts: list[str] = []
        for block in content:
            if block is None:
                continue
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text)
                elif block.get("type") in {"thinking", "reasoning"}:
                    thought = block.get("thinking") or block.get("text") or ""
                    if thought:
                        parts.append(f"_{str(thought).strip()}_")
            else:  # pragma: no cover - defensive
                parts.append(str(block))
        return "\n\n".join(part.strip() for part in parts if part.strip())
    return str(content)


def clean_text(text: str, *, collapse_newlines: bool = True) -> str:
    """Normalise whitespace coming from PDFs / uploads."""
    if not text:
        return ""
    cleaned = _WHITESPACE_RE.sub(" ", text.replace("\r\n", "\n").replace("\r", "\n"))
    cleaned = "\n".join(line.rstrip() for line in cleaned.split("\n"))
    if collapse_newlines:
        cleaned = _MULTI_NEWLINE_RE.sub("\n\n", cleaned)
    return cleaned.strip()


def truncate(text: str, limit: int = 400, suffix: str = "...") -> str:
    """Shorten ``text`` for glass-box previews."""
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - len(suffix))].rstrip() + suffix


def safe_json_loads(raw: Any) -> Any | None:
    """Parse a tool result that is supposed to be JSON, tolerating extra prose."""
    if raw is None:
        return None
    if isinstance(raw, (dict, list)):
        return raw
    text = str(raw).strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    # Fall back to the first JSON object/array found in the blob.
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    return None


# --------------------------------------------------------------------------- #
# base64 / binary helpers
# --------------------------------------------------------------------------- #

_MIME_BY_EXT = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
    "gif": "image/gif",
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "m4a": "audio/mp4",
    "webm": "audio/webm",
    "ogg": "audio/ogg",
    "flac": "audio/flac",
    "mp4": "video/mp4",
    "pdf": "application/pdf",
    "txt": "text/plain",
    "md": "text/markdown",
}


def guess_mime(name: str | None, default: str = "application/octet-stream") -> str:
    """Best-effort MIME type from a file name."""
    if not name:
        return default
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    return _MIME_BY_EXT.get(ext, default)


def bytes_to_b64(data: bytes) -> str:
    """Base64-encode bytes into an ASCII string."""
    return base64.b64encode(data).decode("utf-8")


def b64_to_bytes(data: str) -> bytes:
    """Decode a base64 string back into bytes (never raises on bad input)."""
    try:
        return base64.b64decode(data, validate=False)
    except (binascii.Error, ValueError):  # pragma: no cover - defensive
        return b""


def file_to_b64(uploaded_file: Any, *, default_mime: str | None = None) -> tuple[str, str]:
    """Convert a Streamlit ``UploadedFile`` into ``(base64, mime_type)``."""
    data = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else bytes(uploaded_file)
    name = getattr(uploaded_file, "name", "") or ""
    mime = default_mime or guess_mime(name)
    return bytes_to_b64(data), mime


def text_to_b64(text: str) -> str:
    """Base64-encode a UTF-8 string."""
    return bytes_to_b64(text.encode("utf-8"))


# --------------------------------------------------------------------------- #
# CV / document parsing
# --------------------------------------------------------------------------- #


def extract_pdf_text(data: bytes, *, max_pages: int = 12) -> str:
    """Extract text from PDF bytes using ``pypdf`` (imported lazily)."""
    try:
        from pypdf import PdfReader
        from pypdf.errors import PdfReadError
    except ImportError as exc:  # pragma: no cover - dependency missing
        raise RuntimeError("pypdf is required to parse PDF CVs. Install it with `pip install pypdf`.") from exc

    try:
        reader = PdfReader(io.BytesIO(data))
    except PdfReadError as exc:
        raise ValueError(f"Could not read that PDF: {exc}") from exc
    except Exception as exc:  # noqa: BLE001 - pypdf raises many shapes
        raise ValueError(f"Could not read that PDF: {exc}") from exc

    pages: list[str] = []
    for index, page in enumerate(reader.pages):
        if index >= max_pages:
            break
        try:
            pages.append(page.extract_text() or "")
        except Exception:  # noqa: BLE001 - a single bad page must not kill the CV
            pages.append("")
    text = clean_text("\n\n".join(pages), collapse_newlines=False)
    if not text.strip():
        raise ValueError(
            "No selectable text found in that PDF (it is probably a scanned image). "
            "Paste the CV text into the sidebar instead."
        )
    return text


def parse_cv_upload(uploaded_file: Any) -> str:
    """Read an uploaded CV (``.txt`` / ``.md`` / ``.pdf``) into plain text."""
    if uploaded_file is None:
        return ""
    name = (getattr(uploaded_file, "name", "") or "").lower()
    data = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else bytes(uploaded_file)
    if name.endswith(".pdf") or data[:5] == b"%PDF-":
        return extract_pdf_text(data)
    # collapse_newlines keeps real paragraph breaks ("\n\n") but squashes the
    # 3+ newline runs Word/PDF export leaves behind, so the CV we send to the
    # model is not padded with hundreds of blank lines.
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return clean_text(data.decode(encoding))
        except UnicodeDecodeError:
            continue
    return clean_text(data.decode("utf-8", errors="replace"))


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #


def new_thread_id() -> str:
    """Unique conversation id for one browser session."""
    return uuid.uuid4().hex


def new_session_data(thread_id: str | None = None) -> "SessionData":
    """Create a fresh, empty :class:`SessionData`."""
    return SessionData(thread_id=thread_id or new_thread_id())


@dataclass
class ToolCallRecord:
    """One entry of the glass-box "Tool calls" timeline."""

    name: str
    args: dict[str, Any] = field(default_factory=dict)
    result_preview: str = ""
    ok: bool = True
    latency_s: float | None = None
    step: int = 0
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).strftime("%H:%M:%S"))


@dataclass
class ChatTurn:
    """One rendered chat bubble."""

    role: str  # "user" | "assistant" | "system"
    content: str = ""
    model: str | None = None
    images: list[str] = field(default_factory=list)  # base64 thumbnails
    audio: str | None = None  # base64 clip
    hidden: bool = False  # injected into memory but not shown in the chat
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%H:%M"))


@dataclass
class SessionData:
    """Everything the UI, the tools and the report need for one browser session.

    A single instance lives in ``st.session_state["session"]`` and is also handed
    to the tools through a :class:`contextvars.ContextVar` so that tool calls can
    record notes, sources and scoreboard entries without importing Streamlit.
    """

    thread_id: str = field(default_factory=new_thread_id)
    created_at: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M"))

    # Inputs the user gave us
    cv_text: str = ""
    cv_name: str = ""
    job_text: str = ""

    # Extracted / user-confirmed facts
    company: str = ""
    role: str = ""
    deadline: str = ""

    # Agent outputs
    fit: dict[str, Any] | None = None
    research_summary: str = ""
    tailored_bullets: str = ""
    interview_scores: list[dict[str, Any]] = field(default_factory=list)

    # Observability (glass box)
    chat: list[ChatTurn] = field(default_factory=list)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    sources: list[dict[str, str]] = field(default_factory=list)
    notes: dict[str, str] = field(default_factory=dict)

    # Which model actually answered, for the caption under each bubble
    last_model_used: str = ""

    # --- helpers ---------------------------------------------------------- #

    def reset_conversation(self, *, keep_cv: bool = True) -> None:
        """Wipe conversation memory but (optionally) keep the stored CV."""
        self.thread_id = new_thread_id()
        self.created_at = datetime.now().strftime("%Y-%m-%d %H:%M")
        self.company = ""
        self.role = ""
        self.deadline = ""
        self.fit = None
        self.research_summary = ""
        self.tailored_bullets = ""
        self.interview_scores = []
        self.chat = []
        self.tool_calls = []
        self.sources = []
        self.notes = {}
        self.last_model_used = ""
        if not keep_cv:
            self.cv_text = ""
            self.cv_name = ""
            self.job_text = ""

    def add_note(self, key: str, value: str) -> None:
        """Store (or overwrite) a note. Keys are normalised to snake_case."""
        clean_key = re.sub(r"[^a-z0-9]+", "_", (key or "").strip().lower()).strip("_") or "note"
        self.notes[clean_key] = (value or "").strip()

    def add_source(self, title: str, url: str) -> bool:
        """Add a source, de-duplicated on URL. Returns True if it was new."""
        url = (url or "").strip()
        if not url:
            return False
        if any(existing["url"] == url for existing in self.sources):
            return False
        self.sources.append({"title": (title or url).strip(), "url": url})
        return True

    def average_score(self) -> float | None:
        scores = [float(item["score"]) for item in self.interview_scores if "score" in item]
        if not scores:
            return None
        return round(sum(scores) / len(scores), 2)


# --------------------------------------------------------------------------- #
# Error classification (friendly messages, never a raw traceback)
# --------------------------------------------------------------------------- #

_ERROR_PATTERNS: list[tuple[tuple[str, ...], str, str]] = [
    (
        ("api key not valid", "invalid api key", "unauthenticated", "permission denied", "401", "403"),
        "Gemini rejected the API key",
        "Check `GOOGLE_API_KEY` in your `.env` file or in the deployment secrets, then restart the app.",
    ),
    (
        ("resource_exhausted", "rate limit", "429", "quota", "too many requests"),
        "Gemini rate limit or quota reached",
        "Wait ~20 seconds and retry, or switch to the other model in the sidebar. Free-tier Gemini keys are capped per minute.",
    ),
    (
        ("not found", "404", "is not supported", "invalid model", "unknown model"),
        "That Gemini model name is not available",
        "Set `GEMINI_FAST_MODEL` / `GEMINI_DEEP_MODEL` in `.env` (or secrets) to a model your key can access, e.g. `gemini-2.5-flash`.",
    ),
    (
        ("tavily",),
        "Web research (Tavily) failed",
        "Check `TAVILY_API_KEY`, then press Recall test / retry. Without search the agent will say it cannot verify company facts.",
    ),
    (
        ("timed out", "timeout", "connection", "temporarily unavailable", "503", "502"),
        "Network hiccup while calling the model",
        "Retry the message. If it keeps happening, try the Fast model.",
    ),
    (
        ("pypdf", "could not read that pdf", "no selectable text"),
        "Could not read that PDF",
        "If the PDF is a scan, paste the CV text into the sidebar box instead.",
    ),
    (
        ("google_api_key is missing", "apikey"),
        "No API key configured",
        "Add `GOOGLE_API_KEY` to `.env` locally, or to the secrets of your deployment.",
    ),
]


def classify_error(exc: BaseException) -> tuple[str, str]:
    """Map an exception onto ``(short_title, actionable_hint)``.

    Used by the UI so a missing key, a rate limit or a Tavily failure reads like a
    sentence a student can act on, instead of a stack trace.
    """
    name = type(exc).__name__.lower()
    message = f"{name}: {exc}".lower()
    for needles, title, hint in _ERROR_PATTERNS:
        if any(needle in message for needle in needles):
            return title, hint
    return "Something went wrong", f"`{type(exc).__name__}: {exc}`"


# --------------------------------------------------------------------------- #
# Side-effect capture
#
# These turn a finished agent turn into structured session state (the Job-Fit card,
# the interview scoreboard, the notes list and the downloadable report). They are
# pure functions over SessionData - no Streamlit - so they live here rather than in
# app.py, which keeps them unit testable and importable from the live smoke test.
# --------------------------------------------------------------------------- #


def capture_tool_side_effects(data: SessionData, name: str, raw: str) -> None:
    """Pull structured payloads out of tool results into the session state."""
    parsed = safe_json_loads(raw)
    if not isinstance(parsed, dict):
        return
    if name == "cv_gap_analyzer" and "fit_score" in parsed:
        data.fit = parsed
    elif name == "score_answer" and "score" in parsed:
        parsed.setdefault("step", len(data.interview_scores) + 1)
        data.interview_scores.append(parsed)
    elif name == "save_note" and parsed.get("notes"):
        data.notes = dict(parsed["notes"])


def looks_like_role_line(line: str) -> bool:
    """Cheap check for a "**Role:** Backend Engineer" style markdown line.

    Deliberately plain string matching; no compiled pattern needed.
    """
    return line.strip().lower().startswith(("**role", "role:", "- **role", "**position"))


def extract_role(line: str) -> str:
    return line.split(":", 1)[-1].replace("*", "").strip()[:120]


def capture_answer_side_effects(data: SessionData, answer: str) -> None:
    """Cheap heuristics that feed the downloadable report.

    We deliberately do **not** call an LLM here: the report is built from what the
    agent already produced plus the structured tool outputs.
    """
    if not answer:
        return
    lowered = answer.lower()

    if data.role == "" and any(word in lowered for word in ("the role", "the position", "role:", "position:")):
        for line in answer.splitlines():
            if looks_like_role_line(line):
                data.role = extract_role(line)
                break
    if data.company == "" and any(word in lowered for word in ("company", "at ")):
        for line in answer.splitlines():
            if line.strip().lower().startswith("**company"):
                data.company = line.split(":", 1)[-1].replace("*", "").strip()
                break

    if "### What I found" in answer or "### Company research" in answer:
        data.research_summary = answer
    elif data.research_summary == "" and any(url in answer for url in ("http://", "https://")):
        data.research_summary = answer
    if any(header in answer for header in ("Tailored bullets", "### Your bullets", "CV bullets")):
        data.tailored_bullets = answer


# --------------------------------------------------------------------------- #
# Report export
# --------------------------------------------------------------------------- #


def _bullets(text: str | None) -> list[str]:
    if not text:
        return []
    return [line.strip() for line in text.splitlines() if line.strip()]


def build_session_report(
    session: SessionData,
    *,
    model_name: str,
    mode_label: str,
    temperature: float,
    search_enabled: bool,
) -> str:
    """Render a Markdown session report for download."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    out: list[str] = [
        "# Internship Hunter Copilot - Session Report",
        "",
        f"*Generated {now} | thread `{session.thread_id}`*",
        "",
        "## Target",
        "",
        f"- **Role:** {session.role or '_not specified_'}",
        f"- **Company:** {session.company or '_not specified_'}",
        f"- **Deadline:** {session.deadline or '_not specified_'}",
        f"- **CV provided:** {session.cv_name or ('yes (pasted text)' if session.cv_text else 'no')}",
        "",
        "## Settings",
        "",
        f"- Model: `{model_name}`",
        f"- Mode: **{mode_label}**",
        f"- Temperature: `{temperature}`",
        f"- Web search: {'enabled' if search_enabled else 'disabled'}",
        "",
    ]

    out += ["## Company research", ""]
    if session.research_summary.strip():
        out += [session.research_summary.strip(), ""]
    else:
        out += ["_No research summary captured yet._", ""]
    if session.sources:
        out += ["### Sources", ""]
        out += [f"- [{s['title']}]({s['url']})" for s in session.sources]
        out += [""]

    out += ["## CV / job fit", ""]
    if session.fit:
        matched = session.fit.get("matched", []) or []
        missing = session.fit.get("missing", []) or []
        nice_missing = session.fit.get("nice_to_have_missing", []) or []
        out += [
            f"**Fit score: {session.fit.get('fit_score', 'n/a')}/100**",
            "",
            f"- Matched ({len(matched)}): {', '.join(matched) if matched else '_none_'}",
            f"- Missing ({len(missing)}): {', '.join(missing) if missing else '_none_'}",
            f"- Nice-to-have missing ({len(nice_missing)}): {', '.join(nice_missing) if nice_missing else '_none_'}",
            "",
        ]
    else:
        out += ["_`cv_gap_analyzer` has not run yet._", ""]

    out += ["## Tailored CV bullets", ""]
    if session.tailored_bullets.strip():
        out += [session.tailored_bullets.strip(), ""]
    else:
        out += ["_No tailored bullets captured yet._", ""]

    out += ["## Mock interview scoreboard", ""]
    if session.interview_scores:
        rows = ["| # | Score /10 | Question |", "| --- | --- | --- |"]
        for index, item in enumerate(session.interview_scores, start=1):
            question = str(item.get("question", "")).replace("|", "/")
            rows.append(f"| {index} | {item.get('score', '?')} | {question[:120]} |")
        out += rows + ["", f"**Average: {session.average_score()}/10**", ""]
        for index, item in enumerate(session.interview_scores, start=1):
            out += [f"### Answer {index} - {item.get('score', '?')}/10", ""]
            out += [f"**Question:** {item.get('question', '')}", ""]
            strengths: Iterable[str] = item.get("strengths", []) or []
            improvements: Iterable[str] = item.get("improvements", []) or []
            if strengths:
                out += ["**Strengths**", ""] + [f"- {s}" for s in strengths] + [""]
            if improvements:
                out += ["**Improvements**", ""] + [f"- {s}" for s in improvements] + [""]
            sample = str(item.get("sample_better_answer", "")).strip()
            if sample:
                out += ["**Model answer**", "", sample, ""]
    else:
        out += ["_No mock interview has been scored yet._", ""]

    out += ["## Notes saved by the agent (`save_note`)", ""]
    if session.notes:
        out += [f"- **{key}:** {value}" for key, value in session.notes.items()]
    else:
        out += ["_No notes saved._"]
    out += [""]

    out += ["## Transcript", ""]
    for turn in session.chat:
        if turn.hidden or not turn.content.strip():
            continue
        speaker = "You" if turn.role == "user" else "Copilot"
        model_tag = f" _{turn.model}_" if turn.model else ""
        out += [f"### {speaker}{model_tag} - {turn.timestamp}", "", turn.content.strip(), ""]

    out += ["---", "", "_Generated by Internship Hunter Copilot. Verify every claim before using it in a real application._"]
    return "\n".join(out)


def report_filename(session: SessionData) -> str:
    """Safe download filename for the session report."""
    parts = [p for p in (session.role, session.company) if p]
    slug = re.sub(r"[^A-Za-z0-9]+", "-", " ".join(parts)).strip("-").lower()
    return f"internship-hunter-report-{slug or session.thread_id[:8]}.md"