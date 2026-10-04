"""The four LangChain tools exposed to Internship Hunter Copilot.

Design notes
------------
* Every tool is decorated with ``@tool`` and carries a **docstring**: that docstring
  *is* the tool schema the LLM reads, so it states what the tool does, when to call
  it and what the arguments mean.
* Tools never raise for ordinary failures (missing key, rate limit, network error).
  They return a structured "error" payload instead, so the agent can degrade
  gracefully and tell the user what happened - notebook 1.2 behaviour.
* ``cv_gap_analyzer`` is **deterministic** (curated skill list + regex), no LLM call.
* ``score_answer`` uses a *nested* structured-output LLM call
  (``with_structured_output`` + Pydantic) and falls back to a heuristic score if the
  model is unavailable.
* Tools reach the current browser session through a
  :class:`contextvars.ContextVar` (:data:`ACTIVE_SESSION`), so no Streamlit import
  and no per-session agent rebuild is required.
"""

from __future__ import annotations

import os
import re
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Dict, Iterator, List, Optional, Tuple

from langchain.tools import tool
from pydantic import BaseModel, Field

from utils import SessionData, truncate

# --------------------------------------------------------------------------- #
# Session context (per browser session, set by app.py right before each run)
# --------------------------------------------------------------------------- #

ACTIVE_SESSION: ContextVar[Optional[SessionData]] = ContextVar("ihc_active_session", default=None)


@contextmanager
def active_session(session: Optional[SessionData]) -> Iterator[None]:
    """Bind ``session`` as the active session for the duration of the block."""
    token = ACTIVE_SESSION.set(session)
    try:
        yield
    finally:
        ACTIVE_SESSION.reset(token)


def get_active_session() -> Optional[SessionData]:
    """Return the active :class:`SessionData` (or ``None`` outside a run)."""
    return ACTIVE_SESSION.get()


# --------------------------------------------------------------------------- #
# Curated skill vocabulary
# --------------------------------------------------------------------------- #

#: Canonical skill -> extra surface forms that count as a match in the CV.
SKILL_ALIASES: Dict[str, List[str]] = {
    "Python": ["python", "python3", "py"],
    "JavaScript": ["javascript", "js", "ecmascript"],
    "TypeScript": ["typescript", "ts"],
    "Java": ["java"],
    "Kotlin": ["kotlin"],
    "Swift": ["swift"],
    "Go": ["golang", "go lang"],
    "Rust": ["rust"],
    "C++": ["c++", "cpp"],
    "C#/.NET": ["c#", ".net", "dotnet", "asp.net"],
    "Scala": ["scala"],
    "Ruby": ["ruby", "rails"],
    "PHP": ["php", "laravel"],
    "R": ["rstudio", " r "],
    "SQL": ["sql", "sql92"],
    "Bash": ["bash", "shell scripting", "zsh"],
    # Frontend
    "React": ["react", "react.js", "reactjs", "next.js", "nextjs"],
    "Next.js": ["next.js", "nextjs"],
    "Vue": ["vue", "vue.js", "nuxt"],
    "Angular": ["angular"],
    "Svelte": ["svelte", "sveltekit"],
    "HTML/CSS": ["html", "css", "css3", "sass", "tailwind"],
    "Redux": ["redux", "zustand"],
    "jQuery": ["jquery"],
    # Backend / frameworks
    "Django": ["django"],
    "Flask": ["flask"],
    "FastAPI": ["fastapi"],
    "Node.js": ["node", "node.js", "nodejs", "express", "express.js"],
    "Express": ["express", "express.js"],
    "Spring Boot": ["spring", "spring boot"],
    "GraphQL": ["graphql", "apollo"],
    "REST APIs": ["rest", "rest api", "restful", "restful api"],
    "gRPC": ["grpc", "protobuf"],
    "Microservices": ["microservice", "microservices"],
    "OAuth": ["oauth", "oauth2"],
    # Data / ML
    "Pandas": ["pandas", "numpy", "dataframe"],
    "NumPy": ["numpy"],
    "scikit-learn": ["scikit-learn", "sklearn", "scikit learn"],
    "PyTorch": ["pytorch", "torch"],
    "TensorFlow": ["tensorflow", "keras"],
    "LangChain": ["langchain", "langgraph", "llamaindex"],
    "LLM/GenAI": ["llm", "llms", "genai", "generative ai", "gpt", "openai", "claude", "gemini"],
    "RAG": ["rag", "retrieval augmented", "retrieval-augmented", "vector search", "embeddings"],
    "Vector databases": ["pinecone", "weaviate", "chromadb", "faiss", "qdrant", "milvus"],
    "Pandas/SQL analytics": ["sql", "analytics", "tableau", "power bi"],
    "Data engineering": ["airflow", "dbt", "spark", "etl", "data pipeline", "data engineering"],
    "ETL": ["etl", "airflow", "dbt"],
    "Apache Spark": ["spark", "pyspark"],
    "Kafka": ["kafka", "event streaming"],
    "Airflow": ["airflow", "dag"],
    "dbt": ["dbt"],
    "Data modeling": ["data model", "data modeling", "star schema", "dimensional modeling"],
    "Statistics": ["statistic", "statistical", "hypothesis testing", "regression", "a/b test", "p-value"],
    "Machine Learning": ["machine learning", "ml model", "scikit", "model training"],
    "Deep Learning": ["deep learning", "cnn", "rnn", "transformer", "neural network"],
    "Computer Vision": ["computer vision", "opencv", "image classification"],
    "NLP": ["nlp", "natural language"],
    "MLOps": ["mlops", "model deployment", "model registry", "feature store"],
    # Cloud / infra
    "AWS": ["aws", "amazon web services", "ec2", "s3", "lambda", "sagemaker"],
    "GCP": ["gcp", "google cloud", "bigquery", "cloud run", "vertex ai"],
    "Azure": ["azure"],
    "Docker": ["docker", "dockerfile", "containerization"],
    "Kubernetes": ["kubernetes", "k8s", "eks", "gke"],
    "Terraform": ["terraform", "infrastructure as code", "iac"],
    "CI/CD": ["ci/cd", "cicd", "github actions", "jenkins", "circleci", "gitlab ci"],
    "Linux": ["linux", "unix", "ubuntu"],
    "Git": ["git", "github", "gitlab", "version control"],
    "Observability": ["prometheus", "grafana", "datadog", "opentelemetry", "logging", "monitoring"],
    "Serverless": ["serverless", "lambda", "cloud functions"],
    # Data stores
    "PostgreSQL": ["postgresql", "postgres", "psql"],
    "MySQL": ["mysql", "mariadb"],
    "MongoDB": ["mongodb", "mongo"],
    "Redis": ["redis"],
    "Elasticsearch": ["elasticsearch", "opensearch"],
    # Practices / soft
    "Agile": ["agile", "scrum", "kanban", "sprint"],
    "Unit testing": ["unit test", "pytest", "jest", "junit", "test coverage", "tdd"],
    "System Design": ["system design", "architecture design", "scalability", "distributed systems"],
    "Distributed Systems": ["distributed system", "consensus", "raft", "sharding"],
    "Concurrency": ["concurrency", "asyncio", "multithreading", "threading", "goroutine"],
    "Performance optimization": ["performance", "optimiz", "latency", "profil"],
    "Security": ["security", "owasp", "xss", "sql injection", "authentication", "encryption"],
    "Accessibility": ["accessibility", "a11y", "wcag", "screen reader"],
    "Technical writing": ["technical writing", "documentation", "readme", "design doc"],
    "Code review": ["code review", "peer review"],
    "Mentoring": ["mentor", "mentoring", "onboard"],
    "Product sense": ["product sense", "user research", "customer feedback", "roadmap"],
    "Communication": ["communication", "presenting", "stakeholder"],
}

_NICE_MARKERS = (
    "nice to have",
    "nice-to-have",
    "preferred",
    "bonus",
    "plus",
    "desirable",
    "a plus",
    "good to have",
    "even better",
    "ideally",
)

_STOPWORDS = {
    "experience",
    "work",
    "team",
    "role",
    "job",
    "company",
    "candidate",
    "skill",
    "skills",
    "year",
    "years",
    "student",
    "studentd",
    "internship",
    "intern",
    "graduate",
    "graduation",
    "degree",
    "university",
    "college",
    "we",
    "our",
    "you",
    "your",
    "the",
    "and",
    "with",
    "for",
    "are",
    "will",
    "have",
    "has",
    "this",
    "that",
    "who",
    "you'll",
    "youre",
    "etc",
    "must",
    "should",
    "including",
    "such",
    "using",
    "use",
    "able",
    "ability",
    "knowledge",
    "understanding",
    "familiar",
    "excellent",
    "strong",
    "good",
    "plus",
    "min",
    "max",
    "etc.",
    "please",
    "apply",
    "benefits",
    "salary",
    "location",
    "remote",
    "hybrid",
    "full-time",
    "part-time",
}

#: Tokens that look like technology names but are usually noise in a posting.
_TECH_NOISE = {"api", "apis", "web", "app", "apps", "software", "development", "engineer", "engineering"}


def _alias_pattern(skill: str, aliases: List[str]) -> re.Pattern[str]:
    parts = {re.escape(skill.lower())} | {re.escape(a.lower()) for a in aliases if a.strip()}
    # \b before a leading non-word char (e.g. ".net", "c++") is wrong, so build it manually.
    return re.compile(r"(?<![\w+#.])(" + "|".join(sorted(parts, key=len, reverse=True)) + r")(?![\w+#])", re.IGNORECASE)


_SKILL_PATTERNS: List[Tuple[str, re.Pattern[str]]] = [
    (skill, _alias_pattern(skill, aliases)) for skill, aliases in SKILL_ALIASES.items()
]

#: every surface form we recognise - used to filter out the "extra token" heuristic
_KNOWN_ALIASES: set[str] = {alias.lower() for aliases in SKILL_ALIASES.values() for alias in aliases} | {
    skill.lower() for skill in SKILL_ALIASES
}

# Generic "unknown tech token" heuristic: Kubernetes-style, dot-versions, or
# title-cased words that look like product names (React, Redis, Snowflake...).
_TOKEN_RE = re.compile(r"\b(?:[A-Z][a-z]+(?:[A-Z][a-z]+)+|[A-Z][a-z]+\d+(?:\.\d+)*\.?[a-z]*|[A-Za-z]+\.js|[A-Za-z]+\d{2,}[a-z]*)\b")


def _split_nice_section(job_text: str) -> Tuple[str, str]:
    """Split the posting into ``(required_text, nice_to_have_text)``.

    Line-based state machine rather than a naive "split at the first marker",
    because postings phrase optional requirements in three ways::

        Nice to have: GraphQL          <- heading, everything after it is optional
        Familiarity with Kubernetes is a plus.   <- inline marker, only that clause
        Bonus points for ...                      <- inline marker

    A "required" heading (``Requirements:``, ``Responsibilities``) switches the
    state machine back.
    """
    required_markers = ("requirement", "responsibilit", "qualification", "you have", "what you'll do", "about the role", "must have")
    nice_parts: List[str] = []
    required_parts: List[str] = []
    mode = "required"

    for raw_line in (job_text or "").splitlines():
        line = raw_line.strip(" \t-*•")
        if not line:
            continue
        low = line.lower()

        if any(marker in low for marker in required_markers):
            mode = "required"
            required_parts.append(line)
            continue

        marker = next((m for m in _NICE_MARKERS if m in low), None)
        if marker:
            # Case-insensitive match so "Nice to have:" finds its lowercase marker.
            match = re.search(re.escape(marker), line, re.IGNORECASE)
            assert match is not None  # marker came from `low`
            before = line[: match.start()].strip(" \t-*•")
            after = line[match.end() :].strip()
            if not before:
                # Heading-like marker ("Nice to have: GraphQL") - the rest is optional.
                mode = "nice"
                if after.strip(" .:-,"):
                    nice_parts.append(after)
            elif not after.strip(" .:-,"):
                # Trailing marker ("... is a plus.") - just this line is optional.
                nice_parts.append(line)
            else:
                # Mid-line marker - only the sentence containing it is optional.
                for sentence in re.split(r"(?<=[.!?;])\s+", line) or [line]:
                    if any(m in sentence.lower() for m in _NICE_MARKERS):
                        nice_parts.append(sentence)
                    else:
                        required_parts.append(sentence)
            continue

        (nice_parts if mode == "nice" else required_parts).append(line)

    return "\n".join(required_parts), "\n".join(nice_parts)


def _title_case(text: str) -> str:
    return " ".join(word if word.isupper() or "-" in word else word.capitalize() for word in text.split())


def extract_skills(text: str) -> List[str]:
    """Return the curated skills mentioned in ``text`` (canonical names)."""
    if not text:
        return []
    found: List[str] = []
    for skill, pattern in _SKILL_PATTERNS:
        if pattern.search(text):
            found.append(skill)
    return found


def _extra_tokens(text: str, known: set[str]) -> List[str]:
    """Surface obvious tech-ish tokens that are not in the curated list."""
    extras: List[str] = []
    seen: set[str] = set()
    for token in _TOKEN_RE.findall(text or ""):
        low = token.lower()
        if low in _STOPWORDS or low in _TECH_NOISE or len(low) < 3:
            continue
        if low in seen:
            continue
        if any(low in known for known in known):
            continue
        seen.add(low)
        extras.append(_title_case(token))
    return extras[:12]


def _has(text: str, pattern: re.Pattern[str]) -> bool:
    return bool(pattern.search(text or ""))


# --------------------------------------------------------------------------- #
# Tool 1 - web_search (notebook 1.2 web search)
# --------------------------------------------------------------------------- #

_TAVILY_CLIENT = None
_TAVILY_KEY_USED: str | None = None


def _get_tavily_client() -> Tuple[Any, Optional[str]]:
    """Lazily build a ``TavilyClient``; returns ``(client, error)``.

    The client is cached in a module global keyed by the API key so repeated runs
    reuse one connection pool.
    """
    global _TAVILY_CLIENT, _TAVILY_KEY_USED
    try:
        from tavily import TavilyClient
    except ImportError:  # pragma: no cover - dependency missing
        return None, "The `tavily-python` package is not installed, so web research is disabled."

    api_key = (os.environ.get("TAVILY_API_KEY") or "").strip()
    if not api_key:
        return None, "TAVILY_API_KEY is not set, so web research is disabled."

    if _TAVILY_CLIENT is not None and _TAVILY_KEY_USED == api_key:
        return _TAVILY_CLIENT, None

    try:
        client = TavilyClient(api_key=api_key)
    except Exception as exc:  # noqa: BLE001
        return None, f"Could not initialise Tavily: {exc}"

    _TAVILY_CLIENT = client
    _TAVILY_KEY_USED = api_key
    return client, None


@tool
def web_search(query: str) -> Dict[str, Any]:
    """Search the public web and return the top results for a query.

    Use this for ANY fact about a company: their tech stack, products, recent news,
    engineering culture and reported interview questions. It is the only tool that
    can introduce outside information, so every claim in your answer must come from
    here and must be cited with the returned URL.

    Args:
        query: One focused search query, e.g. "Stripe engineering blog tech stack"
            or "Stripe software engineer intern interview questions". Make 2-4
            separate calls with different angles rather than one broad query.

    Returns:
        A dict with ``query``, ``results`` (list of ``title``/``url``/``snippet``,
        max 5) and, on failure, an ``error`` string instead of raising.
    """
    cleaned = (query or "").strip()
    if not cleaned:
        return {"query": "", "results": [], "error": "Empty query - ask for a specific search string."}

    session = get_active_session()
    client, error = _get_tavily_client()
    if client is None:
        # Graceful degradation: the agent keeps going and tells the user why.
        return {"query": cleaned, "results": [], "error": error or "Search is unavailable."}

    try:
        raw = client.search(
            cleaned,
            search_depth="advanced",
            max_results=5,
            include_answer=False,
            include_raw_content=False,
            topic="general",
        )
    except Exception as exc:  # noqa: BLE001 - network / quota / auth
        return {"query": cleaned, "results": [], "error": f"Tavily search failed: {exc}"}

    results: List[Dict[str, str]] = []
    for item in (raw or {}).get("results", [])[:5]:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        url = str(item.get("url") or "").strip()
        snippet = str(item.get("content") or item.get("raw_content") or "").strip()
        if not url:
            continue
        results.append({"title": title or url, "url": url, "snippet": truncate(snippet, 600)})
        if session is not None:
            session.add_source(title or url, url)

    if not results:
        return {"query": cleaned, "results": [], "error": "No results returned. Try a differently worded query."}
    return {"query": cleaned, "results": results}


# --------------------------------------------------------------------------- #
# Tool 2 - cv_gap_analyzer (notebook 1.2 tools, deterministic, no LLM)
# --------------------------------------------------------------------------- #


@tool
def cv_gap_analyzer(cv_text: str = "", job_text: str = "") -> Dict[str, Any]:
    """Compare a CV against a job posting and return a numeric fit score.

    Deterministic and free: it does not call an LLM, it extracts skills from the
    posting with a curated vocabulary plus regexes and checks them against the CV.

    Call it whenever you have both a CV and a posting - before tailoring bullets and
    in Recruiter mode for the verdict.

    Args:
        cv_text: The candidate's CV text. Leave empty to reuse the CV the user saved
            earlier in this conversation.
        job_text: The job posting text. Leave empty to reuse the posting from this
            conversation.

    Returns:
        ``{"fit_score": 0-100, "matched": [...], "missing": [...],
        "nice_to_have": [...], "nice_to_have_missing": [...],
        "matched_terms": [...], "total_required": int}``. ``fit_score`` weighs
        required skills at 80% and nice-to-haves at 20%; when the posting lists no
        nice-to-haves the full 100% falls onto required skills, so scores stay
        comparable between postings.
    """
    session = get_active_session()
    if session is not None:
        cv_text = cv_text or session.cv_text or session.notes.get("cv_text", "")
        job_text = (
            job_text
            or session.job_text
            or session.notes.get("job_posting_text", "")
            or session.notes.get("job_posting", "")
            or session.notes.get("job_text", "")
        )

    cv_text = (cv_text or "").strip()
    job_text = (job_text or "").strip()

    if not cv_text:
        return {"fit_score": 0, "matched": [], "missing": [], "nice_to_have": [], "nice_to_have_missing": [], "error": "No CV text available. Ask the user to paste or upload their CV first."}
    if not job_text:
        return {"fit_score": 0, "matched": [], "missing": [], "nice_to_have": [], "nice_to_have_missing": [], "error": "No job posting available. Ask the user to paste it first."}

    required_text, nice_text = _split_nice_section(job_text)

    # A skill is *required* when it appears in the main body of the posting (i.e.
    # before the first "nice to have" marker) or in both sections; otherwise it is
    # a nice-to-have.
    main_skills = set(extract_skills(required_text)) | set(_extra_tokens(required_text, _KNOWN_ALIASES))
    nice_skills = set(extract_skills(nice_text)) | set(_extra_tokens(nice_text, _KNOWN_ALIASES))
    # A skill named in the main body is required. Naming it in both sections does not
    # make it "nicer" - it is still required, so only the main body decides.
    required = main_skills
    nice_only = nice_skills - required

    required_sorted = sorted(required)
    nice_sorted = sorted(nice_only)

    matched = [skill for skill in required_sorted if _has(cv_text, _pattern_for(skill))]
    missing = [skill for skill in required_sorted if skill not in matched]
    nice_matched = [skill for skill in nice_sorted if _has(cv_text, _pattern_for(skill))]
    nice_missing = [skill for skill in nice_sorted if skill not in nice_matched]

    # Extra literal keywords from the posting that the CV does mention (ATS hygiene).
    literal_terms = sorted(
        {
            t
            for t in re.findall(r"[A-Za-z][A-Za-z+#.\-/]{2,}", required_text)
            if t.lower() in cv_text.lower() and t.lower() not in _STOPWORDS and t.lower() not in _TECH_NOISE
        }
    )[:20]

    req_total = len(required_sorted)
    req_ratio = (len(matched) / req_total) if req_total else 0.0

    if req_total == 0:
        # Nothing extractable in the posting - do not invent a score.
        fit_score = 0
    elif not nice_sorted:
        # The posting lists no nice-to-haves. Handing out the full 20% bonus here would
        # make the same CV score ~20 points higher on this posting than on one that does
        # list them, for no real reason. Fold the whole 100% onto required skills so two
        # postings are comparable.
        fit_score = int(round(100 * req_ratio))
    else:
        nice_ratio = len(nice_matched) / len(nice_sorted)
        fit_score = int(round(100 * (0.8 * req_ratio + 0.2 * nice_ratio)))

    fit_score = max(0, min(100, fit_score))

    return {
        "fit_score": fit_score,
        "matched": matched,
        "missing": missing,
        "nice_to_have": nice_matched,
        "nice_to_have_missing": nice_missing,
        "matched_terms": literal_terms,
        "total_required": req_total,
        "required_skills": required_sorted,
    }


def _pattern_for(skill: str) -> re.Pattern[str]:
    """Match pattern for a canonical skill (curated or an extracted token)."""
    for canonical, pattern in _SKILL_PATTERNS:
        if canonical == skill:
            return pattern
    return re.compile(r"(?<![\w+#.])" + re.escape(skill.lower()) + r"(?![\w+#])", re.IGNORECASE)


# --------------------------------------------------------------------------- #
# Tool 3 - score_answer (notebook 1.2 tools, nested structured-output call)
# --------------------------------------------------------------------------- #


class AnswerScore(BaseModel):
    """Structured-output schema for :func:`score_answer` (Gemini returns this shape)."""

    score: int = Field(description="Overall score from 1 (very weak) to 10 (interview-ready).")
    strengths: List[str] = Field(description="2-4 specific things the candidate did well.")
    improvements: List[str] = Field(description="2-4 concrete, actionable improvements.")
    sample_better_answer: str = Field(description="A rewritten version of the candidate's own answer that scores higher. Never invent achievements.")
    feedback_summary: str = Field(description="One sentence verdict for the candidate.")


def _heuristic_score(question: str, answer: str) -> Dict[str, Any]:
    """Deterministic fallback scorer used when the scoring model is unavailable."""
    words = len((answer or "").split())
    sentences = [s for s in re.split(r"[.!?]+", answer or "") if s.strip()]
    has_numbers = bool(re.search(r"\b\d[\d,.%]*\b", answer or ""))
    star_words = ("situation", "task", "action", "result", "because", "so that", "we", "i decided")

    score = 3
    score += 2 if words >= 60 else 1 if words >= 25 else 0
    score += 1 if words >= 150 else 0
    score += 1 if len(sentences) >= 4 else 0
    score += 1 if has_numbers else 0
    score += 1 if sum(1 for w in star_words if w in (answer or "").lower()) >= 3 else 0
    score = max(1, min(10, score))

    return {
        "score": score,
        "strengths": [
            f"Answer is {words} words across {len(sentences)} sentences.",
            "Includes concrete numbers." if has_numbers else "No measurable outcome mentioned - add one.",
        ],
        "improvements": [
            "State your personal contribution explicitly (use 'I', not 'we').",
            "Close with the result and what you would do differently next time.",
            "Answer the question asked, in its first two sentences.",
        ],
        "sample_better_answer": (
            "(Heuristic mode - the scoring model was unavailable. "
            "Re-run the interview with a valid GOOGLE_API_KEY for a written model answer.)"
        ),
    }


@tool
def score_answer(question: str, answer: str, rubric: str) -> Dict[str, Any]:
    """Score a candidate's interview answer out of 10 against a rubric.

    Uses a nested structured-output LLM call (Pydantic schema) with the fast model,
    so you get consistent fields every time. In interview modes you MUST call this
    with the previous question and the candidate's answer before asking the next
    question, then relay the score, strengths, improvements and the sample answer.

    Args:
        question: The exact question that was asked.
        answer: The candidate's full answer, verbatim.
        rubric: What "good" means for this question, e.g. "Score on STAR
            completeness plus specificity" or "Score on algorithmic correctness and
            complexity analysis".

    Returns:
        ``{"score": 1-10, "strengths": [...], "improvements": [...],
        "sample_better_answer": "...", "feedback_summary": "..."}``. On failure it
        still returns a usable heuristic score plus an ``error`` field.
    """
    question = (question or "").strip()
    answer = (answer or "").strip()
    rubric = (rubric or "Score relevance, structure, specificity and impact out of 10.").strip()

    if not question or not answer:
        return {"error": "score_answer needs both the question and the candidate's answer.", **_heuristic_score(question, answer)}

    payload = {
        "question": question,
        "answer": answer,
        "rubric": rubric,
    }

    try:
        # Lazy import avoids a circular dependency (agent_factory imports this module).
        from agent_factory import get_scoring_model  # noqa: PLC0415

        model = get_scoring_model()
        if model is None:
            raise RuntimeError("Scoring model unavailable (missing GOOGLE_API_KEY?)")

        structured = model.with_structured_output(AnswerScore)
        result = structured.invoke(
            [("system", "You are a strict but fair technical interviewer grading one answer.")]
            + [("human", _SCORING_PROMPT.format(**payload))],
            # type: ignore[arg-type]
        )
        data = result.model_dump() if hasattr(result, "model_dump") else dict(result or {})
        if not isinstance(data, dict) or "score" not in data:
            raise ValueError("Scoring model returned an unexpected shape")
        data["score"] = max(1, min(10, int(float(data.get("score", 5)))))
        data["question"] = question
        return data
    except Exception as exc:  # noqa: BLE001 - never break the interview loop
        return {"error": f"Falling back to heuristic scoring: {exc}", **_heuristic_score(question, answer), "question": question}


_SCORING_PROMPT = """Grade this interview answer.

QUESTION: {question}

CANDIDATE ANSWER:
{answer}

RUBRIC: {rubric}

Be honest and specific. If the answer is thin, score it low and say exactly what is
missing. The sample_better_answer must stay faithful to the candidate's own
experiences and CV - never invent an achievement they did not describe.
"""


# --------------------------------------------------------------------------- #
# Tool 4 - save_note (notebook 1.3 memory, shown in the glass box)
# --------------------------------------------------------------------------- #


@tool
def save_note(key: str, value: str) -> Dict[str, Any]:
    """Remember a fact about this job search so it survives the whole session.

    Use it the moment you learn something durable, and check your notes before
    asking the user to repeat themselves. Notes appear in the Memory tab of the
    glass-box panel and in the downloadable session report.

    Args:
        key: Short snake_case name, e.g. "target_role", "target_company",
            "deadline", "salary", "location", "interview_format".
        value: The fact itself, one short line, e.g. "Backend intern, Gradia, deadline
            2026-03-01".

    Returns:
        ``{"saved": true, "key": "...", "value": "...", "notes": {...}}`` (all
        current notes) or ``{"error": ...}`` if no session is active.
    """
    session = get_active_session()
    key_clean = (key or "").strip()
    if not key_clean:
        return {"error": "save_note needs a non-empty key."}

    value_clean = (value or "").strip()
    if not value_clean:
        return {"error": "save_note needs a non-empty value."}

    if session is None:
        return {"error": "No active session - the note could not be stored."}

    # Notes are replayed into every later turn, so an unbounded value is a slow
    # memory leak: asked to "remember" a CV, the model will happily store the whole
    # document. Clamp it and say so rather than erroring, so the model cannot loop.
    truncated = False
    if len(value_clean) > MAX_NOTE_CHARS:
        value_clean = value_clean[:MAX_NOTE_CHARS].rstrip() + " …[truncated]"
        truncated = True

    session.add_note(key_clean, value_clean)
    result = {"saved": True, "key": key_clean, "value": value_clean, "notes": dict(session.notes)}
    if truncated:
        result["warning"] = (
            f"Value was over {MAX_NOTE_CHARS} characters and has been truncated. "
            "save_note is for short facts (a role, a company, a deadline) - the CV and "
            "the job posting are already in memory, so do not save them here."
        )
    return result


# --------------------------------------------------------------------------- #
# Registry / introspection helpers used by the UI
# --------------------------------------------------------------------------- #

ALL_TOOLS = [web_search, cv_gap_analyzer, score_answer, save_note]

TOOL_NAMES = [t.name for t in ALL_TOOLS]

#: Longest single note ``save_note`` will store. Notes are replayed into every later
#: turn, so this caps how fast the context can grow.
MAX_NOTE_CHARS = 400

#: short human description per tool for the glass-box tab
TOOL_BLURBS: Dict[str, str] = {
    "web_search": "Tavily web search - the only source of outside facts (notebook 1.2).",
    "cv_gap_analyzer": "Deterministic skill-gap diff between CV and posting (notebook 1.2).",
    "score_answer": "Nested structured-output LLM call that grades an answer 1-10 (notebook 1.2).",
    "save_note": "Persist job-search facts for the whole session (notebook 1.3).",
}