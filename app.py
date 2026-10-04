"""Internship Hunter Copilot - Streamlit UI and orchestration.

This file is intentionally the *only* place that touches ``st.*`` for the chat
loop. Everything it needs is imported from the sibling modules:

============  ============================================================
utils.py      text/base64/PDF helpers, session state, report export
tools.py      the four @tool functions + the active-session context
prompts.py    the four system prompts (one per mode)
agent_factory cached model / agent / **shared** InMemorySaver
============  ============================================================

The five LangChain concepts from the notebooks are wired to the UI here:
1.1 foundational models -> sidebar model switcher + temperature
1.1 prompting           -> mode selector + prompt glass box
1.2 tools               -> tool timeline + fit card + scoreboard
1.2 web search          -> Tavily tool + sources tab
1.3 memory              -> InMemorySaver, thread_id, CV memory, recall test
"""

from __future__ import annotations

import html
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
from langchain.messages import HumanMessage

# ---------------------------------------------------------------- page setup
st.set_page_config(page_title="Internship Hunter Copilot", layout="wide")

from agent_factory import (  # noqa: E402  (imports after set_page_config on purpose)
    MODEL_OPTIONS,
    get_agent,
    get_checkpointer,
    google_api_key,
    inject_secrets_into_env,
    key_status,
    load_local_env,
    make_config,
    resolve_model_name,
    stream_turn,
    tavily_api_key,
)
from prompts import (  # noqa: E402
    DEFAULT_MODE,
    DEFAULT_RUBRICS,
    INTERVIEW_MODES,
    MODE_HINTS,
    MODE_LABELS,
    get_prompt,
    prompt_diff,
)
from tools import TOOL_BLURBS, TOOL_NAMES, active_session  # noqa: E402
from utils import (  # noqa: E402
    ChatTurn,
    SessionData,
    ToolCallRecord,
    b64_to_bytes,
    build_session_report,
    capture_answer_side_effects,
    capture_tool_side_effects,
    classify_error,
    clean_text,
    extract_role,
    extract_text,
    file_to_b64,
    guess_mime,
    looks_like_role_line,
    new_session_data,
    parse_cv_upload,
    report_filename,
    truncate,
)

# ---------------------------------------------------------------- env/keys
load_local_env()
inject_secrets_into_env()


# ========================================================================== #
# CSS - modern, clean, simple styling
# ========================================================================== #
def inject_css() -> None:
    """Modern, clean, premium CSS — deep navy + indigo accent palette."""
    st.markdown(
        r"""
<style>
  /* ================================================================
     INTERNSHIP HUNTER COPILOT  —  Design System v2
     Palette : Navy #0d1117 · Surface #161b22 · Border #21262d
     Accent  : Indigo #6366f1 · Violet #818cf8 · Teal #22d3a0
     Text    : #e6edf3 (primary) · #8b949e (muted) · #c9d1d9 (body)
     Font    : Inter (400-800) from Google Fonts
  ================================================================ */

  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

  /* ── Global ─────────────────────────────────────────────────── */
  html, body, [class*="css"] {
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
      -webkit-font-smoothing: antialiased;
      -moz-osx-font-smoothing: grayscale;
  }
  .block-container {
      padding-top: 1.75rem !important;
      padding-bottom: 2rem !important;
      max-width: 1560px !important;
  }

  /* ── Typography ─────────────────────────────────────────────── */
  h1, h2, h3, h4, h5 {
      font-family: 'Inter', sans-serif;
      letter-spacing: -0.03em;
      font-weight: 700;
  }
  h4 {
      font-size: 0.7rem !important;
      font-weight: 700 !important;
      letter-spacing: 0.1em !important;
      text-transform: uppercase !important;
      color: #484f58 !important;
      margin-bottom: 0.4rem !important;
  }

  /* ── App header ─────────────────────────────────────────────── */
  .ihc-header-wrap {
      margin-bottom: 1.15rem;
      padding-bottom: 0.9rem;
      border-bottom: 1px solid rgba(99, 102, 241, 0.14);
  }
  .ihc-title {
      font-size: 1.3rem;
      font-weight: 800;
      letter-spacing: -0.04em;
      margin: 0 0 0.12rem 0;
      background: linear-gradient(130deg, #e6edf3 0%, #a5b4fc 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
  }
  .ihc-subtitle {
      font-size: 0.8rem;
      color: #484f58;
      font-weight: 400;
  }
  .ihc-subtitle code {
      background: rgba(99, 102, 241, 0.13);
      color: #818cf8;
      border-radius: 4px;
      padding: 0.04rem 0.32rem;
      font-size: 0.77rem;
  }
  .ihc-subtitle b { color: #a5b4fc; font-weight: 600; }

  /* ── Status dots ────────────────────────────────────────────── */
  .ihc-status-row {
      display: flex;
      gap: 1.1rem;
      margin: 0.45rem 0 0.85rem 0;
      font-size: 0.75rem;
      font-weight: 500;
      color: #484f58;
  }
  .ihc-dot {
      display: inline-block;
      width: 6px; height: 6px;
      border-radius: 50%;
      margin-right: 0.28rem;
      position: relative;
      top: -0.5px;
  }
  .ihc-ok-dot  { background: #22d3a0; box-shadow: 0 0 0 2px rgba(34,211,160,.16), 0 0 7px rgba(34,211,160,.35); }
  .ihc-bad-dot { background: #f87171; box-shadow: 0 0 0 2px rgba(248,113,113,.16), 0 0 7px rgba(248,113,113,.35); }

  /* ── Skill chips ────────────────────────────────────────────── */
  .ihc-chip {
      display: inline-flex;
      align-items: center;
      padding: 0.17rem 0.58rem;
      margin: 0.1rem 0.18rem 0.1rem 0;
      border-radius: 99px;
      font-size: 0.72rem;
      font-weight: 600;
      white-space: nowrap;
      letter-spacing: 0.01em;
      transition: transform 0.12s ease, box-shadow 0.12s ease;
      cursor: default;
  }
  .ihc-chip:hover { transform: translateY(-1px); }
  .ihc-chip-ok   { background: rgba(34,211,160,.1); color: #34d399; border: 1px solid rgba(34,211,160,.22); }
  .ihc-chip-ok:hover   { box-shadow: 0 2px 8px rgba(34,211,160,.2); }
  .ihc-chip-miss { background: rgba(248,113,113,.1); color: #fc8181; border: 1px solid rgba(248,113,113,.22); }
  .ihc-chip-miss:hover { box-shadow: 0 2px 8px rgba(248,113,113,.2); }
  .ihc-chip-nice { background: rgba(251,191,36,.1);  color: #fbbf24; border: 1px solid rgba(251,191,36,.22); }
  .ihc-chip-nice:hover { box-shadow: 0 2px 8px rgba(251,191,36,.15); }
  .ihc-chip-neutral { background: rgba(100,116,139,.1); color: #8b949e; border: 1px solid rgba(100,116,139,.18); }

  /* ── Cards ──────────────────────────────────────────────────── */
  .ihc-card {
      border: 1px solid rgba(99,102,241,.12);
      border-radius: 10px;
      padding: 0.9rem 1.1rem;
      margin-bottom: 0.55rem;
      background: linear-gradient(135deg, rgba(13,17,23,.85) 0%, rgba(25,22,60,.35) 100%);
  }
  .ihc-card-title {
      font-size: 0.67rem;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      color: #6366f1;
      font-weight: 700;
      margin-bottom: 0.45rem;
  }
  .ihc-kpi {
      font-size: 2.3rem;
      font-weight: 800;
      line-height: 1;
      letter-spacing: -0.04em;
  }
  .ihc-kpi span { opacity: 0.5; font-size: 1rem; }
  .ihc-muted { color: #484f58; font-size: 0.79rem; }

  /* ── Sources ────────────────────────────────────────────────── */
  .ihc-src {
      font-size: 0.8rem;
      padding: 0.45rem 0.7rem;
      margin-bottom: 0.35rem;
      border-radius: 7px;
      background: rgba(13,17,23,.55);
      border: 1px solid rgba(99,102,241,.09);
      word-break: break-all;
      transition: border-color 0.15s;
  }
  .ihc-src:hover { border-color: rgba(99,102,241,.28); }
  .ihc-src a { color: #818cf8; text-decoration: none; }
  .ihc-src a:hover { color: #a5b4fc; text-decoration: underline; }

  /* ── Tool call timeline ─────────────────────────────────────── */
  .ihc-tl {
      font-size: 0.79rem;
      border-left: 2px solid #6366f1;
      padding: 0.4rem 0.75rem;
      margin-bottom: 0.4rem;
      border-radius: 0 7px 7px 0;
      background: linear-gradient(90deg, rgba(99,102,241,.06) 0%, transparent 100%);
  }

  /* ── Badges ─────────────────────────────────────────────────── */
  .ihc-badge {
      display: inline-block;
      padding: 0.1rem 0.48rem;
      border-radius: 99px;
      font-size: 0.67rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.06em;
  }
  .ihc-badge-high { background: rgba(34,211,160,.11); color: #34d399; border: 1px solid rgba(34,211,160,.28); }
  .ihc-badge-mid  { background: rgba(251,191,36,.11); color: #fbbf24; border: 1px solid rgba(251,191,36,.28); }
  .ihc-badge-low  { background: rgba(248,113,113,.11); color: #fc8181; border: 1px solid rgba(248,113,113,.28); }

  /* ── Sidebar ────────────────────────────────────────────────── */
  section[data-testid="stSidebar"] {
      border-right: 1px solid rgba(99,102,241,.1) !important;
  }
  section[data-testid="stSidebar"] .block-container {
      padding-top: 1.35rem !important;
  }
  section[data-testid="stSidebar"] h2 {
      font-size: 1.05rem;
      font-weight: 800;
      letter-spacing: -0.03em;
      color: #e6edf3;
  }
  section[data-testid="stSidebar"] h4 {
      font-size: 0.67rem !important;
      font-weight: 700 !important;
      letter-spacing: 0.1em !important;
      text-transform: uppercase !important;
      color: #484f58 !important;
      margin-top: 1rem !important;
      margin-bottom: 0.3rem !important;
  }

  /* ── Sidebar brand block ────────────────────────────────────── */
  .ihc-brand {
      display: flex;
      align-items: center;
      gap: 0.6rem;
      margin-bottom: 0.55rem;
  }
  .ihc-brand-icon {
      width: 30px; height: 30px;
      border-radius: 7px;
      background: linear-gradient(135deg, #6366f1 0%, #818cf8 100%);
      display: flex; align-items: center; justify-content: center;
      font-size: 0.85rem; flex-shrink: 0;
      box-shadow: 0 2px 8px rgba(99,102,241,.4);
  }
  .ihc-brand-name {
      font-size: 0.95rem;
      font-weight: 800;
      letter-spacing: -0.03em;
      color: #e6edf3;
      line-height: 1.1;
  }
  .ihc-brand-sub {
      font-size: 0.66rem;
      color: #484f58;
      font-weight: 400;
  }

  /* ── Streamlit overrides ────────────────────────────────────── */

  /* Primary button → indigo gradient */
  .stButton > button[kind="primary"] {
      background: linear-gradient(135deg, #6366f1 0%, #818cf8 100%) !important;
      border: none !important;
      color: #fff !important;
      font-weight: 600 !important;
      font-size: 0.81rem !important;
      border-radius: 7px !important;
      letter-spacing: 0.01em;
      transition: opacity 0.14s ease, transform 0.1s ease, box-shadow 0.14s ease !important;
      box-shadow: 0 1px 6px rgba(99,102,241,.28) !important;
  }
  .stButton > button[kind="primary"]:hover {
      opacity: 0.88 !important;
      transform: translateY(-1px) !important;
      box-shadow: 0 4px 14px rgba(99,102,241,.4) !important;
  }

  /* Secondary button → ghost */
  .stButton > button[kind="secondary"] {
      background: rgba(22,27,34,.7) !important;
      border: 1px solid rgba(99,102,241,.17) !important;
      color: #8b949e !important;
      font-weight: 500 !important;
      font-size: 0.81rem !important;
      border-radius: 7px !important;
      transition: border-color 0.14s ease, color 0.14s ease, background 0.14s ease !important;
  }
  .stButton > button[kind="secondary"]:hover {
      border-color: rgba(99,102,241,.42) !important;
      color: #c7d2fe !important;
      background: rgba(99,102,241,.07) !important;
  }

  /* Chat input */
  .stChatInput textarea {
      border-radius: 10px !important;
      border: 1px solid rgba(99,102,241,.18) !important;
      font-size: 0.88rem !important;
      transition: border-color 0.18s ease, box-shadow 0.18s ease !important;
  }
  .stChatInput textarea:focus {
      border-color: rgba(99,102,241,.48) !important;
      box-shadow: 0 0 0 3px rgba(99,102,241,.09) !important;
  }

  /* Chat bubbles */
  .stChatMessage { border-radius: 10px !important; margin-bottom: 0.45rem !important; }

  /* Expanders */
  .streamlit-expanderHeader {
      font-size: 0.83rem !important;
      font-weight: 600 !important;
      color: #8b949e !important;
      border-radius: 7px !important;
  }
  .streamlit-expanderHeader:hover { color: #c7d2fe !important; }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {
      gap: 0.1rem;
      border-bottom: 1px solid rgba(99,102,241,.1) !important;
  }
  .stTabs [data-baseweb="tab"] {
      font-size: 0.79rem !important;
      font-weight: 600 !important;
      color: #484f58 !important;
      border-radius: 5px 5px 0 0 !important;
      padding: 0.35rem 0.85rem !important;
      transition: color 0.13s ease !important;
  }
  .stTabs [data-baseweb="tab"]:hover { color: #8b949e !important; }
  .stTabs [aria-selected="true"] {
      color: #818cf8 !important;
      border-bottom: 2px solid #6366f1 !important;
  }

  /* Progress bar */
  .stProgress > div > div {
      background: linear-gradient(90deg, #6366f1, #818cf8) !important;
      border-radius: 99px !important;
  }

  /* Metrics */
  [data-testid="stMetricValue"] {
      font-size: 1.45rem !important;
      font-weight: 800 !important;
      letter-spacing: -0.03em !important;
      color: #e6edf3 !important;
  }
  [data-testid="stMetricLabel"] {
      font-size: 0.68rem !important;
      font-weight: 700 !important;
      letter-spacing: 0.07em !important;
      text-transform: uppercase !important;
      color: #484f58 !important;
  }

  /* Status widget (agent is working…) */
  [data-testid="stStatusWidget"] {
      border-radius: 9px !important;
      border: 1px solid rgba(99,102,241,.13) !important;
      background: rgba(13,17,23,.7) !important;
  }

  /* Alerts */
  [data-testid="stAlert"] {
      border-radius: 7px !important;
      font-size: 0.83rem !important;
  }

  /* Dividers */
  hr { border-color: rgba(99,102,241,.09) !important; margin: 0.9rem 0 !important; }

  /* Code */
  code {
      font-family: 'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace !important;
      font-size: 0.81em !important;
  }

  /* Captions */
  .stCaption, [data-testid="stCaptionContainer"] {
      color: #484f58 !important;
      font-size: 0.74rem !important;
  }

  /* Thin scrollbar */
  ::-webkit-scrollbar { width: 4px; height: 4px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: rgba(99,102,241,.22); border-radius: 99px; }
  ::-webkit-scrollbar-thumb:hover { background: rgba(99,102,241,.42); }

  /* File uploader */
  [data-testid="stFileUploaderDropzone"] {
      border: 1px dashed rgba(99,102,241,.22) !important;
      border-radius: 9px !important;
      background: rgba(99,102,241,.025) !important;
      transition: border-color 0.18s, background 0.18s !important;
  }
  [data-testid="stFileUploaderDropzone"]:hover {
      border-color: rgba(99,102,241,.42) !important;
      background: rgba(99,102,241,.05) !important;
  }

  /* Text areas */
  textarea {
      border-radius: 7px !important;
      font-size: 0.83rem !important;
      border: 1px solid rgba(99,102,241,.14) !important;
      transition: border-color 0.18s !important;
  }
  textarea:focus {
      border-color: rgba(99,102,241,.38) !important;
      box-shadow: 0 0 0 2px rgba(99,102,241,.07) !important;
  }

  /* Utility scroll */
  .ihc-scroll { max-height: 22rem; overflow-y: auto; padding-right: 0.3rem; }

  /* Full-bleed glass career hero inspired by the supplied reference */
  .dream-shell {
      position: relative;
      min-height: 390px;
      padding: 1.1rem 1.25rem;
      margin: 0 0 1.4rem;
      border: 1px solid rgba(255,255,255,0.28);
      border-radius: 28px;
      overflow: hidden;
      isolation: isolate;
      background: #15212d;
      box-shadow: 0 24px 65px rgba(0,0,0,0.34);
  }
  .dream-shell::before {
      content: "";
      position: absolute;
      inset: 0;
      z-index: -1;
      background:
          linear-gradient(90deg, rgba(10,17,27,0.72) 0%, rgba(10,17,27,0.24) 48%, rgba(10,17,27,0.08) 100%),
          linear-gradient(0deg, rgba(8,14,23,0.72) 0%, transparent 42%),
          url('https://images.unsplash.com/photo-1497366754035-f200968a6e72?auto=format&fit=crop&w=1800&q=85') center 48% / cover;
      transform: scale(1.015);
  }
  .dream-topbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 0.75rem;
      min-height: 42px;
      position: relative;
      z-index: 1;
  }
  .dream-navgroup {
      display: flex;
      align-items: center;
      gap: 0.4rem;
      flex-wrap: wrap;
  }
  .dream-brand {
      color: #fff;
      font-size: 0.83rem;
      font-weight: 700;
      letter-spacing: -0.02em;
      margin-right: 0.55rem;
      white-space: nowrap;
  }
  .dream-pill {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 30px;
      padding: 0.38rem 0.75rem;
      border-radius: 999px;
      font-size: 0.67rem;
      font-weight: 500;
      color: rgba(255,255,255,0.84);
      background: rgba(255,255,255,0.12);
      border: 1px solid rgba(255,255,255,0.2);
      backdrop-filter: blur(18px);
      -webkit-backdrop-filter: blur(18px);
  }
  .dream-pill.active {
      color: #111827;
      background: rgba(255,255,255,0.91);
      border-color: rgba(255,255,255,0.92);
  }
  .dream-mini-actions {
      display: flex;
      align-items: center;
      gap: 0.45rem;
  }
  .dream-mini-btn {
      width: 32px;
      height: 32px;
      border-radius: 50%;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      color: #fff;
      background: rgba(255,255,255,0.13);
      border: 1px solid rgba(255,255,255,0.24);
      backdrop-filter: blur(16px);
      font-size: 0.75rem;
      font-weight: 700;
  }
  .dream-hero {
      min-height: 270px;
      display: flex;
      align-items: center;
      justify-content: center;
      position: relative;
      z-index: 1;
      text-align: center;
  }
  .dream-copy {
      max-width: 760px;
      padding: 0.6rem 1rem 1.5rem;
  }
  .dream-kicker {
      font-size: 0.7rem;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: rgba(255,255,255,0.78);
      margin: 0 0 0.75rem;
      font-weight: 700;
  }
  .dream-title {
      margin: 0;
      font-size: clamp(2.8rem, 6vw, 5.1rem);
      line-height: 0.88;
      letter-spacing: -0.075em;
      color: #fff;
      font-weight: 800;
      text-shadow: 0 4px 30px rgba(0,0,0,0.28);
  }
  .dream-title em {
      font-style: normal;
      color: rgba(255,255,255,0.94);
      display: block;
      font-weight: 800;
  }
  .dream-badge {
      position: absolute;
      right: 1.25rem;
      bottom: 1.15rem;
      display: flex;
      align-items: center;
      gap: 0.65rem;
      padding: 0.6rem 0.8rem 0.6rem 0.6rem;
      background: rgba(245,248,252,0.84);
      border: 1px solid rgba(255,255,255,0.48);
      border-radius: 15px;
      box-shadow: 0 16px 36px rgba(0,0,0,0.22);
      backdrop-filter: blur(22px);
      -webkit-backdrop-filter: blur(22px);
      z-index: 2;
      max-width: 245px;
  }
  .dream-badge-avatar {
      width: 38px;
      height: 38px;
      border-radius: 13px;
      background: linear-gradient(135deg, #392178, #171044);
      color: white;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      font-weight: 800;
      font-size: 0.92rem;
      box-shadow: 0 8px 18px rgba(23,16,68,0.28);
  }
  .dream-badge-copy {
      display: flex;
      flex-direction: column;
      flex: 1;
      min-width: 0;
  }
  .dream-badge-copy strong {
      font-size: 0.76rem;
      color: #111827;
      line-height: 1.3;
  }
  .dream-badge-copy small {
      color: #64748b;
      font-size: 0.63rem;
      margin-top: 0.12rem;
  }
  .dream-arrow {
      width: 29px;
      height: 29px;
      border-radius: 50%;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      background: rgba(255,255,255,0.9);
      color: #111827;
      font-size: 0.95rem;
      font-weight: 700;
      border: 1px solid rgba(148,163,184,0.22);
  }
  .dream-footnote {
      position: absolute;
      left: 1.25rem;
      bottom: 1.35rem;
      z-index: 2;
      color: rgba(255,255,255,0.78);
      font-size: 0.67rem;
      letter-spacing: 0.04em;
      text-shadow: 0 2px 12px rgba(0,0,0,0.35);
  }
  @media (max-width: 700px) {
      .dream-shell {
          min-height: 390px;
          border-radius: 22px;
          padding: 0.8rem;
      }
      .dream-topbar {
          align-items: flex-start;
      }
      .dream-brand {
          display: none;
      }
      .dream-navgroup {
          gap: 0.3rem;
      }
      .dream-pill {
          padding: 0.35rem 0.55rem;
          font-size: 0.6rem;
      }
      .dream-mini-actions {
          gap: 0.3rem;
      }
      .dream-mini-btn {
          width: 28px;
          height: 28px;
      }
      .dream-title {
          font-size: clamp(2.6rem, 13vw, 4rem);
      }
      .dream-badge {
          right: 0.7rem;
          bottom: 0.7rem;
      }
      .dream-footnote {
          left: 0.8rem;
          bottom: 0.95rem;
          max-width: 36%;
      }
  }

  /* Matching glass surfaces for the workspace below the hero */
  div[data-testid="stHorizontalBlock"]:has(.ihc-header-wrap) > div[data-testid="stColumn"] {
      min-width: 0;
      padding: 1.15rem !important;
      border: 1px solid rgba(255,255,255,0.09);
      border-radius: 22px;
      background: linear-gradient(145deg, rgba(24,31,44,0.78), rgba(14,19,29,0.72));
      box-shadow: 0 20px 48px rgba(0,0,0,0.2), inset 0 1px 0 rgba(255,255,255,0.035);
      backdrop-filter: blur(22px);
      -webkit-backdrop-filter: blur(22px);
  }
  .ihc-header-wrap {
      padding: 0.1rem 0 0.8rem;
      border-bottom-color: rgba(255,255,255,0.1);
  }
  .ihc-subtitle, .ihc-muted {
      color: #a4afc0 !important;
  }
  .ihc-subtitle code {
      background: rgba(255,255,255,0.08);
      color: #d2c9ff;
  }
  div[data-testid="stHorizontalBlock"]:has(.ihc-header-wrap) h4,
  div[data-testid="stHorizontalBlock"]:has(.ihc-header-wrap) h3 {
      color: #f1f5f9;
  }
  div[data-testid="stHorizontalBlock"]:has(.ihc-header-wrap) [data-testid="stAlert"] {
      background: rgba(255,255,255,0.055);
      border: 1px solid rgba(255,255,255,0.1);
      color: #dbe4f0;
      backdrop-filter: blur(14px);
  }
  div[data-testid="stHorizontalBlock"]:has(.ihc-header-wrap) .stButton > button {
      min-height: 2.45rem;
      border-radius: 999px !important;
      background: rgba(255,255,255,0.055) !important;
      border: 1px solid rgba(255,255,255,0.13) !important;
      color: #dce5f1 !important;
      box-shadow: none !important;
  }
  div[data-testid="stHorizontalBlock"]:has(.ihc-header-wrap) .stButton > button:hover {
      background: rgba(255,255,255,0.11) !important;
      border-color: rgba(255,255,255,0.25) !important;
      color: #fff !important;
      transform: translateY(-1px);
  }
  div[data-testid="stHorizontalBlock"]:has(.ihc-header-wrap) .stButton > button[kind="primary"] {
      background: linear-gradient(135deg, #5945aa, #7964d0) !important;
      border-color: rgba(190,176,255,0.28) !important;
      color: #fff !important;
      box-shadow: 0 8px 22px rgba(70,51,151,0.25) !important;
  }
  div[data-testid="stHorizontalBlock"]:has(.ihc-header-wrap) .streamlit-expanderHeader,
  div[data-testid="stHorizontalBlock"]:has(.ihc-header-wrap) [data-testid="stExpander"] {
      border-color: rgba(255,255,255,0.1) !important;
  }
  div[data-testid="stHorizontalBlock"]:has(.ihc-header-wrap) [data-testid="stExpander"] {
      border-radius: 14px;
      background: rgba(255,255,255,0.025);
  }
  div[data-testid="stHorizontalBlock"]:has(.ihc-header-wrap) [data-testid="stChatMessage"] {
      border: 1px solid rgba(255,255,255,0.075);
      border-radius: 16px !important;
      background: rgba(255,255,255,0.045);
      padding: 0.75rem 0.9rem;
      backdrop-filter: blur(14px);
  }
  div[data-testid="stHorizontalBlock"]:has(.ihc-header-wrap) .stChatInput textarea {
      background: rgba(255,255,255,0.075) !important;
      border: 1px solid rgba(255,255,255,0.16) !important;
      border-radius: 16px !important;
      color: #f8fafc !important;
  }
  div[data-testid="stHorizontalBlock"]:has(.ihc-header-wrap) .stChatInput textarea::placeholder {
      color: #98a4b6 !important;
  }
  div[data-testid="stHorizontalBlock"]:has(.ihc-header-wrap) .stTabs [data-baseweb="tab-list"] {
      gap: 0.25rem;
      border-bottom-color: rgba(255,255,255,0.1) !important;
  }
  div[data-testid="stHorizontalBlock"]:has(.ihc-header-wrap) .stTabs [data-baseweb="tab"] {
      color: #9da9bb !important;
  }
  div[data-testid="stHorizontalBlock"]:has(.ihc-header-wrap) .stTabs [aria-selected="true"] {
      color: #e2dcff !important;
      border-bottom-color: #9985ed !important;
  }
  div[data-testid="stHorizontalBlock"]:has(.ihc-header-wrap) .ihc-card {
      border-color: rgba(255,255,255,0.12);
      background: linear-gradient(135deg, rgba(255,255,255,0.09), rgba(255,255,255,0.035));
      backdrop-filter: blur(18px);
  }
  div[data-testid="stHorizontalBlock"]:has(.ihc-header-wrap) .ihc-card-title {
      color: #c2b5ff;
  }
  div[data-testid="stHorizontalBlock"]:has(.ihc-header-wrap) [data-testid="stMetricLabel"] {
      color: #9da9bb !important;
  }
  div[data-testid="stHorizontalBlock"]:has(.ihc-header-wrap) [data-testid="stCode"],
  div[data-testid="stHorizontalBlock"]:has(.ihc-header-wrap) pre {
      border-color: rgba(255,255,255,0.08) !important;
      border-radius: 14px !important;
      background: rgba(5,9,16,0.42) !important;
  }

  /* Sidebar controls share the same translucent charcoal surfaces */
  section[data-testid="stSidebar"] > div:first-child {
      background:
          radial-gradient(ellipse at 0% 0%, rgba(96,75,170,0.13), transparent 42%),
          linear-gradient(180deg, #111722 0%, #0c111a 100%);
      border-right: 1px solid rgba(255,255,255,0.08);
  }
  section[data-testid="stSidebar"] .block-container {
      padding-left: 1.1rem !important;
      padding-right: 1.1rem !important;
  }
  section[data-testid="stSidebar"] h4 {
      color: #bbc5d4 !important;
      letter-spacing: 0.12em !important;
  }
  section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
  section[data-testid="stSidebar"] .stCaption {
      color: #97a3b5 !important;
  }
  section[data-testid="stSidebar"] [data-testid="stExpander"] {
      border: 1px solid rgba(255,255,255,0.1) !important;
      border-radius: 15px !important;
      background: rgba(255,255,255,0.035) !important;
      backdrop-filter: blur(16px);
  }
  section[data-testid="stSidebar"] [data-testid="stExpander"] summary {
      color: #e6ebf2 !important;
  }
  section[data-testid="stSidebar"] [data-testid="stTextInput"] input,
  section[data-testid="stSidebar"] [data-testid="stTextArea"] textarea,
  section[data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] > div,
  section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
      background: rgba(255,255,255,0.055) !important;
      border-color: rgba(255,255,255,0.13) !important;
      border-radius: 12px !important;
      color: #edf2f8 !important;
  }
  section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] label,
  section[data-testid="stSidebar"] [data-testid="stRadio"] label,
  section[data-testid="stSidebar"] [data-testid="stCheckbox"] label,
  section[data-testid="stSidebar"] [data-testid="stToggle"] label {
      color: #d5deea !important;
  }
  section[data-testid="stSidebar"] [data-testid="stRadio"] [data-testid="stMarkdownContainer"] p {
      color: #c3cddd;
  }
  section[data-testid="stSidebar"] .stButton > button,
  section[data-testid="stSidebar"] [data-testid="stDownloadButton"] button {
      min-height: 2.35rem;
      border-radius: 999px !important;
  }
  section[data-testid="stSidebar"] .stButton > button[kind="secondary"],
  section[data-testid="stSidebar"] [data-testid="stDownloadButton"] button {
      background: rgba(255,255,255,0.055) !important;
      border: 1px solid rgba(255,255,255,0.13) !important;
      color: #dce5f1 !important;
  }
  section[data-testid="stSidebar"] .ihc-brand-icon {
      border-radius: 11px;
      box-shadow: 0 7px 20px rgba(99,76,190,0.34);
  }
  section[data-testid="stSidebar"] .ihc-brand-sub {
      color: #98a4b5;
  }
</style>
""",
        unsafe_allow_html=True,
    )



# ========================================================================== #
# Session state bootstrap
# ========================================================================== #
def init_state() -> None:
    """Create the per-browser-session objects exactly once.

    ``st.session_state`` is reset only when the browser session changes, so the
    ``thread_id`` (and therefore the LangGraph conversation memory) is stable across
    Streamlit reruns.
    """
    if "session" not in st.session_state:
        st.session_state.session: SessionData = new_session_data()

    # NOTE: only *non-widget* keys may be seeded here. Streamlit refuses to let
    # st.session_state pre-populate a widget key (`StreamlitValueAssignmentNotAllowedError`),
    # so widget values are read through the helpers below instead.
    defaults: Dict[str, Any] = {
        "last_mode": DEFAULT_MODE,
        "pending": None,  # (text, images, audio) queued by a quick-action button
        "just_saved_cv": False,
        "just_saved_job": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def model_choice() -> str:
    """Sidebar model tier ("Fast" / "Deep"), defaulting before the widget exists."""
    return str(st.session_state.get("model_choice") or "Fast")


def mode() -> str:
    """Active mode key, defaulting before the widget exists."""
    value = st.session_state.get("mode") or DEFAULT_MODE
    return value if value in MODE_LABELS else DEFAULT_MODE


def temperature() -> float:
    """Active temperature, defaulting before the widget exists."""
    try:
        return float(st.session_state.get("temperature", 0.3))
    except (TypeError, ValueError):
        return 0.3


def tools_enabled() -> bool:
    """Whether the agent gets the tools, defaulting before the widget exists."""
    value = st.session_state.get("tools_enabled", True)
    return True if value is None else bool(value)


def seed_widget(key: str, default: Any) -> None:
    """Initialise a widget key **once**, before that widget is instantiated.

    Streamlit forbids writing ``st.session_state[key]`` once a widget with that key
    exists, so every sidebar widget is seeded exactly like this instead of in
    :func:`init_state`.
    """
    if key not in st.session_state:
        st.session_state[key] = default


def session() -> SessionData:
    """Shortcut to the active :class:`SessionData`."""
    return st.session_state.session


# ========================================================================== #
# Sidebar
# ========================================================================== #
def render_sidebar() -> None:
    """Model switcher, mode selector, temperature, CV memory, job description, session controls."""
    with st.sidebar:
        st.markdown(
            """
<div class="ihc-brand">
  <div class="ihc-brand-icon">IH</div>
  <div>
    <div class="ihc-brand-name">Internship Hunter</div>
    <div class="ihc-brand-sub">Copilot &amp; Career Lab</div>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )
        st.caption("Gemini + LangChain \u2014 web research, CV gap analysis & mock interviews.")


        # Widget keys are seeded here, once, *before* the widgets below exist.
        seed_widget("mode", DEFAULT_MODE)
        seed_widget("model_choice", "Fast")
        seed_widget("temperature", 0.3)
        seed_widget("tools_enabled", True)

        status = key_status()
        st.markdown(
            f"""
<div class="ihc-status-row">
  <span><span class="ihc-dot {'ihc-ok-dot' if status['GOOGLE_API_KEY'] else 'ihc-bad-dot'}"></span> GOOGLE_API_KEY</span>
  <span><span class="ihc-dot {'ihc-ok-dot' if status['TAVILY_API_KEY'] else 'ihc-bad-dot'}"></span> TAVILY_API_KEY</span>
</div>
""",
            unsafe_allow_html=True,
        )
        if not status["GOOGLE_API_KEY"]:
            st.error("Missing GOOGLE_API_KEY - see the setup screen.")
        if not status["TAVILY_API_KEY"]:
            st.warning("No TAVILY_API_KEY: research is disabled, the agent will say so.")

        st.divider()

        # ---- 1.1 prompting: the mode selector -----------------------------
        st.markdown("#### Mode")
        render_mode_selector()
        st.caption(MODE_HINTS.get(mode(), ""))
        if st.session_state.get("last_mode", DEFAULT_MODE) != mode():
            st.info(prompt_diff(str(st.session_state.get("last_mode", DEFAULT_MODE)), mode()))

        # ---- 1.1 foundational models: the model switcher ------------------
        st.markdown("#### Model")
        choice_labels = list(MODEL_OPTIONS.keys())
        st.radio(
            "Model tier",
            choice_labels,
            key="model_choice",
            format_func=lambda label: {
                "Fast": f"Fast - {resolve_model_name('Fast')}",
                "Deep": f"Deep - {resolve_model_name('Deep')}",
            }[label],
            help="Switch mid-conversation: the memory stays, only the model changes (shared checkpointer + thread_id).",
        )
        st.caption(
            f"Fast -> `{resolve_model_name('Fast')}`  ·  Deep -> `{resolve_model_name('Deep')}`"
        )

        with st.expander("Settings", expanded=False):
            st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                step=0.05,
                key="temperature",
                help="Applied when the agent is rebuilt (i.e. on the next message after a change).",
            )
            st.toggle(
                "Enable tools (search / analyzer / scorer / notes)",
                key="tools_enabled",
                help="Turn everything off to watch the same prompt behave as a plain chatbot.",
            )
            st.caption(
                f"Rebuilds cached: {len(get_agent_cache_keys())} agent(s) live. "
                f"Checkpointer: `{type(get_checkpointer()).__name__}`."
            )

        st.divider()

        # ---- 1.3 memory: CV & Job Description ----------------------------
        st.markdown("#### Memory & Documents")

        with st.expander("Paste or upload a CV", expanded=not bool(session().cv_text)):
            st.text_area(
                "CV text",
                key="cv_draft",
                height=160,
                placeholder="Paste your CV here (plain text)…",
                help="Stored in this session only, injected into the agent memory with a hidden 'Remember my CV' message.",
            )
            st.file_uploader(
                "…or upload .txt / .md / .pdf",
                type=["txt", "md", "pdf"],
                key="cv_upload",
                help="PDFs are parsed with pypdf. Scanned/image-only PDFs cannot be read - paste the text instead.",
            )
            if st.button("Save CV to memory", use_container_width=True, type="primary"):
                save_cv_to_memory()

        draft = str(st.session_state.get("cv_draft") or "").strip()
        if draft and draft != (session().cv_text or "").strip():
            st.warning("Unsaved CV edits — press **Save CV to memory** before asking the agent to use it.")

        if session().cv_text:
            st.success(
                f"CV loaded · {len(session().cv_text):,} chars"
                + (f" · `{session().cv_name}`" if session().cv_name else "")
            )
            st.caption(f"Thread `{session().thread_id[:8]}` remembers it for the whole session.")

        with st.expander("Target job posting", expanded=not bool(session().job_text)):
            st.text_area(
                "Job description text",
                key="job_draft",
                height=140,
                placeholder="Paste the target job posting / internship description here…",
                help="Used directly by cv_gap_analyzer to compute skill match, gaps, and tailored interview prep.",
            )
            st.file_uploader(
                "…or upload job posting (.txt / .md / .pdf)",
                type=["txt", "md", "pdf"],
                key="job_upload",
                help="Upload a job description file. PDFs and text files are supported.",
            )
            c_job_save, c_job_clear = st.columns([2, 1])
            with c_job_save:
                if st.button("Save Job Posting", use_container_width=True, type="primary"):
                    save_job_to_memory()
            with c_job_clear:
                if st.button("Clear Job", use_container_width=True):
                    session().job_text = ""
                    st.session_state["job_draft"] = ""
                    st.rerun()

        job_draft_cur = str(st.session_state.get("job_draft") or "").strip()
        if job_draft_cur and job_draft_cur != (session().job_text or "").strip():
            st.warning("Unsaved job posting edits — press **Save Job Posting** before asking the agent to use it.")

        if session().job_text:
            st.info(f"Job posting loaded · {len(session().job_text):,} chars")
            if session().role or session().company:
                details = []
                if session().role:
                    details.append(f"Role: {session().role}")
                if session().company:
                    details.append(f"Company: {session().company}")
                st.caption(" · ".join(details))

        st.divider()

        # ---- session controls --------------------------------------------
        st.markdown("#### Session Controls")
        c1, c2 = st.columns(2)
        if c1.button("New session", use_container_width=True, help="New thread_id + cleared history and notes (the CV is kept)."):
            start_new_session()
            st.rerun()
        if c2.button("Recall test", use_container_width=True, help="Ask the agent what it remembers about you and this role."):
            st.session_state.pending = ("What do you remember about me and this role? Reply in a short bulleted recap.", None, None)
            st.rerun()

        if session().tool_calls or session().chat:
            st.download_button(
                "Download session report",
                data=build_session_report(
                    session(),
                    model_name=resolve_model_name(model_choice()),
                    mode_label=MODE_LABELS.get(mode(), mode()),
                    temperature=temperature(),
                    search_enabled=tools_enabled() and bool(tavily_api_key()),
                ),
                file_name=report_filename(session()),
                mime="text/markdown",
                use_container_width=True,
            )

        st.caption(
            "Memory is **in-process** (`InMemorySaver`): it is per browser session and resets when the server restarts."
        )


def render_mode_selector() -> None:
    """Mode picker built from plain buttons without emojis."""
    keys = list(MODE_LABELS)
    for row_start in range(0, len(keys), 2):
        cols = st.columns(2)
        for key, col in zip(keys[row_start : row_start + 2], cols, strict=False):
            selected = key == mode()
            with col:
                clicked = st.button(
                    ("[active] " if selected else "") + MODE_LABELS[key],
                    key=f"mode_btn_{key}",
                    use_container_width=True,
                    type="primary" if selected else "secondary",
                    help=MODE_HINTS.get(key, ""),
                )
            if clicked and not selected:
                st.session_state.last_mode = mode()
                st.session_state.mode = key
                st.rerun()


def get_agent_cache_keys() -> List[Tuple[str, str, float, bool]]:
    """Keys of the memoised agents (safe before any agent exists)."""
    try:
        from agent_factory import agent_cache_info  # local import to avoid cycles

        return agent_cache_info()["cached_agents"]
    except Exception:  # noqa: BLE001
        return []


# ========================================================================== #
# CV & Job Memory
# ========================================================================== #
def save_cv_to_memory() -> None:
    """Persist the CV in session state **and** in the agent's message memory."""
    text = clean_text(str(st.session_state.get("cv_draft") or ""))
    name = ""
    upload = st.session_state.get("cv_upload")  # widget value, read-only access
    if upload is not None:
        try:
            text = parse_cv_upload(upload)
            name = upload.name
        except Exception as exc:  # noqa: BLE001
            title, hint = classify_error(exc)
            st.error(f"{title}. {hint}")
            return

    if not text:
        st.warning("Nothing to save - paste some CV text or upload a file first.")
        return

    session().cv_text = text
    session().cv_name = name

    # 1. session state (so tools + the fit card can use it directly)
    # 2. agent memory (hidden message, no model call)
    try:
        agent = get_agent(
            resolve_model_name(model_choice()),
            mode(),
            temperature(),
            tools_enabled(),
        )
        config = make_config(session().thread_id)
        preview = truncate(text, 6000)
        agent.update_state(
            config,
            {
                "messages": [
                    HumanMessage(
                        content=(
                            "Remember my CV for this whole session. It is my only source of truth about my "
                            "experience - never invent experience that is not in it.\n\n"
                            f"--- MY CV ---\n{preview}\n--- END CV ---"
                        )
                    )
                ]
            },
        )
        st.session_state.just_saved_cv = True
    except Exception as exc:  # noqa: BLE001 - memory injection is best-effort
        title, hint = classify_error(exc)
        st.warning(f"CV saved to session state, but not into agent memory ({title}). {hint}")


def save_job_to_memory() -> None:
    """Persist the target job description into the active session and agent memory."""
    text = clean_text(str(st.session_state.get("job_draft") or ""), collapse_newlines=False)
    upload = st.session_state.get("job_upload")
    if upload is not None:
        try:
            text = parse_cv_upload(upload)
            st.session_state["job_draft"] = text
        except Exception as exc:  # noqa: BLE001
            title, hint = classify_error(exc)
            st.error(f"{title}. {hint}")
            return

    if not text:
        st.warning("Nothing to save - paste a job posting or upload a file first.")
        return

    session().job_text = text
    for line in text.splitlines():
        if looks_like_role_line(line):
            session().role = extract_role(line)
            break
        if line.strip().lower().startswith(("company:", "**company:", "- **company:")):
            session().company = line.split(":", 1)[-1].replace("*", "").strip()
            break

    try:
        agent = get_agent(
            resolve_model_name(model_choice()),
            mode(),
            temperature(),
            tools_enabled(),
        )
        config = make_config(session().thread_id)
        preview = truncate(text, 6000)
        agent.update_state(
            config,
            {
                "messages": [
                    HumanMessage(
                        content=(
                            "Remember the target job posting for this whole session:\n\n"
                            f"--- TARGET JOB POSTING ---\n{preview}\n--- END TARGET JOB POSTING ---"
                        )
                    )
                ]
            },
        )
        st.session_state.just_saved_job = True
    except Exception as exc:  # noqa: BLE001 - memory injection is best-effort
        title, hint = classify_error(exc)
        st.warning(f"Job posting saved to session state, but not into agent memory ({title}). {hint}")


def start_new_session() -> None:
    """New thread_id + cleared UI history, keeping the CV."""
    session().reset_conversation(keep_cv=True)
    st.session_state.mode = DEFAULT_MODE
    st.session_state.last_mode = DEFAULT_MODE
    st.session_state.pending = None
    st.session_state.just_saved_cv = False
    st.session_state.just_saved_job = False


# ========================================================================== #
# Main chat column
# ========================================================================== #
QUICK_ACTIONS: List[Tuple[str, str, str]] = [
    ("Paste job posting", "I want to apply to a role. Here is the job posting - read it, then research the company and tell me what I should know before I apply.", "coach"),
    ("Analyse CV vs job", "Compare my saved CV against this job posting. Run cv_gap_analyzer, then tell me honestly how strong my application is and what to fix first.", "coach"),
    ("Research the company", "Research this company before we do anything else: their tech stack, what they build, recent news, and what their interview process looks like. Cite every source.", "coach"),
    ("Start mock interview", "Let's do a technical mock interview for this role. Ask me ONE question at a time, wait for my answer, score it, then ask the next one.", "technical"),
    ("Tailor my CV bullets", "Rewrite 4 CV bullets from my saved CV so they answer this job posting. Only reword and reorder what my CV already says, and flag anything I cannot honestly claim.", "coach"),
]

MODE_PLACEHOLDER = {
    "coach": "Paste a job posting, or ask: \"how do I prepare for a Stripe SWE internship?\"",
    "recruiter": "Paste a job posting and I will screen it like a recruiter would…",
    "technical": "Answer one question at a time. Type your answer, or use the voice recorder.",
    "behavioral": "Tell me about a time you shipped something under pressure…",
}


def render_quick_actions() -> None:
    """Five one-click demo entry points without emojis."""
    cols = st.columns(len(QUICK_ACTIONS))
    for col, (label, default_prompt, target_mode) in zip(cols, QUICK_ACTIONS, strict=True):
        with col:
            if st.button(label, use_container_width=True, key=f"qa_{label}"):
                if target_mode != "coach" and target_mode != mode():
                    st.session_state.last_mode = mode()
                    st.session_state.mode = target_mode

                # Auto-sync CV and Job drafts if user entered them but didn't click Save
                if not session().cv_text and (st.session_state.get("cv_draft") or st.session_state.get("cv_upload")):
                    save_cv_to_memory()
                if not session().job_text and (st.session_state.get("job_draft") or st.session_state.get("job_upload")):
                    save_job_to_memory()

                action_prompt = default_prompt
                if label == "Analyse CV vs job":
                    if session().job_text:
                        action_prompt = (
                            "Compare my saved CV against this target job posting:\n\n"
                            f"--- TARGET JOB POSTING ---\n{session().job_text}\n--- END TARGET JOB POSTING ---\n\n"
                            "Run cv_gap_analyzer on my saved CV against this posting, then tell me honestly how strong my application is and what to fix first."
                        )
                    else:
                        action_prompt = (
                            "I want to compare my saved CV against a job posting, but no job posting is saved yet. "
                            "Please ask me to paste or upload the target job posting in the sidebar or here in the chat."
                        )
                elif label == "Paste job posting" and session().job_text:
                    action_prompt = (
                        "I want to apply to this role. Here is the job posting:\n\n"
                        f"--- TARGET JOB POSTING ---\n{session().job_text}\n--- END TARGET JOB POSTING ---\n\n"
                        "Read it, research the company with web_search, and tell me what I should know before I apply."
                    )
                elif label == "Research the company" and session().company:
                    action_prompt = (
                        f"Research {session().company} before we do anything else: their tech stack, what they build, "
                        "recent news, and what their interview process looks like. Cite every source."
                    )
                elif label == "Tailor my CV bullets" and session().job_text:
                    action_prompt = (
                        "Rewrite 4 CV bullets from my saved CV so they answer this target job posting:\n\n"
                        f"--- TARGET JOB POSTING ---\n{session().job_text}\n--- END TARGET JOB POSTING ---\n\n"
                        "Only reword and reorder what my CV already says, and flag anything I cannot honestly claim."
                    )

                st.session_state.pending = (action_prompt, None, None)
                st.rerun()


def render_input_area() -> Tuple[Optional[str], List[Tuple[str, str]], Optional[Tuple[str, str]]]:
    """Text / screenshot / voice inputs without emojis. Returns ``(text, images, audio)``."""
    images: List[Tuple[str, str]] = []
    audio: Optional[Tuple[str, str]] = None
    typed: str = ""

    ta_c, mic_c = st.columns([3, 1])

    with ta_c:
        with st.expander("Attach a screenshot of the posting (PNG / JPG)", expanded=False):
            shot = st.file_uploader(
                "Job posting screenshot",
                type=["png", "jpg", "jpeg"],
                key="shot",
                label_visibility="collapsed",
            )
            if shot is not None:
                b64, mime = file_to_b64(shot)
                images.append((b64, mime))
                st.image(b64, width=280, caption=f"{shot.name} · sent as base64 to Gemini")
                if st.button("Describe this posting", key="shot_submit"):
                    typed = "This is a screenshot of a job posting. Read it carefully, then research the company and summarise the role, the stack and what I should emphasise."

    with mic_c:
        with st.expander("Voice answer", expanded=False):
            rec = st.audio_input("Record", key="voice", label_visibility="collapsed")
            if rec is not None:
                b64, mime = file_to_b64(rec, default_mime="audio/wav")
                audio = (b64, mime)
                st.audio(rec, format=guess_mime(rec.name, "audio/wav").split("/")[-1])
                if st.button("Send voice answer", key="voice_submit"):
                    typed = "This is my voice answer to your last question. Transcribe it, then score my answer with score_answer and continue the interview."

    typed_from_widget = st.chat_input(
        MODE_PLACEHOLDER.get(mode(), "Ask me anything about this role…"),
        key="chat_input",
    )
    text = typed_from_widget or typed or None
    return text, images, audio


def render_history() -> None:
    """Replay the stored chat turns cleanly without emojis."""
    data = session()
    if not data.chat:
        st.info(
            "**Quick Start Guide**\n\n"
            "1. Upload or paste your CV in the sidebar, then click **Save CV to memory**\n"
            "2. Paste the target job posting in the sidebar (or in chat) and click **Save Job Posting**\n"
            "3. Click **Analyse CV vs job** to inspect match score and missing requirements\n"
            "4. Click **Research the company** to gather real-time intelligence with sources\n"
            "5. Click **Tailor my CV bullets** to reword experience for this specific role\n"
            "6. Click **Start mock interview** to test your technical or STAR responses\n"
            "7. Use **Recall test** and **Download session report** to export your progress"
        )
        return

    for turn in data.chat:
        if turn.hidden:
            continue
        with st.chat_message("user" if turn.role == "user" else "assistant"):
            if turn.content:
                st.markdown(turn.content)
            for idx, img_b64 in enumerate(turn.images):
                st.image(img_b64, width=260, caption=f"attachment {idx + 1}")
            if turn.audio:
                st.audio(b64_to_bytes(turn.audio), format="audio/wav")
            st.caption(f"{turn.timestamp}" + (f" · {turn.model}" if turn.model else ""))


def build_user_message(
    text: str,
    images: List[Tuple[str, str]],
    audio: Optional[Tuple[str, str]],
):
    """Build the multimodal ``HumanMessage``."""
    image_blocks: List[Tuple[str, str]] = list(images or [])
    blocks: List[Dict[str, Any]] = []
    if text:
        blocks.append({"type": "text", "text": text})

    if not blocks and not image_blocks and not audio:
        blocks.append({"type": "text", "text": "Please continue - review what I sent above and take the next step."})

    for b64, mime in image_blocks:
        blocks.append({"type": "image", "base64": b64, "mime_type": mime})

    if audio:
        blocks.append({"type": "audio", "base64": audio[0], "mime_type": audio[1]})
        if not text:
            blocks.insert(
                0,
                {
                    "type": "text",
                    "text": "Listen to this voice answer. Transcribe it faithfully, then continue the interview loop (score_answer first, next question second).",
                },
            )

    if len(blocks) == 1 and blocks[0]["type"] == "text":
        return HumanMessage(content=blocks[0]["text"])
    return HumanMessage(content=blocks)


def run_agent_turn(text: str, images: List[Tuple[str, str]], audio: Optional[Tuple[str, str]]) -> None:
    """Stream one agent turn, rendering tool calls live inside ``st.status`` without emojis."""
    data = session()
    model_name = resolve_model_name(model_choice())
    active_mode = mode()
    temp = temperature()

    image_blocks: List[Tuple[str, str]] = list(images or [])
    audio_block: Optional[Tuple[str, str]] = tuple(audio) if audio else None  # type: ignore[assignment]
    text = text or ""

    # Store the user turn first so it renders even if the model call fails.
    data.chat.append(
        ChatTurn(
            role="user",
            content=text or ("[screenshot of a job posting]" if image_blocks else "[voice answer]"),
            images=[b64 for b64, _ in image_blocks],
            audio=audio_block[0] if audio_block else None,
        )
    )

    # Auto-sync CV and Job drafts if session is missing them
    if not data.cv_text and (st.session_state.get("cv_draft") or st.session_state.get("cv_upload")):
        save_cv_to_memory()
    if not data.job_text and (st.session_state.get("job_draft") or st.session_state.get("job_upload")):
        save_job_to_memory()

    # Intelligent capture of job posting from chat text
    if text and not data.job_text:
        low = text.lower()
        has_job_markers = any(k in low for k in (
            "requirement", "responsibilit", "qualifi", "looking for", "job description",
            "internship", "about the role", "stack:", "we are hiring", "minimum qualifications"
        ))
        if len(text) > 80 and (has_job_markers or len(text) > 250):
            data.job_text = clean_text(text, collapse_newlines=False)
            for line in text.splitlines():
                if looks_like_role_line(line):
                    data.role = extract_role(line)
                    break
                if line.strip().lower().startswith(("company:", "**company:")):
                    data.company = line.split(":", 1)[-1].replace("*", "").strip()
                    break

    try:
        agent = get_agent(model_name, active_mode, temp, tools_enabled())
    except Exception as exc:  # noqa: BLE001
        title, hint = classify_error(exc)
        st.error(f"{title}. {hint}")
        data.chat.append(ChatTurn(role="assistant", content=f"_I could not start: {title}._"))
        return

    message = build_user_message(text, image_blocks, audio_block)
    config = make_config(data.thread_id)

    final_text = ""
    error: Optional[BaseException] = None

    try:
        with active_session(data):  # tools can save notes / sources / scoreboard
            with st.status("Working…", expanded=True) as status:
                status.write("Sending to **%s** in **%s** mode…" % (model_name, MODE_LABELS[active_mode]))
                try:
                    for event in stream_turn(agent, message, config):
                        kind = event["kind"]

                        if kind == "tool_call":
                            data.tool_calls.append(event["record"])
                            status.write(
                                f"**Step {event['step']}** · `{event['name']}`\n\n"
                                f"`{truncate(event['args_preview'], 220)}`"
                            )

                        elif kind == "tool_result":
                            record: ToolCallRecord = event["record"]
                            latency = event["latency_s"]
                            timing = f"finished in {latency:.1f}s" if latency is not None else "finished"
                            badge_text = "[OK]" if record.ok else "[ALERT]"
                            status.write(
                                f"**{badge_text}** `{record.name}` "
                                f"{timing}\n\nResult: `{truncate(event['raw'], 240)}`"
                            )
                            patched = False
                            for existing in reversed(data.tool_calls):
                                if existing is not record and existing.name == record.name and existing.result_preview == "running…":
                                    existing.result_preview = record.result_preview
                                    existing.latency_s = latency
                                    existing.ok = record.ok
                                    existing.args = record.args
                                    patched = True
                                    break
                            if not patched:
                                data.tool_calls.append(record)
                            capture_tool_side_effects(data, record.name, event["raw"])

                        elif kind == "final":
                            final_text = event["text"]
                            status.update(
                                label=f"Done · {event['steps']} tool call(s)",
                                state="complete",
                                expanded=False,
                            )
                except Exception as exc:  # noqa: BLE001
                    error = exc
                    title, hint = classify_error(exc)
                    status.update(label=f"Failed · {title}", state="error")
    except Exception as exc:  # noqa: BLE001
        error = exc
        title, hint = classify_error(exc)
        st.error(f"**{title}.** {hint}")

    # ---- final answer ----------------------------------------------------
    if not final_text:
        try:
            state = agent.get_state(config)
            messages = (state.values or {}).get("messages", []) if state.values else []
            for msg in reversed(messages or []):
                if type(msg).__name__ == "AIMessage":
                    final_text = extract_text(msg.content)
                    break
        except Exception:  # noqa: BLE001
            final_text = ""

    if error is not None and not final_text:
        title, hint = classify_error(error)
        final_text = f"_The run failed before the model answered: {title}._"

    data.last_model_used = model_name
    data.chat.append(ChatTurn(role="assistant", content=final_text or "_No answer was produced._", model=model_name))
    capture_answer_side_effects(data, final_text)

    with st.chat_message("assistant"):
        st.markdown(final_text or "_No answer was produced._")
        st.caption(f"{model_name} · {MODE_LABELS[active_mode]} · temp {temp}")


# ========================================================================== #
# Job-Fit card + interview scoreboard
# ========================================================================== #
def chips(items: List[str], kind: str) -> None:
    """Render skills as coloured chips without emojis."""
    if not items:
        st.markdown("<span class='ihc-muted'>none</span>", unsafe_allow_html=True)
        return
    rendered = "".join(
        f"<span class='ihc-chip ihc-chip-{kind}'>{html.escape(str(item))}</span>" for item in items[:60]
    )
    st.markdown(rendered, unsafe_allow_html=True)


def render_fit_card() -> None:
    """Prominent card built from the deterministic ``cv_gap_analyzer`` output.

    Rendered inside a collapsible expander so it can be hidden during interviews.
    """
    data = session()
    if not data.fit:
        return
    fit = data.fit
    score = int(fit.get("fit_score", 0) or 0)
    color = "#22c55e" if score >= 70 else "#f59e0b" if score >= 45 else "#ef4444"
    verdict = "STRONG FIT" if score >= 70 else "VIABLE FIT" if score >= 45 else "STRETCH ROLE"

    with st.expander(f"Job-Fit Analysis — {score}/100 {verdict}", expanded=True):
        st.markdown(
            f"""
<div class="ihc-card" style="margin:0">
  <div class="ihc-card-title">cv_gap_analyzer · deterministic</div>
  <div style="display:flex;align-items:baseline;gap:1.1rem;flex-wrap:wrap;">
    <div class="ihc-kpi" style="color:{color}">{score}<span style="font-size:1.05rem;opacity:.65">/100</span></div>
    <div style="font-weight:700;letter-spacing:0.03em;">{verdict}</div>
    <div class="ihc-muted">{fit.get('total_required', 0)} required skills detected</div>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )
        st.progress(min(1.0, score / 100))

        c1, c2, c3 = st.columns([1.1, 1.4, 1.4])
        with c1:
            st.markdown("**Matched**")
            chips(list(fit.get("matched", []) or []), "ok")
        with c2:
            st.markdown("**Missing**")
            chips(list(fit.get("missing", []) or []), "miss")
        with c3:
            st.markdown("**Nice-to-have missing**")
            chips(list(fit.get("nice_to_have_missing", []) or []), "nice")

        if fit.get("error"):
            st.warning(fit["error"])
        st.caption(
            "Score = 80% required skills matched + 20% nice-to-haves matched, computed from a curated skill vocabulary and regexes."
        )


def render_scoreboard() -> None:
    """Running interview scoreboard without emojis."""
    data = session()
    active_mode = mode()
    if active_mode not in INTERVIEW_MODES:
        return
    if not data.interview_scores:
        st.caption("No answers scored yet - answer the first question and the scoreboard will fill up.")
        return

    with st.container(border=True):
        st.markdown("#### Interview Scoreboard")
        st.caption("Default rubric when the agent does not supply its own:")
        st.caption(f"_{DEFAULT_RUBRICS.get(active_mode, DEFAULT_RUBRICS['general'])}_")
        scores = [float(item.get("score", 0) or 0) for item in data.interview_scores]
        st.progress(
            min(1.0, (scores[-1] / 10) if scores else 0.0),
            text=f"Latest answer: {scores[-1]:.0f}/10" if scores else "No answers yet",
        )
        avg = data.average_score() or 0.0
        m1, m2, m3 = st.columns([1, 1, 2.2])
        m1.metric("Answers scored", len(scores))
        m2.metric("Average", f"{avg:.1f}/10", delta=f"{(scores[-1] - avg):+.1f} vs avg")
        with m3:
            try:
                st.line_chart(
                    {"score": scores, "average": [avg] * len(scores)},
                    height=140,
                )
            except Exception:  # noqa: BLE001
                st.caption(" ".join(f"{s:.0f}" for s in scores))

        for item in data.interview_scores:
            score = int(item.get("score", 0) or 0)
            tone_class = "ihc-badge-high" if score >= 7 else "ihc-badge-mid" if score >= 4 else "ihc-badge-low"
            tone_label = "HIGH" if score >= 7 else "MID" if score >= 4 else "LOW"
            with st.expander(
                f"[{tone_label}] Q{item.get('step', '?')} · {score}/10 · {truncate(str(item.get('question', '')), 80)}",
                expanded=False,
            ):
                st.markdown(f"**{truncate(str(item.get('question', '')), 300)}**")
                if item.get("strengths"):
                    st.markdown("**Strengths**")
                    for s in item["strengths"]:
                        st.markdown(f"- {s}")
                if item.get("improvements"):
                    st.markdown("**Improvements**")
                    for s in item["improvements"]:
                        st.markdown(f"- {s}")
                if item.get("sample_better_answer"):
                    st.markdown("**A stronger answer**")
                    st.markdown(f"> {item['sample_better_answer']}")


# ========================================================================== #
# Glass-box panel
# ========================================================================== #
def render_glassbox() -> None:
    """The four tabs that make the LangChain concepts visible, without emojis."""
    data = session()
    tab_model, tab_tools, tab_sources, tab_memory = st.tabs(
        ["Model & Prompt", "Tool calls", "Sources", "Memory"]
    )

    # ---- 1. Model & prompt ----------------------------------------------
    with tab_model:
        active_mode = mode()
        model_name = resolve_model_name(model_choice())
        c1, c2 = st.columns(2)
        c1.metric("Model in use", "Fast" if model_choice() == "Fast" else "Deep")
        c2.metric("Temperature", temperature())
        st.code(
            "ChatGoogleGenerativeAI(\n"
            f"    model={model_name!r},\n"
            f"    temperature={temperature()},\n"
            ")",
            language="python",
        )

        diff_note = prompt_diff(str(st.session_state.get("last_mode", active_mode)), active_mode)
        if diff_note:
            st.info(diff_note)
            if st.button("Acknowledge prompt change", key="ack_prompt"):
                st.session_state.last_mode = active_mode
                st.rerun()

        st.markdown(f"**Mode:** {MODE_LABELS[active_mode]}")
        st.caption(MODE_HINTS[active_mode])
        st.text_area(
            "Active system prompt",
            value=get_prompt(active_mode),
            height=340,
            disabled=True,
            key=f"prompt_{active_mode}",
            help="Read-only. This is exactly what create_agent(system_prompt=...) received.",
        )
        st.caption(
            "Tools wired into this agent: "
            + (", ".join(f"`{n}`" for n in TOOL_NAMES) if tools_enabled() else "_disabled_")
        )
        with st.expander("What each tool is for"):
            for name, blurb in TOOL_BLURBS.items():
                st.markdown(f"`{name}` — {blurb}")

    # ---- 2. Tool calls ---------------------------------------------------
    with tab_tools:
        st.caption(f"{len(data.tool_calls)} call(s) this session. Arguments come from the LLM; results come back as ToolMessages.")
        if not data.tool_calls:
            st.info("No tool calls yet. Ask a question that needs research, a fit score or an answer graded.")
        for record in reversed(data.tool_calls):
            latency = f"{record.latency_s:.1f}s" if record.latency_s is not None else "…"
            badge = "[OK]" if record.ok else "[ALERT]"
            st.markdown(
                f"<div class='ihc-tl'><b>{badge} Step {record.step} · <code>{record.name}</code></b> · {latency} · {record.created_at}</div>",
                unsafe_allow_html=True,
            )
            with st.expander("arguments", expanded=False):
                st.json(record.args, expanded=False)
                st.code(truncate(record.result_preview, 1500) or "…", language="json")

    # ---- 3. Sources ------------------------------------------------------
    with tab_sources:
        st.caption(f"{len(data.sources)} unique source(s) collected by `web_search` (deduplicated by URL).")
        if not data.sources:
            st.info("No sources yet - run **Research the company** or paste a posting and the agent will search for it.")
        seen: set[str] = set()
        for src in data.sources:
            if src["url"] in seen:
                continue
            seen.add(src["url"])
            host = src["url"].split("//")[-1].split("/")[0]
            st.markdown(
                f"<div class='ihc-src'><b>[Source]</b> <a href='{html.escape(src['url'], quote=True)}' "
                f"target='_blank' rel='noopener noreferrer'>{html.escape(src['title'])}</a><br>"
                f"<span class='ihc-muted'>{html.escape(host)}</span></div>",
                unsafe_allow_html=True,
            )
        if not tavily_api_key():
            st.warning("TAVILY_API_KEY is missing, so `web_search` returns a friendly error instead of results.")

    # ---- 4. Memory -------------------------------------------------------
    with tab_memory:
        st.caption(
            f"`InMemorySaver` · thread `{data.thread_id[:8]}` · {len(data.chat)} rendered turn(s). "
            "State below is read straight from the checkpoint with `agent.get_state(config)`."
        )
        st.markdown("**Notes saved by save_note**")
        if data.notes:
            for key, value in data.notes.items():
                st.markdown(f"- `{key}` — {value}")
        else:
            st.caption("_no notes yet_")

        st.markdown("**Thread messages**")
        messages = read_thread_messages()
        if not messages:
            st.caption("_the checkpoint is empty - say hello to start writing memory_")
        with st.container(height=420):
            for role, content in messages:
                role_label = {"human": "User", "ai": "Assistant", "system": "System", "tool": "Tool"}.get(role, role.capitalize())
                st.markdown(
                    f"<div class='ihc-tl'><b>[{role_label}]</b><br>"
                    f"<span class='ihc-muted'>{truncate(content.replace(chr(10), ' '), 320)}</span></div>",
                    unsafe_allow_html=True,
                )
        st.caption("Resets when the server restarts - see the README for a SQLite/Postgres checkpointer.")

    # ---- download report (also in the sidebar) --------------------------
    st.download_button(
        "Download session report (Markdown)",
        data=build_session_report(
            data,
            model_name=resolve_model_name(model_choice()),
            mode_label=MODE_LABELS.get(mode(), mode()),
            temperature=temperature(),
            search_enabled=tools_enabled() and bool(tavily_api_key()),
        ),
        file_name=report_filename(data),
        mime="text/markdown",
        use_container_width=True,
        key="download_report_glassbox",
    )


def read_thread_messages() -> List[Tuple[str, str]]:
    """Read the checkpointed messages for the active thread (best effort)."""
    try:
        agent = get_agent(
            resolve_model_name(model_choice()),
            mode(),
            temperature(),
            tools_enabled(),
        )
        state = agent.get_state(make_config(session().thread_id))
        values = state.values or {}
        out: List[Tuple[str, str]] = []
        for msg in values.get("messages", []) or []:
            role = {
                "HumanMessage": "human",
                "AIMessage": "ai",
                "SystemMessage": "system",
                "ToolMessage": "tool",
            }.get(type(msg).__name__, "other")
            out.append((role, extract_text(getattr(msg, "content", "")) or "(no text)"))
        return out
    except Exception:  # noqa: BLE001
        return []


# ========================================================================== #
# Setup screen
# ========================================================================== #
def render_setup_screen() -> None:
    """Friendly first-run screen without emojis."""
    st.title("Internship Hunter Copilot")
    st.warning("**GOOGLE_API_KEY is not configured.** The agent cannot run without it.")
    st.markdown(
        """
### Local setup
1. Create a file named `.env` next to `app.py`:
```
GOOGLE_API_KEY=your-gemini-key
TAVILY_API_KEY=your-tavily-key
# Optional: pin the models your key can access.
# GEMINI_FAST_MODEL=gemini-2.5-flash
# GEMINI_DEEP_MODEL=gemini-flash-latest
```
2. Restart: `streamlit run app.py`

### Deployment
* **Streamlit Community Cloud** -> *Settings -> Secrets*:
```
GOOGLE_API_KEY = "..."
TAVILY_API_KEY = "..."
```
* **Hugging Face Spaces** -> *Variables and secrets* with the same two names.
"""
    )
    if st.button("I added the key - reload"):
        st.rerun()


# ========================================================================== #
# Entry point
# ========================================================================== #
def main() -> None:
    init_state()
    inject_css()

    if not google_api_key():
        render_setup_screen()
        st.stop()

    render_sidebar()

    st.markdown(
        """
<div class="dream-shell">
  <div class="dream-topbar">
    <div class="dream-navgroup">
      <span class="dream-brand">INTERNSHIP HUNTER</span>
      <span class="dream-pill active">Career</span>
      <span class="dream-pill">Interview prep</span>
      <span class="dream-pill">CV match</span>
      <span class="dream-pill">Research</span>
    </div>
    <div class="dream-mini-actions" aria-hidden="true">
      <span class="dream-mini-btn">⌕</span>
      <span class="dream-mini-btn">↗</span>
    </div>
  </div>

  <div class="dream-hero">
    <div class="dream-copy">
      <div class="dream-kicker">Your career, in motion</div>
      <h1 class="dream-title">A New Way To<em>Launch Your Career</em></h1>
    </div>
  </div>

  <div class="dream-footnote">CV MATCH &nbsp;·&nbsp; INTERVIEWS &nbsp;·&nbsp; RESEARCH</div>
  <div class="dream-badge">
    <div class="dream-badge-avatar">IH</div>
    <div class="dream-badge-copy">
      <strong>Your career copilot</strong>
      <small>One workspace. Every next step.</small>
    </div>
    <div class="dream-arrow">↗</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.85, 1], gap="large")

    with left:
        active_mode = mode()
        st.markdown(
            f"""
<div class="ihc-header-wrap">
  <h3 class="ihc-title">Internship Hunter Copilot</h3>
  <div class="ihc-subtitle">Mode <b>{MODE_LABELS[active_mode]}</b> · model <code>{resolve_model_name(model_choice())}</code> · thread <code>{session().thread_id[:8]}</code></div>
</div>
""",
            unsafe_allow_html=True,
        )

        # Show a clear banner in interview modes so the user knows to type below
        if active_mode in INTERVIEW_MODES:
            st.info(
                "**Interview Mode** — Read the question above, then type your answer in the chat box below. "
                "One question at a time. Your scores appear in the panel on the right."
            )

        render_quick_actions()

        pending = st.session_state.pop("pending", None)
        if pending is not None:
            text, images, audio = pending
        else:
            text, images, audio = render_input_area()

        if st.session_state.get("just_saved_cv"):
            st.session_state.just_saved_cv = False
            st.success("CV stored in session state and injected into the agent memory as a hidden message.")

        if st.session_state.get("just_saved_job"):
            st.session_state.just_saved_job = False
            st.success("Target job description saved to session and ready for gap analysis.")

        # Chat history and live turn — no cards between these and the input
        render_history()
        if text or images or audio:
            run_agent_turn(text or "", images, audio)

    with right:
        # Fit card and scoreboard live here so they never obstruct the chat
        active_mode = mode()
        if session().fit or (active_mode in INTERVIEW_MODES and session().interview_scores):
            st.markdown("#### Analysis & Scores")
        render_fit_card()
        render_scoreboard()
        if session().fit or (active_mode in INTERVIEW_MODES and session().interview_scores):
            st.divider()
        render_glassbox()


if __name__ == "__main__":
    main()