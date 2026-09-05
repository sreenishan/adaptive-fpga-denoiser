"""AdaptiveDenoise — adaptive image denoising platform (AI classification + FPGA RTL).

Run with:
    streamlit run app/streamlit_app.py

────────────────────────────────────────────────────────────────────────────
HONESTY CONTRACT
────────────────────────────────────────────────────────────────────────────
This UI shows measured values or it shows nothing. Specifically:

  · Every dashboard/analytics number is derived from runs performed in this
    session. There are no illustrative totals, no seeded usage counters and
    no invented trends. Zero runs renders an empty state, not a placeholder
    figure.
  · Timings are Python/CPU wall-clock. No FPGA board is attached, so no
    hardware throughput, utilisation or temperature is displayed anywhere.
  · MSE/PSNR/SSIM require a clean reference. Without one the pipeline returns
    metrics=None and the UI says why (see PipelineResult.metrics_note).
  · Confidence is the classifier's or it is absent. A manual choice shows
    "n/a", never 100%.
  · Surfaces with no backend (Projects, Billing) state that plainly instead
    of rendering fabricated records, and expose no control that pretends to
    perform an action it cannot.
"""

from __future__ import annotations

import base64
import html
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from denoising.config import (  # noqa: E402
    CLASSES,
    load_dataset_config,
    load_hardware_config,
    load_inference_config,
)
from denoising.dataset import synthetic_sources  # noqa: E402
from denoising.filters import FILTER_FOR_CLASS, estimate_noise_variance  # noqa: E402
from denoising.noise import (  # noqa: E402
    add_gaussian_noise,
    add_salt_pepper_noise,
    add_speckle_noise,
)
from denoising.pipeline import process_image  # noqa: E402
from denoising.preprocessing import to_grayscale  # noqa: E402

try:
    from denoising.model.inference import load_classifier as _load_classifier
    _CLASSIFIER_AVAILABLE = True
except ImportError:
    _CLASSIFIER_AVAILABLE = False

st.set_page_config(
    page_title="AdaptiveDenoise",
    page_icon="◈",
    layout="wide",
    # "auto" keeps the rail open on a wide screen and collapses it on a narrow
    # one. With "expanded" the sidebar opened over the content on a phone,
    # taking 256px of a 375px viewport — 68% of the screen — so the page behind
    # it was unreadable until the user found the collapse control.
    initial_sidebar_state="auto",
)

# ═══════════════════════════════════════════════════════════════════════════
# DESIGN TOKENS
# ═══════════════════════════════════════════════════════════════════════════

T = {
    # surfaces — deep navy-slate (more premium than pure black)
    "bg":         "#070B14",
    "surface":    "#0D1117",
    "elevated":   "#161C2B",
    "hover":      "#1C2434",
    # borders — blue-tinted for a cooler, more tech feel
    "border":     "rgba(99,130,170,0.14)",
    "border_st":  "rgba(99,130,170,0.27)",
    # text — slightly cool whites (softer on dark backgrounds)
    "text":       "#E6EDF3",
    "text_2":     "#8B949E",
    "text_3":     "#484F58",
    # accent — teal
    "accent":     "#00C9A7",
    "accent_hi":  "#2ED9B8",
    "accent_dim": "rgba(0,201,167,0.12)",
    # semantic
    "ok":         "#3FB950",
    "warn":       "#D29922",
    "err":        "#F85149",
    "info":       "#58A6FF",
    "violet":     "#BC8CFF",
    "cyan":       "#39D5E8",
}

# Noise class → (accent, label)
NOISE_META = {
    "clean":       (T["ok"],     "Clean"),
    "salt_pepper": (T["warn"],   "Salt & Pepper"),
    "gaussian":    (T["accent"], "Gaussian"),
    "speckle":     (T["violet"], "Speckle"),
}

FILTER_META = {
    "bypass":   (T["text_3"], "Bypass",           "Passes the image through untouched."),
    "median":   (T["accent"], "Median filter",    "Rank filter — removes impulse outliers without blurring edges."),
    "gaussian": (T["violet"], "Gaussian filter",  "Binomial smoothing — suppresses broad additive noise."),
    "wiener":   (T["ok"],     "Wiener filter",    "Adaptive — attenuates by local SNR, preserving detail."),
}


def _label(c: str) -> str:
    return NOISE_META.get(c, (T["text_3"], c))[1]


def _ncolor(c: str) -> str:
    return NOISE_META.get(c, (T["text_3"], c))[0]


def _flabel(f: str) -> str:
    return FILTER_META.get(f, (T["text_3"], f, ""))[1]


def _fcolor(f: str) -> str:
    return FILTER_META.get(f, (T["text_3"], f, ""))[0]


# ═══════════════════════════════════════════════════════════════════════════
# GLOBAL STYLESHEET
# ═══════════════════════════════════════════════════════════════════════════

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;450;500;600;700&display=swap');

:root {{
  --bg:{T['bg']}; --surface:{T['surface']}; --elevated:{T['elevated']}; --hover:{T['hover']};
  --border:{T['border']}; --border-st:{T['border_st']};
  --text:{T['text']}; --text-2:{T['text_2']}; --text-3:{T['text_3']};
  --accent:{T['accent']}; --accent-hi:{T['accent_hi']}; --accent-dim:{T['accent_dim']};
  --ok:{T['ok']}; --warn:{T['warn']}; --err:{T['err']}; --info:{T['info']};
  --s1:4px; --s2:8px; --s3:12px; --s4:16px; --s5:20px; --s6:24px; --s8:32px; --s10:40px; --s12:48px;
  --r-xs:6px; --r-sm:8px; --r-md:10px; --r-lg:14px; --r-xl:18px; --r-full:100px;

  /* Type scale. Seven steps, no half-pixels — the file previously carried
     sixteen sizes including 10.5/11.5/12.5/13.5, which is an eyeballed ramp
     rather than a scale. Each step is a visible jump from its neighbour. */
  --fs-0:11px;   /* uppercase eyebrows, badges, table headers */
  --fs-1:12px;   /* meta, captions, helper text */
  --fs-2:13px;   /* table cells, key-value rows, dense body */
  --fs-3:14px;   /* body */
  --fs-4:16px;   /* card titles */
  --fs-5:20px;   /* section headings */
  --fs-6:28px;   /* page title */
  /* Weights. Three, not the nine (450/500/550/600/640/650/700/750/800) that
     were in use — adjacent steps like 640 vs 650 are indistinguishable and
     communicate no hierarchy. */
  --fw-normal:500; --fw-medium:600; --fw-bold:700;
  --glow: 0 0 28px rgba(0,201,167,0.20); --glow-sm: 0 0 14px rgba(0,201,167,0.13);
  --shadow-sm: 0 1px 2px rgba(0,0,0,0.5), 0 2px 6px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.03);
  --shadow-md: 0 4px 16px rgba(0,0,0,0.55), 0 2px 4px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.04);
  --shadow-lg: 0 8px 32px rgba(0,0,0,0.6), 0 4px 12px rgba(0,0,0,0.4);
  --ease:cubic-bezier(0.32,0.72,0,1);
  --font:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',system-ui,sans-serif;
  --mono:'SF Mono',ui-monospace,'JetBrains Mono','Cascadia Code',Consolas,monospace;
}}

/* ── base ── */
html, body, [class*="css"], .stApp {{ font-family: var(--font); -webkit-font-smoothing: antialiased; }}
.stApp {{ background: var(--bg); color: var(--text); }}
[data-testid="stMain"] .block-container,
.main .block-container {{ padding: var(--s8) var(--s10) var(--s12); max-width: 1480px; }}
[data-testid="stHeader"] {{ background: transparent !important; }}
#MainMenu, footer {{ visibility: hidden; }}

h1,h2,h3,h4 {{ color: var(--text); letter-spacing: -0.026em; font-weight: 700; }}
[data-testid="stMain"] p {{ color: var(--text-2); font-size: var(--fs-3); line-height: 1.65; }}
.tnum {{ font-variant-numeric: tabular-nums; font-feature-settings:'tnum' 1; }}

/* Streamlit wraps every button label in its own <p>, so the blanket rule above
   used to repaint it --text-2. On the teal primary that was grey #8B949E on
   #00A88C — a 1.02:1 contrast ratio, i.e. an invisible label. The button
   already carries the correct colour; the label just has to inherit it. */
[data-testid="stMain"] .stButton > button p,
[data-testid="stMain"] .stDownloadButton > button p,
[data-testid="stMain"] .stFormSubmitButton > button p {{
  color: inherit !important; font-size: inherit !important;
  font-weight: inherit !important; line-height: 1.2 !important;
}}

/* ── focus: visible everywhere, keyboard-first ── */
*:focus-visible {{
  outline: 2px solid var(--accent) !important; outline-offset: 2px !important;
  border-radius: var(--r-sm);
}}

/* ══ SIDEBAR ══ */
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div,
[data-testid="stSidebarContent"] {{ background: linear-gradient(180deg,#0A0E18 0%,#070B14 100%) !important; }}
/* Streamlit renders the sidebar as a resizable section carrying an INLINE
   width (300px default). Only `width` with !important overrides an inline
   declaration — min/max-width alone are silently outranked. */
[data-testid="stSidebar"] {{
  border-right: 1px solid rgba(99,130,170,0.15) !important;
  width: 256px !important; min-width: 256px !important; max-width: 256px !important;
  flex: 0 0 256px !important;
}}
/* The drag-to-resize handle is a control that cannot do anything: the rail is
   pinned to a fixed width with !important above, so a drag is overridden the
   moment it lands. It carries NO data-testid in this build, so the rule that
   named stSidebarResizeHandle matched nothing and the handle stayed live.
   Selected instead as the sidebar's one direct child that is not the content
   wrapper — structural, and it survives an emotion-class rename. */
[data-testid="stSidebar"] > div:not([data-testid="stSidebarContent"]) {{
  display: none !important;
}}
[data-testid="stSidebar"] > div:first-child {{ padding-top: 0 !important; }}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span:not(.nav-label):not(.sb),
[data-testid="stSidebar"] label {{ color: var(--text-2) !important; }}
[data-testid="stSidebar"] .stCaption p {{ color: var(--text-3) !important; }}

[data-testid="stSidebarCollapseButton"] button,
[data-testid="stSidebar"] button[kind="header"] {{
  color: var(--text-3) !important; background: transparent !important;
  opacity: 0.6 !important; border-radius: var(--r-sm) !important;
  transition: all 160ms var(--ease) !important;
}}
[data-testid="stSidebarCollapseButton"] button:hover,
[data-testid="stSidebar"] button[kind="header"]:hover {{
  color: var(--text) !important; opacity: 1 !important;
  background: rgba(255,255,255,0.06) !important;
}}

/* exact nav grid */
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{ gap: 0 !important; }}
[data-testid="stSidebar"] .element-container {{ margin: 0 !important; }}
/* Streamlit ships stMarkdownContainer with margin-bottom:-16px, which would
   collapse each 42px nav block to 26px and de-register the button overlay. */
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"]:has(.nav-row) {{ margin-bottom: 0 !important; }}

[data-testid="stSidebar"] .nav-row {{
  display:flex; align-items:center; gap:11px; height:40px; box-sizing:border-box;
  padding:0 var(--s3); margin:0 var(--s3) 2px; border-radius:var(--r-md);
  border:1px solid transparent; transition: background 170ms var(--ease), border-color 170ms var(--ease);
}}
[data-testid="stSidebar"] .nav-row svg {{
  width:17px; height:17px; flex-shrink:0; fill:none; stroke:currentColor;
  stroke-width:1.75; stroke-linecap:round; stroke-linejoin:round;
}}
[data-testid="stSidebar"] .nav-label {{
  font-size:var(--fs-3); font-weight:500; letter-spacing:-0.006em; line-height:1;
  white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
}}
[data-testid="stSidebar"] .nav-idle {{ color:#505070; }}
[data-testid="stSidebar"] .nav-idle .nav-label {{ color:#505070; }}
[data-testid="stSidebar"] .nav-idle svg {{ opacity:.65; }}
[data-testid="stSidebar"] .nav-active {{
  position:relative;
  background: linear-gradient(96deg,rgba(0,201,167,.18),rgba(0,201,167,.07));
  border-color: rgba(0,201,167,.32);
  box-shadow: inset 0 1px 0 rgba(255,255,255,.05), 0 0 16px rgba(0,201,167,.10), var(--shadow-sm);
}}
[data-testid="stSidebar"] .nav-active .nav-label {{ color:#E0FBF5; font-weight:700; }}
[data-testid="stSidebar"] .nav-active svg {{ color: var(--accent-hi); filter:drop-shadow(0 0 5px rgba(0,201,167,.45)); }}
[data-testid="stSidebar"] .nav-active::before {{
  content:""; position:absolute; left:-7px; top:9px; bottom:9px; width:3px;
  border-radius:4px; background: linear-gradient(180deg,#2AEBB8,#00927A);
  box-shadow: 0 0 8px rgba(0,201,167,.7);
}}

/* transparent hit target laid exactly over each row */
[data-testid="stSidebar"] .stButton {{ margin-top:-42px !important; margin-bottom:2px !important; position:relative; z-index:3; }}
[data-testid="stSidebar"] .stButton > button {{
  height:40px !important; width:calc(100% - 24px) !important; margin:0 var(--s3) !important;
  padding:0 !important; background:transparent !important; border:none !important;
  box-shadow:none !important; border-radius:var(--r-md) !important;
  color:transparent !important; font-size:0 !important;
  transition: background 170ms var(--ease) !important;
}}
[data-testid="stSidebar"] .stButton > button:hover {{ background: rgba(255,255,255,.045) !important; }}
[data-testid="stSidebar"] .stButton > button p {{ color:transparent !important; font-size:0 !important; }}

[data-testid="stSidebar"] .nav-group {{
  font-size:var(--fs-0); font-weight:700; letter-spacing:.14em; text-transform:uppercase;
  color:rgba(99,130,170,.55); padding:0 var(--s5); margin:20px 0 6px;
}}
[data-testid="stSidebar"] .nav-group-first {{ margin-top:var(--s1); }}

/* ══ BUTTONS ══ */
[data-testid="stMain"] .stButton > button,
[data-testid="stMain"] .stDownloadButton > button {{
  /* Never break a label mid-phrase: "＋ New / processing" across two lines
     reads as two controls at a glance. */
  white-space:nowrap;
  height:38px; border-radius:var(--r-md) !important; font-family:var(--font) !important;
  font-size:var(--fs-3) !important; font-weight:var(--fw-medium) !important; letter-spacing:-0.004em !important;
  transition: all 170ms var(--ease) !important;
}}
/* primary */
[data-testid="stMain"] .stButton > button[kind="primary"],
[data-testid="stMain"] .stDownloadButton > button {{
  background: linear-gradient(135deg,#00DEB9 0%,#00A88C 100%) !important; color:#020F0B !important;
  border:1px solid rgba(0,201,167,.35) !important;
  box-shadow: 0 2px 6px rgba(0,0,0,.4), 0 0 16px rgba(0,201,167,.22) !important;
  font-weight:700 !important; letter-spacing:-.005em !important;
}}
[data-testid="stMain"] .stButton > button[kind="primary"]:hover,
[data-testid="stMain"] .stDownloadButton > button:hover {{
  background: linear-gradient(135deg,#2AEBB8 0%,#00C9A7 100%) !important;
  box-shadow: 0 4px 18px rgba(0,201,167,.45) !important;
  transform: translateY(-1px) !important;
}}
[data-testid="stMain"] .stButton > button[kind="primary"]:active {{
  background: linear-gradient(135deg,#00A88C,#008B75) !important;
  transform: translateY(0) !important; box-shadow: none !important;
}}
/* secondary */
[data-testid="stMain"] .stButton > button[kind="secondary"] {{
  background: rgba(22,22,34,.9) !important; color: var(--text) !important;
  border:1px solid var(--border-st) !important; box-shadow:none !important;
}}
[data-testid="stMain"] .stButton > button[kind="secondary"]:hover {{
  background: var(--hover) !important; border-color: rgba(0,201,167,.25) !important;
}}
[data-testid="stMain"] .stButton > button:disabled {{ opacity:.42 !important; cursor:not-allowed !important; }}

/* ══ FORM CONTROLS ══
   Base colours come from .streamlit/config.toml (Streamlit's own theme), which
   is why these rules only adjust geometry and typography rather than repainting
   every widget. Selects render as .react-aria-ComboBox here — NOT the
   data-baseweb="select" that older Streamlit builds emit. */
[data-testid="stMain"] .react-aria-ComboBox > div,
[data-testid="stMain"] [data-testid="stSelectbox"] > div > div {{
  border-radius: var(--r-md) !important; font-size:var(--fs-3) !important;
}}
[role="listbox"], .react-aria-Popover {{
  border-radius: var(--r-md) !important; border:1px solid var(--border-st) !important;
}}
[role="option"] {{ font-size:var(--fs-3) !important; }}

[data-testid="stMain"] label, [data-testid="stMain"] [data-testid="stWidgetLabel"] p {{
  font-size:var(--fs-2) !important; font-weight:500 !important; color: var(--text-2) !important;
}}
[data-testid="stMain"] input {{ border-radius: var(--r-md) !important; font-size:var(--fs-3) !important; }}

/* uploader dropzone is a <section data-testid="stFileUploaderDropzone"> */
[data-testid="stFileUploaderDropzone"] {{
  border:1.5px dashed rgba(0,201,167,.25) !important; border-radius: var(--r-lg) !important;
  background: rgba(0,201,167,.03) !important;
  transition: border-color 180ms var(--ease),background 180ms var(--ease) !important;
}}
[data-testid="stFileUploaderDropzone"]:hover {{
  border-color: rgba(0,201,167,.5) !important;
  background: rgba(0,201,167,.06) !important;
}}
[data-testid="stFileUploaderDropzone"] small {{ color: var(--text-3) !important; }}
[data-testid="stFileUploaderDropzone"] button {{
  border-radius: var(--r-sm) !important; font-size:var(--fs-2) !important;
  border:1px solid var(--border-st) !important;
}}

/* ══ TABS ══ (this build emits data-testid="stTab", not data-baseweb) ══ */
[data-testid="stTabs"] [role="tablist"] {{ gap: var(--s1); border-bottom:1px solid var(--border); }}
[data-testid="stTab"] {{
  background: transparent !important; color: var(--text-3) !important;
  font-size:var(--fs-3) !important; font-weight:500 !important; padding: 9px var(--s3) !important;
  border-radius: var(--r-sm) var(--r-sm) 0 0 !important; transition: color 170ms var(--ease) !important;
}}
[data-testid="stTab"]:hover {{ color: var(--text-2) !important; background: rgba(0,201,167,.04) !important; }}
[data-testid="stTab"][aria-selected="true"] {{ color: var(--accent-hi) !important; font-weight:600 !important; }}
/* The active-tab indicator. The rule here used to target
   [data-baseweb="tab-highlight"], which this build does not render — so the
   selected tab was distinguished by colour alone. Drawn from aria-selected
   instead: an ARIA state is part of the accessibility contract and does not
   churn between Streamlit releases the way an internal class does. */
[data-testid="stTab"][aria-selected="true"] {{
  box-shadow: inset 0 -2px 0 0 var(--accent);
}}

/* ══ EXPANDER ══ */
[data-testid="stExpander"] {{
  background: var(--surface) !important; border:1px solid var(--border) !important;
  border-radius: var(--r-lg) !important;
}}
[data-testid="stExpander"] summary {{ font-size:var(--fs-3) !important; font-weight:500 !important; color: var(--text-2) !important; }}
[data-testid="stExpander"] summary:hover {{ color: var(--text) !important; }}

/* ══ PROGRESS ══ */
[data-testid="stMain"] .stProgress > div > div {{ background: rgba(0,201,167,.1) !important; height:4px !important; border-radius:var(--r-full) !important; }}
[data-testid="stMain"] .stProgress > div > div > div {{
  background: linear-gradient(90deg,var(--accent),var(--accent-hi)) !important;
  border-radius:var(--r-full) !important; box-shadow:0 0 6px rgba(0,201,167,.4) !important;
}}

/* ══ TOGGLE ══
   Nothing here on purpose. This build renders a toggle as a plain
   <input type="checkbox"> inside [data-testid="stCheckbox"] — no
   data-baseweb attribute and no aria-checked — so the rule that used to sit
   here (`[data-baseweb="checkbox"] div[aria-checked="true"]`) matched nothing.
   The accent already comes from primaryColor in .streamlit/config.toml, which
   is the right place for it. */

/* ══ COLUMNS ══ */
[data-testid="stColumn"] {{ padding: 0 6px !important; }}
[data-testid="stColumn"]:first-child {{ padding-left:0 !important; }}
[data-testid="stColumn"]:last-child {{ padding-right:0 !important; }}

/* ══ SCROLLBAR ══ */
::-webkit-scrollbar {{ width:5px; height:5px; }}
::-webkit-scrollbar-track {{ background:transparent; }}
::-webkit-scrollbar-thumb {{ background: rgba(0,201,167,.18); border-radius:var(--r-full); }}
::-webkit-scrollbar-thumb:hover {{ background: rgba(0,201,167,.35); }}

/* ══ SHARED COMPONENT CLASSES ══ */
.card {{
  background: linear-gradient(160deg, var(--elevated) 0%, rgba(13,17,23,.97) 100%);
  border:1px solid var(--border); border-radius: var(--r-xl);
  padding: var(--s6); transition: border-color 220ms var(--ease),box-shadow 220ms var(--ease);
  box-shadow: var(--shadow-sm);
  position:relative; overflow:hidden;
}}
.card::after {{
  content:""; position:absolute; top:0; left:0; right:0; height:1px;
  background: linear-gradient(90deg,transparent,rgba(255,255,255,.05),transparent);
  pointer-events:none;
}}
.card-i {{ transition: border-color 200ms var(--ease),box-shadow 200ms var(--ease),transform 180ms var(--ease); cursor:pointer; }}
.card-i:hover {{
  border-color: rgba(0,201,167,.28);
  box-shadow: var(--shadow-md), 0 0 24px rgba(0,201,167,.08);
  transform: translateY(-2px);
}}

.kpi {{
  position:relative; overflow:hidden;
  background: linear-gradient(155deg, var(--elevated) 0%, rgba(13,17,23,.95) 100%);
  border:1px solid var(--border); border-radius: var(--r-xl);
  padding: var(--s5) var(--s5) var(--s4);
  box-shadow: var(--shadow-sm);
  transition: border-color 220ms var(--ease),box-shadow 220ms var(--ease),transform 180ms var(--ease);
}}
.kpi::before {{
  content:""; position:absolute; top:0; left:0; right:0; height:2px;
  background: currentColor; opacity:.35; border-radius: var(--r-xl) var(--r-xl) 0 0;
}}
.kpi::after {{
  content:""; position:absolute; top:2px; left:0; right:0; height:1px;
  background: linear-gradient(90deg,transparent,rgba(255,255,255,.07),transparent);
}}
.kpi:hover {{
  border-color: var(--border-st);
  box-shadow: var(--shadow-md), 0 0 18px rgba(0,0,0,.1);
  transform: translateY(-2px);
}}
.kpi-h {{ display:flex; align-items:center; gap:10px; margin-bottom:16px; }}
.kpi-icon {{
  width:36px; height:36px; border-radius:var(--r-md); display:flex; align-items:center;
  justify-content:center; flex-shrink:0;
  box-shadow: 0 2px 8px rgba(0,0,0,.45), inset 0 1px 0 rgba(255,255,255,.06);
}}
.kpi-icon svg {{ width:16px; height:16px; fill:none; stroke:currentColor; stroke-width:1.9; stroke-linecap:round; stroke-linejoin:round; }}
.kpi-l {{ font-size:var(--fs-0); font-weight:700; color: var(--text-3); letter-spacing:.12em; text-transform:uppercase; }}
.kpi-v {{ font-size:var(--fs-6); font-weight:700; color: var(--text); letter-spacing:-0.04em; line-height:1.0;
         font-variant-numeric: tabular-nums; }}
.kpi-s {{ font-size:var(--fs-1); color: var(--text-3); margin-top:9px; line-height:1.55; }}

.badge {{
  display:inline-flex; align-items:center; gap:5px; padding:2.5px 9px; border-radius:var(--r-full);
  font-size:var(--fs-0); font-weight:700; line-height:1.5; white-space:nowrap;
  letter-spacing:.015em;
}}
.dot {{ width:5px; height:5px; border-radius:50%; flex-shrink:0; }}

.sec {{ display:flex; align-items:center; gap:var(--s3); margin: var(--s8) 0 var(--s5); }}
.sec-t {{
  font-size:var(--fs-0); font-weight:700; color: var(--accent); letter-spacing:.15em; text-transform:uppercase;
  white-space:nowrap; padding:4px 11px; border-radius:var(--r-xs);
  background: rgba(0,201,167,.07); border:1px solid rgba(0,201,167,.18);
  box-shadow: 0 0 12px rgba(0,201,167,.06);
}}
.sec-r {{ flex:1; height:1px; background: linear-gradient(90deg,rgba(0,201,167,.18),transparent); }}

/* Two hierarchy levels below .sec — see card_title() / group_label(). */
.card-t {{
  font-size: var(--fs-2); font-weight: var(--fw-medium); color: var(--text);
  letter-spacing:-0.01em; margin:0 0 var(--s3);
}}
.group-t {{
  font-size: var(--fs-3); font-weight: var(--fw-medium); color: var(--text);
  letter-spacing:-0.01em; margin: var(--s6) 0 var(--s1);
}}
.group-t-first {{ margin-top:0; }}

/* Status strip — the header's live system state. Each chip is one measured or
   configured fact, never a decorative label. */
.sstrip {{ display:flex; flex-wrap:wrap; gap:var(--s2); margin:0 0 var(--s6); }}
.schip {{
  display:inline-flex; align-items:center; gap:9px; padding:7px 13px;
  border-radius:var(--r-full); background:var(--elevated);
  border:1px solid var(--border); box-shadow:var(--shadow-sm);
}}
.schip-dot {{ width:6px; height:6px; border-radius:50%; flex-shrink:0; }}
.schip-l {{
  font-size:var(--fs-0); font-weight:var(--fw-bold); letter-spacing:.1em;
  text-transform:uppercase; color:var(--text-3); white-space:nowrap;
}}
.schip-v {{
  font-size:var(--fs-2); font-weight:var(--fw-medium); color:var(--text);
  font-variant-numeric:tabular-nums; white-space:nowrap;
}}

.tbl {{ width:100%; border-collapse:collapse; font-size:var(--fs-2); }}
.tbl th {{
  text-align:left; padding:10px var(--s3); font-size:var(--fs-0); font-weight:700; color: var(--text-3);
  text-transform:uppercase; letter-spacing:.12em; border-bottom:1px solid var(--border); white-space:nowrap;
  background: rgba(255,255,255,.02);
}}
.tbl td {{ padding:11px var(--s3); color: var(--text-2); border-bottom:1px solid rgba(99,130,170,.06); white-space:nowrap; vertical-align:middle; }}
.tbl tr:last-child td {{ border-bottom:none; }}
.tbl tbody tr {{ transition: background 130ms var(--ease); }}
.tbl tbody tr:hover {{ background: rgba(255,255,255,.025); }}
.tbl-w {{ overflow-x:auto; border:1px solid var(--border); border-radius: var(--r-lg); background: var(--surface); box-shadow: var(--shadow-sm); }}

/* Code listings.
   `code_block()` emits <pre class="code">, but Streamlit rewrites that into its
   own <div data-testid="stMarkdownPre"> and DROPS the class — so the wrapper is
   what has to be styled. The <pre> still earns its place: Markdown does not
   descend into one, which is what stops a `# comment` becoming an <h1>. */
.code,
[data-testid="stMain"] [data-testid="stMarkdownPre"] {{
  display:block; margin:0;
  background:rgba(4,4,10,.95); border:1px solid var(--border); border-radius: var(--r-md);
  padding: var(--s4); font-family: var(--mono) !important; font-size: var(--fs-2); line-height:1.7;
  color:#C8D3E8; overflow-x:auto; white-space:pre; tab-size:4;
}}
[data-testid="stMain"] [data-testid="stMarkdownPre"] * {{
  font-family: var(--mono) !important; font-size: inherit !important; color: inherit !important;
}}
.code .k {{ color:#33DEC0; }} .code .s {{ color:#A7F0D8; }} .code .c {{ color:#404060; font-style:italic; }}

.empty {{
  display:flex; flex-direction:column; align-items:center; justify-content:center;
  padding: var(--s12) var(--s6); border:1.5px dashed rgba(0,201,167,.16); border-radius: var(--r-xl);
  background: radial-gradient(ellipse at 50% 0%, rgba(0,201,167,.04) 0%, transparent 70%);
  text-align:center; gap:var(--s2);
}}
.empty svg {{ width:28px; height:28px; fill:none; stroke: var(--accent); opacity:.35; stroke-width:1.6; stroke-linecap:round; stroke-linejoin:round; }}

.skel {{ background: linear-gradient(90deg,rgba(0,201,167,.04) 25%,rgba(0,201,167,.09) 37%,rgba(0,201,167,.04) 63%);
        background-size:400% 100%; animation: shimmer 1.4s ease infinite; border-radius:var(--r-sm); }}
@keyframes shimmer {{ 0%{{background-position:100% 0}} 100%{{background-position:0 0}} }}
@keyframes spin {{ to {{ transform: rotate(360deg) }} }}
@keyframes rise {{ from {{opacity:0; transform:translateY(6px)}} to {{opacity:1; transform:none}} }}
.rise {{ animation: rise 220ms var(--ease) both; }}

/* ══ RESPONSIVE ══ */
/* Streamlit pins a 200px floor on its sidebar from a stylesheet that outranks
   author CSS, so a 64–72px icon rail is not reachable without replacing the
   sidebar component. 200px comfortably fits the labels, so the compact state
   keeps them and sheds the status/preferences panels instead — an empty 200px
   column of icons would waste the width rather than reflow it. */
@media (max-width: 1100px) {{
  /* min/max-width must be overridden too. The base rule pins all three at
     256px with !important, so overriding `width` alone did nothing — the
     min-width floor held the rail at 256px and this whole block was inert. */
  /* 224px, not 200px: the wordmark needs ~122px beside a 38px icon, and at
     200px "AdaptiveDenoise" wrapped onto two lines while "Processing History"
     truncated to an ellipsis. The brand also steps down a size here. */
  [data-testid="stSidebar"][data-testid="stSidebar"] {{
    width:224px !important; min-width:224px !important; max-width:224px !important;
    flex:0 0 224px !important;
  }}
  [data-testid="stSidebar"] .sb-brand-name {{ font-size: var(--fs-3) !important; }}
  [data-testid="stSidebar"] .sb-brand-icon {{ width:32px !important; height:32px !important; }}
  [data-testid="stSidebar"] .sb-detail {{ display:none !important; }}
  [data-testid="stSidebar"] .nav-row {{ margin:0 var(--s2) 2px; padding:0 10px; gap:9px; }}
  [data-testid="stSidebar"] .nav-label {{ font-size:var(--fs-2); }}
  [data-testid="stSidebar"] .stButton > button {{
    width:calc(100% - 16px) !important; margin:0 var(--s2) !important;
  }}
  [data-testid="stMain"] .block-container {{ padding: var(--s5) var(--s5) var(--s10); }}
}}
/* 900px, not 760px. Between those widths the three-up action row squeezed the
   primary button to 116px against a ~150px label: wrapping split it across two
   lines, and nowrap clipped it outright. Full-width stacked actions fit at any
   width in this range, and a portrait tablet reads a single column better than
   a cramped three-up one. Above 900px the narrowest action column is ~164px,
   which clears the label. */
@media (max-width: 900px) {{
  [data-testid="stMain"] .block-container {{ padding: var(--s4) var(--s4) var(--s8); }}
  .kpi-v {{ font-size:var(--fs-5); }}
  .page-t {{ font-size:var(--fs-5) !important; }}
  /* Real reflow: columns stack rather than being squeezed side by side.
     min-width:100% alone did nothing — Streamlit's row is a nowrap flex
     container, so the children just compressed. The row has to be allowed to
     wrap before a 100% child can take its own line. */
  [data-testid="stHorizontalBlock"] {{
    flex-wrap: wrap !important; gap: var(--s3) !important;
  }}
  [data-testid="stColumn"] {{
    padding:0 !important; min-width:100% !important; flex: 1 1 100% !important;
  }}

  /* Touch targets. 38-40px is below the 44px minimum for a finger, and these
     are the app's primary actions. */
  [data-testid="stMain"] .stButton > button,
  [data-testid="stMain"] .stDownloadButton > button {{ height:44px !important; }}
  [data-testid="stMain"] .react-aria-ComboBox > div,
  [data-testid="stMain"] [data-testid="stSelectbox"] > div > div {{ min-height:44px !important; }}

  /* The empty state's 48px vertical padding is generous on a phone, where the
     viewport is the scarce resource rather than the horizontal space. */
  .empty {{ padding: var(--s8) var(--s4); }}
  .sec {{ margin: var(--s6) 0 var(--s4); }}
  /* The status strip is five facts; at full padding they wrapped to four rows
     and ate a fifth of an 812px phone screen. Tightening the chip rather than
     dropping a fact — each one is real state a user came here to read. */
  /* One scrollable row, the standard mobile chip pattern. Wrapping put five
     facts on four lines; a partially-visible trailing chip signals the scroll
     and keeps every fact reachable. */
  .sstrip {{
    flex-wrap:nowrap; overflow-x:auto; gap:6px; margin-bottom: var(--s4);
    padding-bottom:4px; -webkit-overflow-scrolling:touch;
    scrollbar-width:none;
  }}
  .sstrip::-webkit-scrollbar {{ display:none; }}
  .schip {{ padding:5px 10px; gap:7px; flex:0 0 auto; }}
  .schip-l {{ letter-spacing:.06em; }}
  .card {{ padding: var(--s4); }}

  /* A wide table scrolls inside its own frame instead of widening the page. */
  .tbl-w {{ -webkit-overflow-scrolling: touch; }}
  figure img {{ height:auto; }}
}}
</style>
"""

# ═══════════════════════════════════════════════════════════════════════════
# ICONS  (one family: Lucide-style outline, 24 viewBox, stroke set by CSS)
# ═══════════════════════════════════════════════════════════════════════════

ICONS = {
    "dashboard": '<rect x="3" y="3" width="7.5" height="7.5" rx="1.6"/><rect x="13.5" y="3" width="7.5" height="7.5" rx="1.6"/><rect x="3" y="13.5" width="7.5" height="7.5" rx="1.6"/><rect x="13.5" y="13.5" width="7.5" height="7.5" rx="1.6"/>',
    "process":   '<path d="M20.5 14.5V6a2 2 0 0 0-2-2H5a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h9.5"/><circle cx="8.5" cy="9" r="1.8"/><path d="M3 16.5l4.2-4.2a1.8 1.8 0 0 1 2.5 0l3.6 3.6"/><path d="M18.5 16v6M15.5 19h6"/>',
    "projects":  '<path d="M21 19V9a1.8 1.8 0 0 0-1.8-1.8h-6.4L10.6 5H4.8A1.8 1.8 0 0 0 3 6.8V19a1.8 1.8 0 0 0 1.8 1.8h14.4A1.8 1.8 0 0 0 21 19z"/>',
    "history":   '<circle cx="12" cy="12" r="8.5"/><path d="M12 7.2V12l3.2 1.9"/>',
    "fpga":      '<rect x="7" y="7" width="10" height="10" rx="1.5"/><rect x="10.5" y="10.5" width="3" height="3" rx=".6"/><path d="M9.5 3v4M14.5 3v4M9.5 17v4M14.5 17v4M3 9.5h4M3 14.5h4M17 9.5h4M17 14.5h4"/>',
    "api":       '<path d="M15.5 17.5 21 12l-5.5-5.5"/><path d="M8.5 6.5 3 12l5.5 5.5"/>',
    "analytics": '<path d="M3.5 20.5h17"/><path d="M6.8 20.5v-5.2M12 20.5V8.8M17.2 20.5V4.5"/>',
    "billing":   '<rect x="2.8" y="5.2" width="18.4" height="13.6" rx="2.2"/><path d="M2.8 10h18.4"/><path d="M6.5 14.6h3"/>',
    "settings":  '<path d="M5 8.5h14M5 15.5h14"/><circle cx="9.5" cy="8.5" r="2.2"/><circle cx="15" cy="15.5" r="2.2"/>',
    # utility
    "image":     '<rect x="3" y="4" width="18" height="16" rx="2"/><circle cx="8.5" cy="9.5" r="1.8"/><path d="m3 16.5 4.5-4.5a1.8 1.8 0 0 1 2.5 0L21 21"/>',
    "clock":     '<circle cx="12" cy="12" r="8.5"/><path d="M12 7.2V12l3.2 1.9"/>',
    "gauge":     '<path d="M12 14.5 16 9"/><path d="M4.2 17.5a9 9 0 1 1 15.6 0"/><circle cx="12" cy="17.5" r="1.4"/>',
    "cpu":       '<rect x="7" y="7" width="10" height="10" rx="1.5"/><path d="M9.5 3v4M14.5 3v4M9.5 17v4M14.5 17v4M3 9.5h4M3 14.5h4M17 9.5h4M17 14.5h4"/>',
    "brain":     '<path d="M12 5.5a3 3 0 0 0-5.6-1.5A2.8 2.8 0 0 0 4 9a2.9 2.9 0 0 0 .6 4.4A3 3 0 0 0 7 18.5a3 3 0 0 0 5 1.3z"/><path d="M12 5.5a3 3 0 0 1 5.6-1.5A2.8 2.8 0 0 1 20 9a2.9 2.9 0 0 1-.6 4.4A3 3 0 0 1 17 18.5a3 3 0 0 1-5 1.3z"/>',
    "check":     '<path d="M20 6 9 17l-5-5"/>',
    "alert":     '<path d="M12 8.5v4.5M12 16.5h.01"/><circle cx="12" cy="12" r="9"/>',
    "x":         '<path d="M18 6 6 18M6 6l12 12"/>',
    "search":    '<circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/>',
    "download":  '<path d="M12 3v12"/><path d="m7.5 10.5 4.5 4.5 4.5-4.5"/><path d="M4 20h16"/>',
    "arrow":     '<path d="M5 12h14"/><path d="m13 6 6 6-6 6"/>',
    "plus":      '<path d="M12 5v14M5 12h14"/>',
    "layers":    '<path d="m12 3 9 5-9 5-9-5 9-5z"/><path d="m3 13 9 5 9-5"/>',
    "file":      '<path d="M14 3v5h5"/><path d="M19 21H5a1.8 1.8 0 0 1-1.8-1.8V4.8A1.8 1.8 0 0 1 5 3h9l6.8 6.8V19.2A1.8 1.8 0 0 1 19 21z"/>',
    "shield":    '<path d="M12 21s7.5-3.4 7.5-9.4V5.6L12 3 4.5 5.6v6C4.5 17.6 12 21 12 21z"/>',
}


def ico(k: str, size: int = 16, color: str = "currentColor", sw: float = 1.8) -> str:
    return (
        f'<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" aria-hidden="true" '
        f'style="width:{size}px;height:{size}px;fill:none;stroke:{color};stroke-width:{sw};'
        f'stroke-linecap:round;stroke-linejoin:round;flex-shrink:0;">{ICONS[k]}</svg>'
    )


# ═══════════════════════════════════════════════════════════════════════════
# COMPONENT PRIMITIVES
# ═══════════════════════════════════════════════════════════════════════════

def esc(s) -> str:
    return html.escape(str(s))


def page_head(title: str, sub: str, crumb: list[str] | None = None) -> str:
    trail = ""
    if crumb:
        parts = []
        for i, c in enumerate(crumb):
            last = i == len(crumb) - 1
            col = T["text_2"] if last else T["text_3"]
            parts.append(f'<span style="color:{col};font-size:var(--fs-1);font-weight:{"550" if last else "400"};">{esc(c)}</span>')
            if not last:
                parts.append(f'<span style="color:{T["text_3"]};opacity:.5;font-size:var(--fs-1);">/</span>')
        trail = (
            '<div style="display:flex;align-items:center;gap:7px;margin-bottom:8px;">'
            + "".join(parts) + "</div>"
        )
    return (
        f'<div class="rise" style="margin-bottom:34px;">{trail}'
        f'<div style="display:flex;align-items:center;gap:16px;margin-bottom:9px;">'
        f'<div style="width:4px;height:38px;border-radius:4px;flex-shrink:0;'
        f'background:linear-gradient(180deg,#2AEBB8 0%,{T["accent"]} 55%,{T["cyan"]} 100%);'
        f'box-shadow:0 0 14px rgba(0,201,167,.6);"></div>'
        f'<h1 class="page-t" style="font-size:var(--fs-6);font-weight:700;letter-spacing:-0.036em;'
        f'margin:0;line-height:1.15;color:{T["text"]};">{esc(title)}</h1></div>'
        f'<p style="font-size:var(--fs-3);color:{T["text_2"]};margin:0 0 0 20px;line-height:1.65;'
        f'letter-spacing:-.005em;">{sub}</p></div>'
    )


def section(title: str) -> str:
    """A major page region. The top of the in-page hierarchy below the title."""
    return f'<div class="sec"><span class="sec-t">{esc(title)}</span><span class="sec-r"></span></div>'


def status_chip(label: str, value: str, color: str) -> str:
    """One fact in the header status strip: a state light, a label and a value.

    The colour is semantic, never decorative — amber means a thing is genuinely
    not configured, and nothing here reports a figure the build cannot measure.
    """
    return (
        f'<div class="schip">'
        f'<span class="schip-dot" style="background:{color};box-shadow:0 0 6px {color}99;"></span>'
        f'<span class="schip-l">{esc(label)}</span>'
        f'<span class="schip-v">{esc(value)}</span></div>'
    )


def status_strip(*chips: str) -> str:
    return f'<div class="sstrip">{"".join(chips)}</div>'


def card_title(title: str) -> str:
    """The label at the top of a card — one level below `section()`.

    This role was previously written inline at sixteen call sites across two
    sizes and six different bottom margins (3/4/6/10/12/16px), so the same kind
    of heading sat at a different height on every page. One helper, one margin.
    """
    return f'<div class="card-t">{esc(title)}</div>'


def group_label(title: str, first: bool = False) -> str:
    """A field-group heading inside a settings tab — below `card_title()`.

    `first` drops the leading margin so a group opening a tab does not push
    itself away from the tab strip.
    """
    return f'<div class="group-t{" group-t-first" if first else ""}">{esc(title)}</div>'


def badge(text: str, color: str, dot: bool = False, solid: bool = False) -> str:
    if solid:
        style = f"background:{color};color:#0B1220;"
    else:
        style = f"background:{color}1A;color:{color};border:1px solid {color}33;"
    d = f'<span class="dot" style="background:{color};"></span>' if dot else ""
    return f'<span class="badge" style="{style}">{d}{esc(text)}</span>'


def kpi(icon: str, label: str, value: str, sub: str, color: str = None) -> str:
    color = color or T["accent"]
    return (
        f'<div class="kpi" style="color:{color};">'
        f'<div class="kpi-h">'
        f'<div class="kpi-icon" style="background:{color}18;color:{color};">{ico(icon,15,color,1.9)}</div>'
        f'<span class="kpi-l">{esc(label)}</span></div>'
        f'<div class="kpi-v">{value}</div>'
        f'<div class="kpi-s">{sub}</div></div>'
    )


def card(body: str, pad: str = "20px", extra: str = "", interactive: bool = False,
         max_w: int | None = None) -> str:
    """A surface panel.

    `max_w` caps the width for read-only key/value panels. A settings card left
    to fill a 1100px column puts its label at the far left and its value at the
    far right, so the eye crosses ~600px of nothing to pair the two. Reading
    columns want a bound; grids and tables do not, so this is opt-in.
    """
    cls = "card card-i" if interactive else "card"
    width = f"max-width:{max_w}px;" if max_w else ""
    return f'<div class="{cls}" style="padding:{pad};{width}{extra}">{body}</div>'


def alert(title: str, body: str, kind: str = "info") -> str:
    cfg = {
        "info":    (T["info"], "alert"),
        "success": (T["ok"],   "check"),
        "warning": (T["warn"], "alert"),
        "error":   (T["err"],  "x"),
    }
    color, icon = cfg.get(kind, cfg["info"])
    return (
        f'<div role="status" style="display:flex;gap:12px;padding:12px 16px;border-radius:var(--r-lg);'
        f'background:{color}0D;border:1px solid {color}2A;border-left:3px solid {color};margin:10px 0;'
        f'box-shadow:0 2px 8px rgba(0,0,0,.2);">'
        f'<div style="margin-top:2px;flex-shrink:0;">{ico(icon,14,color,2.0)}</div><div style="min-width:0;">'
        f'<div style="font-size:var(--fs-2);font-weight:700;color:{color};margin-bottom:3px;letter-spacing:-.01em;">{esc(title)}</div>'
        f'<div style="font-size:var(--fs-2);color:{T["text_2"]};line-height:1.58;">{body}</div></div></div>'
    )


def empty(icon: str, title: str, why: str, next_step: str) -> str:
    return (
        f'<div class="empty rise" style="padding:40px 24px;">'
        f'<div style="width:52px;height:52px;border-radius:50%;background:{T["elevated"]};'
        f'border:1.5px solid {T["border_st"]};display:flex;align-items:center;justify-content:center;'
        f'margin:0 auto 14px;">{ico(icon,24,T["text_3"],1.6)}</div>'
        f'<div style="font-size:var(--fs-4);font-weight:700;color:{T["text"]};margin-bottom:6px;">{esc(title)}</div>'
        f'<div style="font-size:var(--fs-2);color:{T["text_2"]};max-width:440px;line-height:1.65;margin:0 auto;">{why}</div>'
        f'<div style="font-size:var(--fs-2);color:{T["accent"]};max-width:440px;line-height:1.6;margin:10px auto 0;'
        f'padding:8px 14px;background:{T["accent_dim"]};border-radius:var(--r-sm);border:1px solid rgba(0,201,167,.2);">'
        f'{next_step}</div>'
        f'</div>'
    )


def table(
    headers: list[str],
    rows: list[list[str]],
    align_r: set[int] = frozenset(),
    flex_col: int = 0,
) -> str:
    """A data table whose metadata columns hug their content.

    `flex_col` is the column that absorbs the leftover width — normally the one
    holding the identifier or the prose. Every other column is sized to its
    content (`width:1%` with `nowrap` is the standard trick), which keeps the
    numbers and badges in a tight, scannable group instead of being spread
    across the full width. Without this a three-column table stretched ~400px
    of content over 1092px, leaving a 489px void between a module name and its
    size — the eye had to travel the whole row to pair a value with its label.
    """
    def w(i: int) -> str:
        return "width:100%;" if i == flex_col else "width:1%;white-space:nowrap;"

    th = "".join(
        f'<th style="text-align:{"right" if i in align_r else "left"};{w(i)}">{esc(h)}</th>'
        for i, h in enumerate(headers)
    )
    body = ""
    for r in rows:
        tds = "".join(
            f'<td style="text-align:{"right" if i in align_r else "left"};{w(i)}'
            f'{"font-variant-numeric:tabular-nums;" if i in align_r else ""}'
            # The flexible column is the only one allowed to wrap, and it
            # truncates rather than forcing the whole table into a scrollbar.
            f'{"max-width:0;overflow:hidden;text-overflow:ellipsis;" if i == flex_col else ""}">{c}</td>'
            for i, c in enumerate(r)
        )
        body += f"<tr>{tds}</tr>"
    return f'<div class="tbl-w"><table class="tbl"><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'


def kv_rows(pairs: list[tuple[str, str]]) -> str:
    out = ""
    for k, v in pairs:
        out += (
            f'<div style="display:flex;justify-content:space-between;align-items:center;gap:16px;'
            f'padding:9px 0;border-bottom:1px solid rgba(99,130,170,.07);">'
            f'<span style="font-size:var(--fs-1);color:{T["text_3"]};font-weight:500;letter-spacing:.01em;">{esc(k)}</span>'
            f'<span style="font-size:var(--fs-2);color:{T["text_2"]};font-variant-numeric:tabular-nums;'
            f'text-align:right;font-weight:500;">{v}</span></div>'
        )
    return f'<div>{out}</div>'


def meter(pct: float, color: str, height: int = 6) -> str:
    w = max(0.0, min(100.0, pct))
    return (
        f'<div role="progressbar" aria-valuenow="{w:.0f}" aria-valuemin="0" aria-valuemax="100" '
        f'style="background:rgba(99,130,170,.11);border-radius:var(--r-full);height:{height}px;overflow:hidden;'
        f'box-shadow:inset 0 1px 2px rgba(0,0,0,.3);">'
        f'<div style="width:{w}%;height:100%;background:linear-gradient(90deg,{color}CC,{color});'
        f'border-radius:var(--r-full);transition:width 280ms var(--ease);'
        f'box-shadow:0 0 6px {color}60;"></div></div>'
    )


def code_block(text: str) -> str:
    """Render a code sample verbatim.

    The element MUST be <pre>, not <div>. Streamlit runs the string through a
    Markdown parser before the HTML reaches the page, and Markdown descends
    into a <div>: every ``# comment`` in a Python sample became an <h1> (anchor
    link and all) and every blank line split the listing into <p> blocks. The
    API page's samples were unreadable because of it. Markdown does not process
    the content of a <pre>, so the listing survives as written.
    """
    return f'<pre class="code">{esc(text)}</pre>'


# ── SVG charts (no external libs; full control, theme-consistent) ──────────

def chart_bars(values: list[float], labels: list[str], color: str, unit: str = "", h: int = 150) -> str:
    if not values:
        return ""
    n, w, gap = len(values), 640, 8
    # Cap the bar width and centre the group. Dividing the full width by n made
    # a single run render as one 640px block filling the whole card — a solid
    # slab that reads as a broken chart rather than as one measurement.
    bw = min(56.0, (w - gap * (n - 1)) / n)
    group_w = n * bw + gap * (n - 1)
    x0 = max(0.0, (w - group_w) / 2)
    top = max(values) or 1.0
    bars, lbls, vals = "", "", ""
    for i, v in enumerate(values):
        bh = max(2.0, (v / top) * (h - 36))
        x, y = x0 + i * (bw + gap), (h - 24) - bh
        bars += (
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{bh:.1f}" rx="4" fill="{color}" opacity="0.82">'
            f'<title>{esc(labels[i])}: {v:.4g}{unit}</title></rect>'
        )
        # value label above bar
        if n <= 16 and v > 0:
            vals += (
                f'<text x="{x + bw/2:.1f}" y="{y - 4:.1f}" text-anchor="middle" '
                f'font-size="10" fill="{color}" opacity=".9" font-weight="600" '
                f'font-family="Inter,sans-serif">{v:.4g}</text>'
            )
        if n <= 14:
            lbls += (
                f'<text x="{x + bw/2:.1f}" y="{h - 6}" text-anchor="middle" '
                f'font-size="10.5" fill="{T["text_3"]}" font-family="Inter,sans-serif">{esc(labels[i][:9])}</text>'
            )
    grid = "".join(
        f'<line x1="0" y1="{(h-24) * f:.1f}" x2="{w}" y2="{(h-24)*f:.1f}" '
        f'stroke="{T["border"]}" stroke-width="1" stroke-dasharray="{"none" if f==1.0 else "4 3"}"/>'
        for f in (0.0, 0.5, 1.0)
    )
    return (
        f'<svg viewBox="0 0 {w} {h}" style="width:100%;height:auto;overflow:visible;" role="img">'
        f'{grid}{bars}{vals}{lbls}</svg>'
    )


def chart_donut(segments: list[tuple[str, int, str]], size: int = 128) -> str:
    total = sum(s[1] for s in segments) or 1
    r, cx, sw = size / 2 - 12, size / 2, 13
    circ = 2 * math.pi * r
    off, arcs = 0.0, ""
    for name, val, col in segments:
        if val == 0:
            continue
        frac = val / total
        arcs += (
            f'<circle cx="{cx}" cy="{cx}" r="{r}" fill="none" stroke="{col}" stroke-width="{sw}" '
            f'stroke-dasharray="{circ*frac:.2f} {circ:.2f}" stroke-dashoffset="{-off:.2f}" '
            f'transform="rotate(-90 {cx} {cx})" stroke-linecap="butt">'
            f'<title>{esc(name)}: {val} ({frac*100:.0f}%)</title></circle>'
        )
        off += circ * frac
    return (
        f'<svg viewBox="0 0 {size} {size}" style="width:{size}px;height:{size}px;" role="img">'
        f'<circle cx="{cx}" cy="{cx}" r="{r}" fill="none" stroke="rgba(148,163,184,.09)" stroke-width="{sw}"/>'
        f'{arcs}<text x="{cx}" y="{cx-2}" text-anchor="middle" font-size="21" font-weight="650" '
        f'fill="{T["text"]}" font-family="Inter,sans-serif">{total}</text>'
        f'<text x="{cx}" y="{cx+14}" text-anchor="middle" font-size="10" fill="{T["text_3"]}" '
        f'font-family="Inter,sans-serif">runs</text></svg>'
    )


def chart_line(values: list[float], labels: list[str], color: str, unit: str = "", h: int = 150) -> str:
    if len(values) < 2:
        return ""
    w = 640
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    pad = span * 0.15
    lo, hi = lo - pad, hi + pad
    span = hi - lo
    pts = [(i * w / (len(values) - 1), (h - 24) - ((v - lo) / span) * (h - 34)) for i, v in enumerate(values)]
    path = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    area = path + f" L{w},{h-24} L0,{h-24} Z"
    dots = "".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{T["bg"]}" stroke="{color}" stroke-width="2">'
        f'<title>{esc(labels[i])}: {values[i]:.3g}{unit}</title></circle>'
        for i, (x, y) in enumerate(pts)
    )
    zero_y = None
    if lo < 0 < hi:
        zero_y = (h - 24) - ((0 - lo) / span) * (h - 34)
    zline = (
        f'<line x1="0" y1="{zero_y:.1f}" x2="{w}" y2="{zero_y:.1f}" stroke="{T["text_3"]}" '
        f'stroke-width="1" stroke-dasharray="3 3" opacity=".55"/>' if zero_y is not None else ""
    )
    return (
        f'<svg viewBox="0 0 {w} {h}" style="width:100%;height:auto;overflow:visible;" role="img">'
        f'<defs><linearGradient id="lg" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0%" stop-color="{color}" stop-opacity=".22"/>'
        f'<stop offset="100%" stop-color="{color}" stop-opacity="0"/></linearGradient></defs>'
        f'<line x1="0" y1="{h-24}" x2="{w}" y2="{h-24}" stroke="{T["border"]}" stroke-width="1"/>'
        f'{zline}<path d="{area}" fill="url(#lg)"/>'
        f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>'
        f'{dots}</svg>'
    )


def legend(items: list[tuple[str, str, str]]) -> str:
    return (
        '<div style="display:flex;flex-direction:column;gap:9px;">'
        + "".join(
            f'<div style="display:flex;align-items:center;gap:9px;">'
            f'<span style="width:8px;height:8px;border-radius:2px;background:{c};flex-shrink:0;"></span>'
            f'<span style="font-size:var(--fs-2);color:{T["text_2"]};flex:1;">{esc(n)}</span>'
            f'<span style="font-size:var(--fs-2);color:{T["text"]};font-weight:600;'
            f'font-variant-numeric:tabular-nums;">{esc(v)}</span></div>'
            for n, v, c in items
        )
        + "</div>"
    )


# ═══════════════════════════════════════════════════════════════════════════
# IMAGE HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def b64(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf.tobytes()).decode() if ok else ""


def png(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes() if ok else b""


def fmt_psnr(v: float) -> str:
    return "∞" if math.isinf(v) else f"{v:.2f}"


def decode_upload(data: bytes) -> np.ndarray | None:
    raw = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    return to_grayscale(img[:, :, ::-1]).astype(np.uint8) if img.ndim == 3 else img.astype(np.uint8)


def img_frame(b64s: str, title: str, tag_html: str) -> str:
    return (
        f'<figure style="margin:0;background:{T["elevated"]};border:1px solid {T["border"]};'
        f'border-radius:var(--r-xl);overflow:hidden;box-shadow:0 2px 10px rgba(0,0,0,.4);">'
        f'<figcaption style="padding:10px 14px;background:rgba(255,255,255,.025);'
        f'border-bottom:1px solid {T["border"]};display:flex;align-items:center;'
        f'justify-content:space-between;gap:8px;">'
        f'<span style="font-size:var(--fs-2);font-weight:700;color:{T["text"]};">{esc(title)}</span>'
        f'{tag_html}</figcaption>'
        f'<div style="padding:10px;"><img src="data:image/png;base64,{b64s}" alt="{esc(title)}" '
        f'style="width:100%;border-radius:var(--r-md);display:block;image-rendering:pixelated;"/></div></figure>'
    )


def comparison_slider(before: str, after: str, uid: str = "cmp") -> str:
    # The frame is capped rather than filling the column. `width:100%` on a
    # 224x224 sample stretched it to 1092x1092 — a 4.9x upscale of a small
    # image, wrapped in a 1094px-tall block that was mostly empty. 560px keeps
    # the upscale modest and the whole control on screen at once.
    return f"""
<div style="margin:12px auto 6px;max-width:560px;">
  <div id="w_{uid}" style="position:relative;overflow:hidden;border-radius:var(--r-lg);cursor:col-resize;
       user-select:none;border:1px solid {T['border']};" role="group" aria-label="Before and after comparison">
    <img src="data:image/png;base64,{after}" alt="After" style="width:100%;display:block;" draggable="false"/>
    <div id="c_{uid}" style="position:absolute;inset:0 auto 0 0;width:50%;overflow:hidden;pointer-events:none;">
      <img src="data:image/png;base64,{before}" alt="Before" style="width:200%;max-width:none;display:block;" draggable="false"/>
    </div>
    <div id="h_{uid}" style="position:absolute;top:0;left:50%;transform:translateX(-50%);width:2px;
         height:100%;background:rgba(255,255,255,.9);pointer-events:none;
         box-shadow:0 0 8px rgba(0,201,167,.6);">
      <div style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);width:40px;height:40px;
           background:{T['elevated']};border:2px solid rgba(0,201,167,.6);border-radius:50%;
           box-shadow:0 0 16px rgba(0,201,167,.35),0 3px 12px rgba(0,0,0,.6);
           display:flex;align-items:center;justify-content:center;">
        <svg viewBox="0 0 24 24" style="width:16px;height:16px;fill:none;stroke:{T['accent']};stroke-width:2.2;
             stroke-linecap:round;stroke-linejoin:round;"><path d="m9 7-5 5 5 5M15 7l5 5-5 5"/></svg>
      </div>
    </div>
    <span style="position:absolute;top:12px;left:12px;padding:4px 10px;border-radius:var(--r-xs);font-size:var(--fs-0);
          font-weight:700;letter-spacing:.09em;color:{T['text']};background:rgba(7,11,20,.82);
          border:1px solid rgba(255,255,255,.12);">BEFORE</span>
    <span style="position:absolute;top:12px;right:12px;padding:4px 10px;border-radius:var(--r-xs);font-size:var(--fs-0);
          font-weight:700;letter-spacing:.09em;color:{T['text']};background:rgba(7,11,20,.82);
          border:1px solid rgba(255,255,255,.12);">AFTER</span>
  </div>
  <div style="text-align:center;font-size:var(--fs-1);color:{T['text_3']};margin-top:8px;">
    Drag to compare · input on the left, filtered output on the right
  </div>
</div>
<script>
(function(){{
  var w=document.getElementById('w_{uid}'),c=document.getElementById('c_{uid}'),
      h=document.getElementById('h_{uid}'),d=false;
  if(!w) return;
  function go(x){{var r=w.getBoundingClientRect(),p=Math.min(.99,Math.max(.01,(x-r.left)/r.width));
    c.style.width=(p*100)+'%';h.style.left=(p*100)+'%';}}
  w.addEventListener('mousedown',function(e){{d=true;go(e.clientX);e.preventDefault();}});
  document.addEventListener('mousemove',function(e){{if(d)go(e.clientX);}});
  document.addEventListener('mouseup',function(){{d=false;}});
  w.addEventListener('touchstart',function(e){{d=true;go(e.touches[0].clientX);}},{{passive:true}});
  document.addEventListener('touchmove',function(e){{if(d)go(e.touches[0].clientX);}},{{passive:true}});
  document.addEventListener('touchend',function(){{d=false;}});
}})();
</script>"""


# ═══════════════════════════════════════════════════════════════════════════
# REAL DATA — config, environment, and the session job store
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_cfg():
    return load_inference_config(), load_dataset_config(), load_hardware_config()


@st.cache_resource(show_spinner=False)
def load_clf(ckpt: str):
    if not _CLASSIFIER_AVAILABLE:
        return None
    p = Path(ckpt)
    if not p.exists():
        return None
    try:
        return _load_classifier(p, load_inference_config())
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def samples(n: int, w: int, h: int, seed: int) -> dict[str, np.ndarray]:
    from denoising.config import ImageConfig
    return {s.source_id: s.image for s in synthetic_sources(n, ImageConfig(w, h, True), seed)}


@st.cache_data(show_spinner=False)
def rtl_inventory() -> list[tuple[str, int]]:
    """Actual .sv files on disk with their byte size. Empty if none."""
    d = _ROOT / "rtl"
    if not d.is_dir():
        return []
    return sorted((f.name, f.stat().st_size) for f in d.glob("*.sv"))


def jobs() -> list[dict]:
    return st.session_state.setdefault("jobs", [])


def record_job(result, ref, label: str) -> None:
    """Append one REAL completed run. Nothing else ever writes to this store."""
    h, w = result.input.shape
    m, nm = result.metrics, result.noisy_metrics
    jobs().append({
        "ts":        datetime.now(),
        "label":     label,
        "w": w, "h": h,
        "noise":     result.noise_class,
        "filter":    result.selected_filter,
        "conf":      result.confidence,
        "fallback":  result.decision.used_fallback,
        "code":      result.decision.control_code,
        "mse":       m.mse  if m else None,
        "psnr":      m.psnr if m else None,
        "ssim":      m.ssim if m else None,
        "mse0":      nm.mse  if nm else None,
        "psnr0":     nm.psnr if nm else None,
        "gain":      result.psnr_improvement,
        "ms":        float(result.timings_ms.get("total", 0.0)),
        "ms_filter": float(result.timings_ms.get("filter", 0.0)),
        "referenced": ref is not None,
    })


def job_stats() -> dict:
    J = jobs()
    if not J:
        return {"n": 0}
    times = [j["ms"] for j in J]
    gains = [j["gain"] for j in J if j["gain"] is not None]
    px = sum(j["w"] * j["h"] for j in J)
    return {
        "n": len(J),
        "avg_ms": sum(times) / len(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "gains": gains,
        "avg_gain": (sum(gains) / len(gains)) if gains else None,
        "px": px,
        "referenced": sum(1 for j in J if j["referenced"]),
    }


# ═══════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════

def init() -> None:
    for k, v in dict(
        nav="Dashboard", step=1, mode="quick",
        image=None, reference=None, truth=None, result=None,
        source_label=None, dev_mode=False, use_ai=True, toast=None,
    ).items():
        st.session_state.setdefault(k, v)
    st.session_state.setdefault("jobs", [])


def reset_wizard() -> None:
    for k in ("step", "image", "reference", "truth", "result", "source_label"):
        st.session_state.pop(k, None)
    init()


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════

NAV_GROUPS = [
    ("Primary",        [("Dashboard", "dashboard"), ("New Processing", "process"),
                        ("Projects", "projects"), ("Processing History", "history")]),
    ("Infrastructure", [("FPGA Devices", "fpga"), ("API", "api")]),
    ("Insights",       [("Analytics", "analytics")]),
    ("Account",        [("Billing", "billing"), ("Settings", "settings")]),
]


def sb_status(label: str, value: str, color: str) -> str:
    return (
        f'<div style="display:flex;align-items:center;justify-content:space-between;padding:5px 0;">'
        f'<span class="sb" style="font-size:var(--fs-1);color:#7C8AA4;">{esc(label)}</span>'
        f'<span class="sb" style="display:inline-flex;align-items:center;gap:6px;font-size:var(--fs-1);'
        f'color:#AEBACE;font-weight:500;"><span style="width:5px;height:5px;border-radius:50%;'
        f'background:{color};flex-shrink:0;"></span>{esc(value)}</span></div>'
    )


def sidebar(clf_ready: bool, hw) -> None:
    ss = st.session_state
    n_rtl = len(rtl_inventory())

    st.sidebar.markdown(
        '<div style="padding:20px 20px 16px;">'
        '<div style="display:flex;align-items:center;gap:13px;">'
        '<div class="sb-brand-icon" style="width:38px;height:38px;border-radius:var(--r-lg);flex-shrink:0;'
        'background:linear-gradient(145deg,#1DEDC3 0%,#00A889 100%);'
        'box-shadow:inset 0 1px 0 rgba(255,255,255,.3),0 4px 16px rgba(0,201,167,.45),0 0 0 1px rgba(0,201,167,.2);'
        'display:flex;align-items:center;justify-content:center;">'
        '<svg viewBox="0 0 24 24" style="width:18px;height:18px;fill:none;stroke:#fff;stroke-width:2.1;'
        'stroke-linecap:round;stroke-linejoin:round;"><path d="m12 3 8 4.5-8 4.5-8-4.5L12 3z"/>'
        '<path d="m4 12.5 8 4.5 8-4.5"/></svg></div>'
        '<div style="min-width:0;">'
        '<div class="sb-brand-name" style="font-size:var(--fs-4);font-weight:700;color:#EDF3F9;'
        'letter-spacing:-0.025em;line-height:1.2;white-space:nowrap;">'
        'AdaptiveDenoise</div>'
        '<div style="font-size:var(--fs-0);color:rgba(0,201,167,.65);margin-top:3px;letter-spacing:.08em;'
        'font-weight:700;text-transform:uppercase;white-space:nowrap;">'
        'AI · FPGA · VISION</div></div></div></div>'
        '<div style="height:1px;margin:0 14px 4px;background:linear-gradient(90deg,transparent,'
        'rgba(99,130,170,.15) 25%,rgba(99,130,170,.15) 75%,transparent);"></div>',
        unsafe_allow_html=True,
    )

    for gi, (group, items) in enumerate(NAV_GROUPS):
        st.sidebar.markdown(
            f'<div class="nav-group{" nav-group-first" if gi == 0 else ""}">{group}</div>',
            unsafe_allow_html=True,
        )
        for label, key in items:
            active = ss.nav == label
            st.sidebar.markdown(
                f'<div class="nav-row {"nav-active" if active else "nav-idle"}" aria-hidden="true" '
                f'title="{esc(label)}">{ico(key,17)}<span class="nav-label">{esc(label)}</span></div>',
                unsafe_allow_html=True,
            )
            if st.sidebar.button(label, key=f"nav_{label}", use_container_width=True):
                ss.nav = label
                if label == "New Processing":
                    reset_wizard()
                st.rerun()

    st.sidebar.markdown(
        '<div class="sb-detail"><div style="height:1px;margin:20px 16px 0;'
        'background:linear-gradient(90deg,transparent,rgba(148,163,184,.11) 20%,'
        'rgba(148,163,184,.11) 80%,transparent);"></div>'
        '<div class="nav-group" style="margin:16px 0 5px;">System</div>'
        '<div style="margin:0 12px;padding:10px 13px;border-radius:var(--r-lg);background:rgba(255,255,255,.018);'
        'border:1px solid rgba(99,130,170,.09);box-shadow:0 1px 4px rgba(0,0,0,.25);">'
        + sb_status("Classifier", "Loaded" if clf_ready else "Not trained",
                    T["ok"] if clf_ready else T["warn"])
        + sb_status("Compute", "CPU (software)", T["accent"])
        + sb_status("RTL sources", f"{n_rtl} module{'s' if n_rtl != 1 else ''}",
                    T["ok"] if n_rtl else T["text_3"])
        + sb_status("Synthesis", "Configured" if hw.synthesis.configured else "No target",
                    T["ok"] if hw.synthesis.configured else T["warn"])
        + '</div></div>',
        unsafe_allow_html=True,
    )

    st.sidebar.markdown(
        '<div class="sb-detail nav-group" style="margin:16px 0 2px;">Preferences</div>',
        unsafe_allow_html=True,
    )
    ss.dev_mode = st.sidebar.toggle(
        "Developer mode", value=ss.dev_mode,
        help="Reveals RTL control codes, filter kernel parameters, noise variance and per-stage timings.",
    )

    st.sidebar.markdown(
        f'<div class="sb-detail" style="margin:16px 14px 22px;padding-top:13px;'
        f'border-top:1px solid rgba(148,163,184,.08);">'
        f'<div style="font-size:var(--fs-0);color:#55637E;line-height:1.6;">'
        f'Research build · no board attached.<br/>All timings are CPU wall-clock.</div></div>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════
# PAGE · DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════

def page_dashboard(cfg, ds, hw, clf_ready: bool) -> None:
    ss = st.session_state
    S = job_stats()
    n_rtl = len(rtl_inventory())

    # A greeting was the page title here. On an engineering dashboard the
    # heading should say what the screen is, and the space above the fold
    # should carry system state rather than the time of day.
    st.markdown(
        page_head("System dashboard",
                  "Adaptive image denoising · CNN noise classification · FPGA RTL reference."),
        unsafe_allow_html=True,
    )

    st.markdown(
        status_strip(
            status_chip("FPGA", "Synthesis configured" if hw.synthesis.configured else "No board attached",
                        T["ok"] if hw.synthesis.configured else T["warn"]),
            status_chip("Classifier", "CNN loaded" if clf_ready else "Manual selection",
                        T["ok"] if clf_ready else T["warn"]),
            status_chip("Compute", "CPU software", T["info"]),
            status_chip("Stream", f'{hw.stream.image_width}×{hw.stream.image_height} · {hw.stream.pixel_width}-bit',
                        T["accent"]),
            status_chip("RTL", f'{n_rtl} module{"s" if n_rtl != 1 else ""}',
                        T["ok"] if n_rtl else T["text_3"]),
        ),
        unsafe_allow_html=True,
    )

    c1, c2, _ = st.columns([1.8, 1.5, 3.7])
    with c1:
        if st.button("＋  New processing", type="primary", use_container_width=True):
            ss.nav, ss.step = "New Processing", 1
            reset_wizard()
            ss.nav = "New Processing"
            st.rerun()
    with c2:
        if st.button("View analytics", type="secondary", use_container_width=True):
            ss.nav = "Analytics"
            st.rerun()

    st.markdown(section("System status"), unsafe_allow_html=True)
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.markdown(kpi("fpga", "FPGA target",
                        "Configured" if hw.synthesis.configured else "No board",
                        f"{hw.synthesis.vendor} · {hw.synthesis.device}" if hw.synthesis.configured
                        else "no synthesis vendor or device set",
                        T["ok"] if hw.synthesis.configured else T["warn"]), unsafe_allow_html=True)
    with s2:
        st.markdown(kpi("brain", "Classification",
                        "CNN" if clf_ready else "Manual",
                        "trained checkpoint loaded" if clf_ready else "no checkpoint found",
                        T["ok"] if clf_ready else T["warn"]), unsafe_allow_html=True)
    with s3:
        st.markdown(kpi("cpu", "Stream geometry",
                        f'{hw.stream.image_width}×{hw.stream.image_height}',
                        f'{hw.stream.pixel_width}-bit · {hw.stream.boundary_policy} boundary',
                        T["accent"]), unsafe_allow_html=True)
    with s4:
        st.markdown(kpi("layers", "RTL sources", f'{n_rtl}',
                        f'testbenches pass vs the golden model' if n_rtl
                        else "no modules found on disk",
                        T["ok"] if n_rtl else T["text_3"]), unsafe_allow_html=True)

    st.markdown(section("Session metrics"), unsafe_allow_html=True)

    if S["n"] == 0:
        st.markdown(
            empty("gauge", "No runs recorded yet",
                  "Metrics on this page are computed from images you actually process. "
                  "Nothing is pre-seeded, so there is nothing to average yet.",
                  "Start a run from <b>New processing</b> — the first result populates every "
                  "figure here, plus History and Analytics."),
            unsafe_allow_html=True,
        )
    else:
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(kpi("image", "Images processed", f'{S["n"]}',
                            f'{S["px"]/1e6:.2f} megapixels total', T["accent"]), unsafe_allow_html=True)
        with k2:
            st.markdown(kpi("clock", "Avg processing time", f'{S["avg_ms"]:.1f}<span style="font-size:var(--fs-3);color:{T["text_3"]};"> ms</span>',
                            f'range {S["min_ms"]:.1f}–{S["max_ms"]:.1f} ms · CPU', T["violet"]), unsafe_allow_html=True)
        with k3:
            if S["avg_gain"] is not None:
                st.markdown(kpi("gauge", "Avg PSNR gain",
                                f'{S["avg_gain"]:+.2f}<span style="font-size:var(--fs-3);color:{T["text_3"]};"> dB</span>',
                                f'over {len(S["gains"])} run(s) with a reference',
                                T["ok"] if S["avg_gain"] > 0 else T["warn"]), unsafe_allow_html=True)
            else:
                st.markdown(kpi("gauge", "Avg PSNR gain", '<span style="color:'+T["text_3"]+';">—</span>',
                                "needs a clean reference image", T["text_3"]), unsafe_allow_html=True)
        with k4:
            st.markdown(kpi("brain", "Classification",
                            "CNN" if clf_ready else "Manual",
                            "trained checkpoint loaded" if clf_ready else "no checkpoint found",
                            T["ok"] if clf_ready else T["warn"]), unsafe_allow_html=True)

    # ── environment ──
    st.markdown(section("Environment"), unsafe_allow_html=True)
    e1, e2 = st.columns([1, 1], gap="medium")
    with e1:
        st.markdown(
            card(
                card_title('Processing pipeline')
                + kv_rows([
                    ("Classifier", badge("Loaded", T["ok"], dot=True) if clf_ready
                     else badge("Not trained", T["warn"], dot=True)),
                    ("Confidence threshold", f'{cfg.confidence.threshold:.2f}'),
                    ("Low-confidence fallback", _flabel(cfg.confidence.fallback)),
                    ("Boundary mode", cfg.filters.boundary_mode),
                    ("Compute device", "CPU (software reference)"),
                ])
            ),
            unsafe_allow_html=True,
        )
    with e2:
        st.markdown(
            card(
                card_title('Hardware target')
                + kv_rows([
                    ("RTL modules on disk", f'{n_rtl}'),
                    ("Stream geometry", f'{hw.stream.image_width}×{hw.stream.image_height} · {hw.stream.pixel_width}-bit'),
                    ("Boundary policy", hw.stream.boundary_policy),
                    ("Simulator", hw.simulation.simulator),
                    ("Synthesis target", badge("Configured", T["ok"], dot=True) if hw.synthesis.configured
                     else badge("Not set", T["warn"], dot=True)),
                ])
            ),
            unsafe_allow_html=True,
        )

    if not hw.synthesis.configured:
        st.markdown(
            alert("No FPGA board attached",
                  "<code>configs/hardware.yaml</code> has no synthesis vendor or device, so no hardware "
                  "throughput, utilisation or temperature is reported anywhere in this interface. "
                  "Every timing shown is Python/CPU wall-clock.", "info"),
            unsafe_allow_html=True,
        )

    # ── recent ──
    if S["n"]:
        st.markdown(section("Recent runs"), unsafe_allow_html=True)
        st.markdown(history_table(jobs()[-5:][::-1]), unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE · NEW PROCESSING  (workflow preserved: Upload → Analysis → Result)
# ═══════════════════════════════════════════════════════════════════════════

def stepper(cur: int) -> str:
    steps = [("Upload", "Select an image"), ("Analysis", "Detect noise & pick filter"), ("Result", "Compare & export")]
    out = []
    for i, (name, desc) in enumerate(steps, 1):
        done, active = i < cur, i == cur
        if done:
            circ = (
                f'<div style="width:28px;height:28px;border-radius:50%;background:{T["accent"]};'
                f'box-shadow:0 0 10px rgba(0,201,167,.4);display:flex;align-items:center;'
                f'justify-content:center;flex-shrink:0;">{ico("check",14,"#040F0B",2.5)}</div>'
            )
        elif active:
            circ = (
                f'<div style="width:28px;height:28px;border-radius:50%;background:{T["accent"]};'
                f'box-shadow:0 0 14px rgba(0,201,167,.55);color:#040F0B;display:flex;align-items:center;'
                f'justify-content:center;font-size:var(--fs-2);font-weight:700;flex-shrink:0;">{i}</div>'
            )
        else:
            circ = (
                f'<div style="width:28px;height:28px;border-radius:50%;background:transparent;'
                f'border:1.5px solid {T["border_st"]};color:{T["text_3"]};display:flex;align-items:center;'
                f'justify-content:center;font-size:var(--fs-2);font-weight:600;flex-shrink:0;">{i}</div>'
            )
        tc = T["text"] if active else (T["text_2"] if done else T["text_3"])
        out.append(
            f'<div style="display:flex;align-items:center;gap:10px;min-width:0;">{circ}'
            f'<div style="min-width:0;">'
            f'<div style="font-size:var(--fs-2);font-weight:{"700" if active else "500"};'
            f'color:{tc};line-height:1.2;white-space:nowrap;">{name}</div>'
            f'<div style="font-size:var(--fs-1);color:{T["text_3"]};white-space:nowrap;margin-top:1px;">{desc}</div>'
            f'</div></div>'
        )
        if i < len(steps):
            done_line = i < cur
            out.append(
                f'<div style="flex:1;height:1.5px;min-width:20px;margin:0 10px;border-radius:2px;'
                f'background:{"linear-gradient(90deg," + T["accent"] + "," + T["accent"] + ")" if done_line else T["border"]};">'
                f'</div>'
            )
    return (
        f'<nav aria-label="Progress" style="display:flex;align-items:center;background:{T["elevated"]};'
        f'border:1px solid {T["border"]};border-radius:var(--r-xl);padding:16px 20px;margin-bottom:26px;'
        f'box-shadow:0 2px 8px rgba(0,0,0,.3);">'
        + "".join(out) + "</nav>"
    )


def page_processing(cfg, ds, clf, clf_ready: bool) -> None:
    ss = st.session_state
    st.markdown(
        page_head("New processing",
                  "Upload an image, review the detected noise, and apply the recommended filter.",
                  ["Primary", "New Processing"]),
        unsafe_allow_html=True,
    )
    step = ss.step
    if step == 1 or ss.image is None:
        step1(cfg, ds)
    elif step == 2:
        step2(cfg, clf, clf_ready)
    else:
        step3(cfg)


def step1(cfg, ds) -> None:
    ss = st.session_state
    st.markdown(stepper(1), unsafe_allow_html=True)

    m1, m2, _ = st.columns([1.4, 1.6, 4.0])
    with m1:
        if st.button("Quick", type="primary" if ss.mode == "quick" else "secondary", use_container_width=True):
            ss.mode = "quick"; st.rerun()
    with m2:
        if st.button("Advanced", type="primary" if ss.mode == "advanced" else "secondary", use_container_width=True):
            ss.mode = "advanced"; st.rerun()
    st.markdown(
        f'<div style="font-size:var(--fs-1);color:{T["text_3"]};margin:6px 0 20px;">'
        + ("Quick mode runs the recommended filter with configured defaults."
           if ss.mode == "quick" else
           "Advanced mode exposes filter override and full pipeline parameters.")
        + '</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.15, 1], gap="large")

    with left:
        st.markdown(
            f'<div style="font-size:var(--fs-3);font-weight:600;color:{T["text"]};margin-bottom:10px;">'
            f'Upload an image</div>', unsafe_allow_html=True)
        up = st.file_uploader("Drop a file or browse", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
                              label_visibility="collapsed")
        st.markdown(
            f'<div style="display:flex;gap:18px;margin-top:10px;font-size:var(--fs-1);color:{T["text_3"]};">'
            f'<span>PNG · JPEG · BMP · TIFF</span><span>Converted to 8-bit greyscale</span></div>',
            unsafe_allow_html=True,
        )
        if up is not None:
            img = decode_upload(up.getvalue())
            if img is None:
                st.markdown(
                    alert("Could not read that file",
                          "The file was received but no image could be decoded from it. It may be corrupt "
                          "or use an unsupported encoding. Try re-exporting as PNG or JPEG.", "error"),
                    unsafe_allow_html=True,
                )
            elif img.ndim != 2 or min(img.shape) < 3:
                st.markdown(
                    alert("Image too small",
                          f"A {img.shape[1]}×{img.shape[0]} image is smaller than the 3×3 filter window. "
                          "Use an image at least 3 pixels on each side.", "error"),
                    unsafe_allow_html=True,
                )
            else:
                ss.image, ss.reference, ss.truth = img, None, None
                ss.source_label, ss.step = up.name, 2
                st.rerun()

    with right:
        st.markdown(
            f'<div style="font-size:var(--fs-3);font-weight:600;color:{T["text"]};margin-bottom:10px;">'
            f'Or use a sample</div>', unsafe_allow_html=True)
        pool = samples(6, ds.image.width, ds.image.height, ds.split.seed)
        pick = st.selectbox("Sample image", list(pool), label_visibility="collapsed")
        if pick:
            prev = pool[pick]
            h, w = prev.shape
            st.markdown(
                img_frame(b64(prev), pick, badge(f"{w}×{h}", T["text_3"])),
                unsafe_allow_html=True,
            )
            st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
            kind = st.selectbox("Add synthetic noise", ["None — keep clean", "Salt & pepper", "Gaussian", "Speckle"], index=2)
            amt = None
            if kind == "Salt & pepper":
                amt = st.slider("Fraction of pixels affected", 0.01, 0.30, 0.10, 0.01)
            elif kind == "Gaussian":
                amt = st.slider("Sigma (normalised)", 0.01, 0.30, 0.08, 0.01)
            elif kind == "Speckle":
                amt = st.slider("Variance (normalised)", 0.01, 0.30, 0.08, 0.01)

            st.markdown(
                f'<div style="font-size:var(--fs-1);color:{T["text_3"]};margin:2px 0 12px;line-height:1.55;">'
                f'A sample keeps its clean original, so MSE, PSNR and SSIM can be measured. '
                f'An uploaded photo has no reference and will show no quality metrics.</div>',
                unsafe_allow_html=True,
            )

            if st.button("Continue", type="primary", use_container_width=True):
                clean = pool[pick]
                if kind == "Salt & pepper":
                    noisy, truth = add_salt_pepper_noise(clean, amt, seed=0), "salt_pepper"
                elif kind == "Gaussian":
                    noisy, truth = add_gaussian_noise(clean, 0.0, amt, seed=0), "gaussian"
                elif kind == "Speckle":
                    noisy, truth = add_speckle_noise(clean, amt, seed=0), "speckle"
                else:
                    noisy, truth = clean, "clean"
                ss.image, ss.reference, ss.truth = noisy, clean, truth
                ss.source_label = f"{pick} · {kind.split(' —')[0].lower()}"
                ss.step = 2
                st.rerun()


def step2(cfg, clf, clf_ready: bool) -> None:
    ss = st.session_state
    img, ref, truth = ss.image, ss.reference, ss.truth
    st.markdown(stepper(2), unsafe_allow_html=True)

    left, right = st.columns([1, 1.1], gap="large")

    h, w = img.shape
    var = estimate_noise_variance(img)

    with left:
        st.markdown(img_frame(b64(img), "Input", badge(f"{w}×{h}", T["text_3"])), unsafe_allow_html=True)
        st.markdown(
            card(kv_rows([
                ("Source", esc(ss.source_label or "uploaded")),
                ("Dimensions", f"{w} × {h} px"),
                ("Format", "8-bit greyscale"),
                ("Clean reference", badge("Available", T["ok"], dot=True) if ref is not None
                 else badge("None", T["text_3"], dot=True)),
            ]), pad="16px", extra="margin-top:12px;"),
            unsafe_allow_html=True,
        )

    with right:
        use_ai = clf_ready and ss.use_ai
        if use_ai:
            ph = st.empty()
            ph.markdown(
                card(
                    f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:14px;">'
                    f'<div style="width:15px;height:15px;border:2px solid {T["accent"]};'
                    f'border-top-color:transparent;border-radius:50%;animation:spin .8s linear infinite;"></div>'
                    f'<span style="font-size:var(--fs-2);color:{T["text_2"]};">Classifying noise…</span></div>'
                    f'<div class="skel" style="height:11px;width:60%;margin-bottom:9px;"></div>'
                    f'<div class="skel" style="height:11px;width:85%;margin-bottom:9px;"></div>'
                    f'<div class="skel" style="height:11px;width:45%;"></div>'
                ),
                unsafe_allow_html=True,
            )
            result = process_image(img, cfg, classifier=clf, reference=ref)
            ph.empty()
        else:
            if not clf_ready:
                st.markdown(
                    alert("No trained classifier",
                          "No checkpoint was found at <code>models/checkpoints/best_model.pt</code>, so the "
                          "noise class cannot be predicted. Choose it manually below, or train a model with "
                          "<code>python scripts/train.py</code>.", "warning"),
                    unsafe_allow_html=True,
                )
            idx = CLASSES.index(truth) if truth in CLASSES else 1
            mc = st.selectbox("Noise class", list(CLASSES), index=idx, format_func=_label)
            result = process_image(img, cfg, noise_class=mc, reference=ref)

        ss.result = result
        rec = result.selected_filter
        nc, fc = _ncolor(result.noise_class), _fcolor(rec)
        conf = result.confidence

        body = (
            f'<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:16px;">'
            f'<span style="font-size:var(--fs-2);font-weight:600;color:{T["text"]};">'
            f'{"Detection" if use_ai else "Manual classification"}</span>'
            f'{badge("CNN", T["accent"]) if use_ai else badge("Manual", T["text_3"])}</div>'

            f'<div style="font-size:var(--fs-1);color:{T["text_3"]};font-weight:600;letter-spacing:.06em;'
            f'text-transform:uppercase;margin-bottom:7px;">Detected noise</div>'
            f'<div style="margin-bottom:16px;">{badge(_label(result.noise_class), nc, dot=True)}</div>'

            f'<div style="font-size:var(--fs-1);color:{T["text_3"]};font-weight:600;letter-spacing:.06em;'
            f'text-transform:uppercase;margin-bottom:7px;">Confidence</div>'
        )
        if conf is not None:
            cc = T["ok"] if conf >= cfg.confidence.threshold else T["warn"]
            body += (
                f'<div style="display:flex;align-items:center;gap:11px;margin-bottom:16px;">'
                f'<span style="font-size:var(--fs-5);font-weight:700;color:{cc};font-variant-numeric:tabular-nums;">'
                f'{conf*100:.1f}%</span><div style="flex:1;">{meter(conf*100, cc, 5)}</div></div>'
            )
        else:
            body += (
                f'<div style="font-size:var(--fs-2);color:{T["text_3"]};margin-bottom:16px;">'
                f'n/a — a manual choice carries no measured confidence.</div>'
            )

        body += (
            f'<div style="font-size:var(--fs-1);color:{T["text_3"]};font-weight:600;letter-spacing:.06em;'
            f'text-transform:uppercase;margin-bottom:7px;">Selected filter</div>'
            f'<div style="padding:13px 15px;border-radius:var(--r-md);background:{fc}0F;border:1px solid {fc}2E;">'
            f'<div style="font-size:var(--fs-3);font-weight:600;color:{fc};">{_flabel(rec)}</div>'
            f'<div style="font-size:var(--fs-1);color:{T["text_2"]};margin-top:4px;line-height:1.5;">'
            f'{esc(FILTER_META[rec][2])}</div></div>'
        )
        st.markdown(card(body), unsafe_allow_html=True)

        if result.decision.used_fallback:
            st.markdown(
                alert("Low-confidence fallback applied",
                      f"Confidence fell below the configured threshold of "
                      f"{cfg.confidence.threshold:.2f}, so the pipeline used the "
                      f"<b>{_flabel(cfg.confidence.fallback)}</b> instead of the class-matched filter.",
                      "warning"),
                unsafe_allow_html=True,
            )

        if truth and truth != "clean" and truth != result.noise_class:
            who = "The classifier" if use_ai else "The manual selection"
            st.markdown(
                alert("Classification does not match the applied noise",
                      f"{who} reported <b>{_label(result.noise_class)}</b>, but "
                      f"<b>{_label(truth)}</b> was the noise actually added. The filter below is "
                      f"the one chosen for the reported class — metrics will show the real consequence.",
                      "warning"),
                unsafe_allow_html=True,
            )

    if ss.mode == "advanced":
        with st.expander("Pipeline parameters"):
            a, b = st.columns(2)
            with a:
                st.markdown(kv_rows([
                    ("Median kernel", f'{cfg.filters.median.kernel_size}×{cfg.filters.median.kernel_size}'),
                    ("Gaussian kernel", f'{cfg.filters.gaussian.kernel_size}×{cfg.filters.gaussian.kernel_size}'),
                    ("Gaussian sigma", f'{cfg.filters.gaussian.sigma:g}'),
                    ("Integer kernel", "yes" if cfg.filters.gaussian.integer_kernel else "no"),
                ]), unsafe_allow_html=True)
            with b:
                nv = cfg.filters.wiener.noise_variance
                st.markdown(kv_rows([
                    ("Wiener kernel", f'{cfg.filters.wiener.kernel_size}×{cfg.filters.wiener.kernel_size}'),
                    ("Wiener noise variance", f'{nv:g}' if nv is not None else "estimated per-image"),
                    ("Estimated variance", f'{var:.2f}'),
                    ("RTL control code", f"2'b{result.decision.control_code:02b}"),
                ]), unsafe_allow_html=True)

    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
    a, b, _ = st.columns([1.9, 1.2, 3.9])
    with a:
        if st.button(f"Apply {_flabel(result.selected_filter).lower()}", type="primary", use_container_width=True):
            ss.step = 3
            st.rerun()
    with b:
        if st.button("Back", type="secondary", use_container_width=True):
            reset_wizard()
            st.rerun()


def step3(cfg) -> None:
    ss = st.session_state
    result, ref = ss.result, ss.reference
    st.markdown(stepper(3), unsafe_allow_html=True)

    if not ss.get("_recorded"):
        record_job(result, ref, ss.source_label or "uploaded image")
        ss._recorded = True

        stages = [("Reading image", .18), ("Classifying noise", .42),
                  ("Selecting filter", .60), ("Applying filter", .88), ("Measuring quality", 1.0)]
        slot, bar = st.empty(), st.progress(0)
        for name, frac in stages:
            slot.markdown(
                f'<div style="display:flex;align-items:center;gap:10px;padding:2px 0;">'
                f'<div style="width:14px;height:14px;border:2px solid {T["accent"]};'
                f'border-top-color:transparent;border-radius:50%;animation:spin .7s linear infinite;"></div>'
                f'<span style="font-size:var(--fs-2);color:{T["text_2"]};">{name}…</span></div>',
                unsafe_allow_html=True,
            )
            bar.progress(frac)
            time.sleep(0.13)
        slot.empty(); bar.empty()

    nc, fc = _ncolor(result.noise_class), _fcolor(result.selected_filter)
    total_ms = result.timings_ms.get("total", 0.0)
    gain = result.psnr_improvement

    st.markdown(
        f'<div class="rise" style="display:flex;align-items:center;gap:14px;margin-bottom:22px;'
        f'padding:14px 18px;background:{T["elevated"]};border:1px solid {T["border"]};'
        f'border-radius:var(--r-lg);box-shadow:0 2px 8px rgba(0,0,0,.25);">'
        f'<div style="width:36px;height:36px;border-radius:50%;background:{T["ok"]}1F;'
        f'border:1.5px solid {T["ok"]}40;flex-shrink:0;'
        f'display:flex;align-items:center;justify-content:center;">{ico("check",18,T["ok"],2.5)}</div>'
        f'<div><div style="font-size:var(--fs-4);font-weight:700;color:{T["text"]};line-height:1.25;">Processing complete</div>'
        f'<div style="font-size:var(--fs-2);color:{T["text_3"]};margin-top:2px;">{esc(ss.source_label or "uploaded image")}</div></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(kpi("layers", "Noise class", _label(result.noise_class),
                        f"confidence {result.confidence*100:.1f}%" if result.confidence is not None
                        else "manual selection", nc), unsafe_allow_html=True)
    with k2:
        st.markdown(kpi("shield", "Filter applied", _flabel(result.selected_filter),
                        f"RTL code 2'b{result.decision.control_code:02b}" if ss.dev_mode
                        else FILTER_META[result.selected_filter][2].split("—")[0].strip(), fc),
                    unsafe_allow_html=True)
    with k3:
        if gain is not None:
            st.markdown(kpi("gauge", "PSNR gain",
                            f'{gain:+.2f}<span style="font-size:var(--fs-3);color:{T["text_3"]};"> dB</span>',
                            "vs the noisy input", T["ok"] if gain > 0 else T["err"]), unsafe_allow_html=True)
        else:
            st.markdown(kpi("gauge", "PSNR gain", f'<span style="color:{T["text_3"]};">—</span>',
                            "no clean reference supplied", T["text_3"]), unsafe_allow_html=True)
    with k4:
        st.markdown(kpi("clock", "Processing time",
                        f'{total_ms:.1f}<span style="font-size:var(--fs-3);color:{T["text_3"]};"> ms</span>',
                        "CPU wall-clock", T["violet"]), unsafe_allow_html=True)

    # ── comparison ──
    st.markdown(section("Comparison"), unsafe_allow_html=True)
    if ref is not None:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(img_frame(b64(ref), "Clean original", badge("Reference", T["ok"])), unsafe_allow_html=True)
        with c2:
            st.markdown(img_frame(b64(result.input), "Noisy input", badge(_label(result.noise_class), nc)), unsafe_allow_html=True)
        with c3:
            st.markdown(img_frame(b64(result.output), "Denoised output", badge(_flabel(result.selected_filter), fc)), unsafe_allow_html=True)
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(img_frame(b64(result.input), "Input", badge(_label(result.noise_class), nc)), unsafe_allow_html=True)
        with c2:
            st.markdown(img_frame(b64(result.output), "Denoised output", badge(_flabel(result.selected_filter), fc)), unsafe_allow_html=True)

    st.markdown(comparison_slider(b64(result.input), b64(result.output)), unsafe_allow_html=True)

    # ── metrics ──
    st.markdown(section("Quality metrics"), unsafe_allow_html=True)
    if result.metrics is not None:
        m, nm = result.metrics, result.noisy_metrics
        rows = [
            ["MSE", f'{nm.mse:.2f}', f'{m.mse:.2f}',
             f'<span style="color:{T["ok"] if m.mse < nm.mse else T["err"]};">{m.mse - nm.mse:+.2f}</span>',
             "lower is better"],
            ["PSNR", f'{fmt_psnr(nm.psnr)} dB', f'{fmt_psnr(m.psnr)} dB',
             (f'<span style="color:{T["ok"] if gain > 0 else T["err"]};">{gain:+.2f} dB</span>'
              if gain is not None else "—"), "higher is better"],
            ["SSIM", f'{nm.ssim:.4f}', f'{m.ssim:.4f}',
             f'<span style="color:{T["ok"] if m.ssim > nm.ssim else T["err"]};">{m.ssim - nm.ssim:+.4f}</span>',
             "1.0 is identical"],
        ]
        st.markdown(table(["Metric", "Noisy input", "Denoised", "Change", ""], rows, align_r={1, 2, 3}),
                    unsafe_allow_html=True)
        if gain is not None and gain < 0:
            st.markdown(
                alert("This filter reduced image quality",
                      f"PSNR fell by {abs(gain):.2f} dB. The filter applied does not match the noise "
                      f"actually present. The figures above are shown as measured, not adjusted.", "warning"),
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            alert("No quality metrics for this run",
                  esc(result.metrics_note or "no clean reference was supplied") + ".", "info"),
            unsafe_allow_html=True,
        )

    if ss.dev_mode:
        st.markdown(section("Pipeline detail"), unsafe_allow_html=True)
        t = result.timings_ms
        h, w = result.input.shape
        st.markdown(
            card(kv_rows([
                ("Image", f"{w} × {h} px · 8-bit greyscale"),
                ("Estimated noise variance", f"{estimate_noise_variance(result.input):.3f}"),
                ("Class → filter", f"{result.noise_class} → {result.selected_filter}"),
                ("RTL control code", f"2'b{result.decision.control_code:02b} ({result.decision.control_code})"),
                ("Low-confidence fallback", "yes" if result.decision.used_fallback else "no"),
                ("Preprocess", f'{t.get("preprocess",0):.3f} ms'),
                ("Classify", f'{t.get("classify",0):.3f} ms'),
                ("Filter", f'{t.get("filter",0):.3f} ms'),
                ("Total", f'{t.get("total",0):.3f} ms'),
            ])),
            unsafe_allow_html=True,
        )

    # ── actions ──
    st.markdown(section("Export"), unsafe_allow_html=True)
    a, b, c, _ = st.columns([1.4, 1.4, 1.4, 3])
    with a:
        st.download_button("Download output", png(result.output), "denoised.png", "image/png",
                           use_container_width=True)
    with b:
        st.download_button("Download input", png(result.input), "input.png", "image/png",
                           use_container_width=True)
    with c:
        if st.button("Process another", type="secondary", use_container_width=True):
            ss.pop("_recorded", None)
            reset_wizard()
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# PAGE · HISTORY
# ═══════════════════════════════════════════════════════════════════════════

def history_table(J: list[dict]) -> str:
    rows = []
    for j in J:
        conf = f'{j["conf"]*100:.1f}%' if j["conf"] is not None else '<span style="opacity:.5;">n/a</span>'
        if j["gain"] is None:
            gain = '<span style="opacity:.5;">—</span>'
        else:
            gc = T["ok"] if j["gain"] > 0 else T["err"]
            gain = f'<span style="color:{gc};">{j["gain"]:+.2f}</span>'
        status = (badge("Completed", T["ok"], dot=True) if j["gain"] is None or j["gain"] >= 0
                  else badge("Degraded", T["warn"], dot=True))
        rows.append([
            f'<span style="color:{T["text_3"]};">{j["ts"].strftime("%H:%M:%S")}</span>',
            f'<span style="color:{T["text"]};">{esc(j["label"][:34])}</span>',
            f'{j["w"]}×{j["h"]}',
            badge(_label(j["noise"]), _ncolor(j["noise"])),
            badge(_flabel(j["filter"]), _fcolor(j["filter"])),
            conf, gain, f'{j["ms"]:.1f} ms', status,
        ])
    return table(
        ["Time", "Source", "Size", "Noise", "Filter", "Confidence", "PSNR Δ", "Duration", "Status"],
        rows, align_r={2, 5, 6, 7}, flex_col=1,
    )


def page_history() -> None:
    J = jobs()
    st.markdown(
        page_head("Processing history",
                  "Every run performed in this session, in order.",
                  ["Primary", "Processing History"]),
        unsafe_allow_html=True,
    )

    if not J:
        st.markdown(
            empty("history", "No processing history",
                  "This log records runs from the current session. History is held in memory and is "
                  "cleared when the server restarts — there is no database behind it.",
                  "Run an image through <b>New processing</b> and it will appear here immediately."),
            unsafe_allow_html=True,
        )
        return

    f1, f2, f3 = st.columns([1.4, 1.4, 3])
    with f1:
        nf = st.selectbox("Noise class", ["All"] + [_label(c) for c in CLASSES])
    with f2:
        ff = st.selectbox("Filter", ["All"] + [FILTER_META[f][1] for f in FILTER_META])

    view = J[::-1]
    if nf != "All":
        view = [j for j in view if _label(j["noise"]) == nf]
    if ff != "All":
        view = [j for j in view if _flabel(j["filter"]) == ff]

    st.markdown(
        f'<div style="font-size:var(--fs-2);color:{T["text_3"]};margin:14px 0 10px;">'
        f'{len(view)} of {len(J)} run{"s" if len(J) != 1 else ""}</div>',
        unsafe_allow_html=True,
    )
    if not view:
        st.markdown(
            empty("search", "No runs match these filters",
                  "The session has runs recorded, but none with the noise class and filter you selected.",
                  "Reset one of the filters above to “All”."),
            unsafe_allow_html=True,
        )
    else:
        st.markdown(history_table(view), unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE · ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════

def page_analytics() -> None:
    J, S = jobs(), job_stats()
    st.markdown(
        page_head("Analytics",
                  "Derived from runs in this session. No historical or synthetic data is included.",
                  ["Insights", "Analytics"]),
        unsafe_allow_html=True,
    )

    if not J:
        st.markdown(
            empty("analytics", "Nothing to chart yet",
                  "Every chart on this page is computed from real runs. With no runs recorded there is "
                  "no distribution, no timing series and no quality trend to show.",
                  "Process at least two images to see trends; one is enough for the distributions."),
            unsafe_allow_html=True,
        )
        return

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(kpi("image", "Total runs", f'{S["n"]}', "this session", T["accent"]), unsafe_allow_html=True)
    with k2:
        st.markdown(kpi("clock", "Median duration",
                        f'{sorted(j["ms"] for j in J)[len(J)//2]:.1f}<span style="font-size:var(--fs-3);color:{T["text_3"]};"> ms</span>',
                        "CPU wall-clock", T["violet"]), unsafe_allow_html=True)
    with k3:
        st.markdown(kpi("layers", "Megapixels", f'{S["px"]/1e6:.2f}', "total processed", T["cyan"]), unsafe_allow_html=True)
    with k4:
        st.markdown(kpi("gauge", "With reference", f'{S["referenced"]}/{S["n"]}',
                        "runs with measurable quality", T["ok"]), unsafe_allow_html=True)

    st.markdown(section("Distribution"), unsafe_allow_html=True)
    d1, d2 = st.columns([1, 1], gap="medium")

    with d1:
        segs = [(_label(c), sum(1 for j in J if j["noise"] == c), _ncolor(c)) for c in CLASSES]
        st.markdown(
            card(
                card_title('Noise classification') +
                f'<div style="display:flex;align-items:center;gap:24px;">'
                f'<div>{chart_donut(segs)}</div>'
                f'<div style="flex:1;min-width:0;">'
                + legend([(n, str(v), c) for n, v, c in segs]) + '</div></div>'
            ),
            unsafe_allow_html=True,
        )

    with d2:
        fl = list(FILTER_META)
        counts = [sum(1 for j in J if j["filter"] == f) for f in fl]
        st.markdown(
            card(
                card_title('Filter selection')
                + chart_bars([float(c) for c in counts], [FILTER_META[f][1].replace(" filter", "") for f in fl],
                             T["accent"], " runs", 128)
            ),
            unsafe_allow_html=True,
        )

    st.markdown(section("Trends"), unsafe_allow_html=True)
    st.markdown(
        card(
            card_title('Processing duration per run') +
            f'<div style="font-size:var(--fs-1);color:{T["text_3"]};margin-bottom:16px;">'
            f'Milliseconds, CPU wall-clock, in run order</div>'
            + chart_bars([j["ms"] for j in J], [f'#{i+1}' for i in range(len(J))], T["violet"], " ms", 160)
        ),
        unsafe_allow_html=True,
    )

    gains = [(i + 1, j["gain"]) for i, j in enumerate(J) if j["gain"] is not None]
    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
    if len(gains) >= 2:
        st.markdown(
            card(
                card_title('PSNR gain per run') +
                f'<div style="font-size:var(--fs-1);color:{T["text_3"]};margin-bottom:16px;">'
                f'Decibels versus the noisy input · dashed line is zero (no improvement)</div>'
                + chart_line([g for _, g in gains], [f'#{i}' for i, _ in gains], T["ok"], " dB", 160)
            ),
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            card(
                card_title('PSNR gain per run') +
                f'<div style="font-size:var(--fs-2);color:{T["text_2"]};line-height:1.6;">'
                f'Needs at least two runs that supplied a clean reference; '
                f'{len(gains)} recorded so far. Use a sample image with synthetic noise — '
                f'an uploaded photo has no original to measure against.</div>'
            ),
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════════
# PAGE · FPGA DEVICES
# ═══════════════════════════════════════════════════════════════════════════

def page_fpga(hw) -> None:
    inv = rtl_inventory()
    st.markdown(
        page_head("FPGA devices",
                  "RTL sources, stream configuration and synthesis target.",
                  ["Infrastructure", "FPGA Devices"]),
        unsafe_allow_html=True,
    )

    if not hw.synthesis.configured:
        st.markdown(
            alert("No board attached",
                  "<code>configs/hardware.yaml</code> declares no synthesis vendor or device. The file "
                  "states these stay null until a real tool run produces them, so no device list, "
                  "utilisation, temperature or throughput is shown — none has been measured.", "warning"),
            unsafe_allow_html=True,
        )

    st.markdown(section("Stream configuration"), unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.markdown(
            card(kv_rows([
                ("Pixel width", f"{hw.stream.pixel_width}-bit greyscale"),
                ("Frame geometry", f"{hw.stream.image_width} × {hw.stream.image_height}"),
                ("Boundary policy", hw.stream.boundary_policy),
                ("Back-pressure", "enabled" if hw.stream.backpressure else "disabled (valid-only)"),
            ])),
            unsafe_allow_html=True,
        )
    with c2:
        tol = hw.simulation.max_abs_error
        st.markdown(
            card(kv_rows([
                ("Simulator", hw.simulation.simulator),
                ("Median tolerance", f'{tol["median"]} LSB' + (" (bit-exact)" if tol["median"] == 0 else "")),
                ("Gaussian tolerance", f'{tol["gaussian"]} LSB' + (" (bit-exact)" if tol["gaussian"] == 0 else "")),
                ("Wiener tolerance", f'{tol["wiener"]} LSB'),
            ])),
            unsafe_allow_html=True,
        )

    st.markdown(section("RTL sources"), unsafe_allow_html=True)
    if not inv:
        st.markdown(
            empty("cpu", "No RTL sources found",
                  "No <code>.sv</code> files are present in <code>rtl/</code>. The hardware pipeline "
                  "cannot be simulated or synthesised without them.",
                  "Add the SystemVerilog modules to <code>rtl/</code> and reload this page."),
            unsafe_allow_html=True,
        )
    else:
        rows = [[
            f'<span style="font-family:var(--mono);color:{T["text"]};">{esc(name)}</span>',
            f'{size:,} B',
            badge("Present", T["ok"], dot=True),
        ] for name, size in inv]
        st.markdown(table(["Module", "Size", "Status"], rows, align_r={1}), unsafe_allow_html=True)
        st.markdown(
            f'<div style="font-size:var(--fs-1);color:{T["text_3"]};margin-top:12px;line-height:1.6;">'
            f'{len(inv)} source file{"s" if len(inv) != 1 else ""} on disk. Presence is verified by reading '
            f'<code>rtl/</code> — it is not a claim that they have been simulated or synthesised.</div>',
            unsafe_allow_html=True,
        )

    st.markdown(section("Synthesis target"), unsafe_allow_html=True)
    st.markdown(
        card(kv_rows([
            ("Vendor", hw.synthesis.vendor or f'<span style="color:{T["text_3"]};">not set</span>'),
            ("Device", hw.synthesis.device or f'<span style="color:{T["text_3"]};">not set</span>'),
            ("Clock", f'{hw.synthesis.clock_mhz} MHz' if hw.synthesis.clock_mhz
             else f'<span style="color:{T["text_3"]};">not set</span>'),
            ("Tool version", hw.synthesis.tool_version or f'<span style="color:{T["text_3"]};">not set</span>'),
        ])),
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════
# PAGE · API
# ═══════════════════════════════════════════════════════════════════════════

def page_api(cfg) -> None:
    st.markdown(
        page_head("API",
                  "Programmatic access to the denoising pipeline.",
                  ["Infrastructure", "API"]),
        unsafe_allow_html=True,
    )
    st.markdown(
        alert("Python API only",
              "This build exposes no HTTP service, so there are no endpoints, keys or rate limits to show. "
              "The pipeline is importable as a Python package — the examples below are the real, "
              "current interface and run as written.", "info"),
        unsafe_allow_html=True,
    )

    t1, t2, t3 = st.tabs(["Quickstart", "Reference", "Result schema"])

    with t1:
        st.markdown(
            f'<div style="font-size:var(--fs-2);color:{T["text_2"]};margin:14px 0 12px;line-height:1.6;">'
            f'Classify an image and apply the matched filter in one call.</div>',
            unsafe_allow_html=True,
        )
        st.markdown(code_block(
            "import cv2\n"
            "from denoising.config import load_inference_config\n"
            "from denoising.pipeline import process_image\n\n"
            "config = load_inference_config()\n"
            "image  = cv2.imread('noisy.png', cv2.IMREAD_GRAYSCALE)\n\n"
            "# Manual class — confidence is None, deliberately.\n"
            "result = process_image(image, config, noise_class='salt_pepper')\n\n"
            "print(result.selected_filter)   # 'median'\n"
            "print(result.decision.control_code)  # 1  -> RTL 2'b01\n"
            "cv2.imwrite('denoised.png', result.output)"
        ), unsafe_allow_html=True)

        st.markdown(
            f'<div style="font-size:var(--fs-2);color:{T["text_2"]};margin:20px 0 12px;line-height:1.6;">'
            f'With a trained classifier and a clean reference, so quality is measurable.</div>',
            unsafe_allow_html=True,
        )
        st.markdown(code_block(
            "from denoising.model.inference import load_classifier\n\n"
            f"clf = load_classifier('{cfg.model_path.name}', config)\n"
            "result = process_image(image, config, classifier=clf, reference=clean)\n\n"
            "print(result.confidence)         # e.g. 0.981\n"
            "print(result.metrics.psnr)       # dB against the reference\n"
            "print(result.psnr_improvement)   # gain vs the noisy input"
        ), unsafe_allow_html=True)

    with t2:
        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
        st.markdown(table(
            ["Parameter", "Type", "Description"],
            [
                ['<span style="font-family:var(--mono);color:'+T["text"]+';">image</span>', "ndarray",
                 "2-D uint8 greyscale image."],
                ['<span style="font-family:var(--mono);color:'+T["text"]+';">config</span>', "InferenceConfig",
                 "From <code>load_inference_config()</code>."],
                ['<span style="font-family:var(--mono);color:'+T["text"]+';">noise_class</span>', "str | None",
                 "Manual class. Mutually exclusive with <code>classifier</code>."],
                ['<span style="font-family:var(--mono);color:'+T["text"]+';">classifier</span>', "NoiseClassifier | None",
                 "Anything with <code>.predict(image) -> (class, confidence)</code>."],
                ['<span style="font-family:var(--mono);color:'+T["text"]+';">reference</span>', "ndarray | None",
                 "Clean original. Without it no quality metrics are computed."],
                ['<span style="font-family:var(--mono);color:'+T["text"]+';">resize</span>', "bool",
                 "Resize to the configured geometry first. Off by default."],
            ],
            flex_col=2,
        ), unsafe_allow_html=True)
        st.markdown(
            alert("Exactly one of noise_class or classifier",
                  "Passing both, or neither, raises <code>ValueError</code>. A prediction and a manual "
                  "override are different claims and the pipeline refuses to blur them.", "info"),
            unsafe_allow_html=True,
        )

    with t3:
        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
        st.markdown(code_block(
            "result.as_dict() ->\n"
            "{\n"
            "  'noise_class':      'salt_pepper',\n"
            "  'confidence':       0.981 | None,   # None for a manual choice\n"
            "  'selected_filter':  'median',\n"
            "  'used_fallback':    False,\n"
            "  'control_code':     1,              # RTL filter_sel, 2'b01\n"
            "  'metrics':          {'mse': .., 'psnr': .., 'ssim': ..} | None,\n"
            "  'metrics_note':     'no clean reference was supplied, ...' | None,\n"
            "  'psnr_improvement': 8.42 | None,\n"
            "  'timings_ms':       {'preprocess':.., 'classify':.., 'filter':.., 'total':..},\n"
            "}"
        ), unsafe_allow_html=True)
        st.markdown(
            f'<div style="font-size:var(--fs-2);color:{T["text_2"]};margin-top:14px;line-height:1.65;">'
            f'<code>metrics</code> is <code>None</code> whenever no reference was given, and '
            f'<code>metrics_note</code> then explains why. <code>confidence</code> is <code>None</code> '
            f'for a manual class rather than <code>1.0</code> — recording certainty nobody measured '
            f'would put a measurement-shaped number where no measurement happened.</div>',
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════════
# PAGE · PROJECTS  /  BILLING   (no backend — stated, not faked)
# ═══════════════════════════════════════════════════════════════════════════

def page_projects() -> None:
    st.markdown(
        page_head("Projects", "Group related images and processing runs.", ["Primary", "Projects"]),
        unsafe_allow_html=True,
    )
    st.markdown(
        empty("projects", "Projects are not available in this build",
              "Projects need somewhere to persist. This application keeps no database — runs live in "
              "session memory and are lost when the server restarts — so there is nothing to group and "
              "no project to list.",
              "Runs are still tracked for the current session under <b>Processing History</b>."),
        unsafe_allow_html=True,
    )


def page_billing() -> None:
    st.markdown(
        page_head("Billing", "Plan, usage and invoices.", ["Account", "Billing"]),
        unsafe_allow_html=True,
    )
    st.markdown(
        alert("No billing backend",
              "This is a local research build with no account system, metering or payment provider. "
              "Showing a plan, a quota bar or an invoice list here would be inventing records — so "
              "none are shown.", "info"),
        unsafe_allow_html=True,
    )
    st.markdown(
        empty("billing", "Nothing to bill",
              "There is no subscription, no usage meter and no payment method associated with this "
              "installation. Processing runs locally on your own CPU at no metered cost.",
              "Session run counts are visible on the <b>Dashboard</b> and in <b>Analytics</b>."),
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════
# PAGE · SETTINGS
# ═══════════════════════════════════════════════════════════════════════════

def page_settings(cfg, ds, hw, clf_ready: bool) -> None:
    ss = st.session_state
    st.markdown(
        page_head("Settings", "Interface preferences and the loaded configuration.",
                  ["Account", "Settings"]),
        unsafe_allow_html=True,
    )

    t1, t2, t3, t4 = st.tabs(["General", "Processing", "Hardware", "About"])

    with t1:
        st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
        st.markdown(group_label("Classification", first=True), unsafe_allow_html=True)
        if clf_ready:
            ss.use_ai = st.toggle("Classify noise automatically", value=ss.use_ai,
                                  help="Off: choose the noise class by hand on every run.")
            st.markdown(
                f'<div style="font-size:var(--fs-1);color:{T["text_3"]};margin:-6px 0 20px;">'
                f'When off, every run asks you to pick the class and reports no confidence.</div>',
                unsafe_allow_html=True)
        else:
            st.markdown(
                alert("No trained checkpoint",
                      f"Nothing was loaded from <code>{esc(str(cfg.model_path.relative_to(_ROOT)) if cfg.model_path.is_relative_to(_ROOT) else cfg.model_path)}</code>. "
                      "Automatic classification is unavailable until a model is trained; the noise class "
                      "must be chosen manually on each run.", "warning"),
                unsafe_allow_html=True)

        st.markdown(group_label("Display"), unsafe_allow_html=True)
        ss.dev_mode = st.toggle("Developer mode", value=ss.dev_mode,
                                help="Reveals RTL control codes, kernel parameters and per-stage timings.")
        st.markdown(
            f'<div style="font-size:var(--fs-1);color:{T["text_3"]};margin:-6px 0 0;">'
            f'Adds a pipeline-detail panel to results and parameter readouts to the analysis step.</div>',
            unsafe_allow_html=True)

    with t2:
        st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
        st.markdown(
            alert("Read-only",
                  "These values come from <code>configs/inference.yaml</code>. Editing them here would "
                  "desynchronise the UI from the file every other tool reads, so this view does not "
                  "offer controls it cannot honour.", "info"),
            unsafe_allow_html=True)
        a, b = st.columns(2, gap="medium")
        with a:
            st.markdown(card(
                card_title('Selection')
                + kv_rows([
                    ("Confidence threshold", f"{cfg.confidence.threshold:.2f}"),
                    ("Fallback filter", _flabel(cfg.confidence.fallback)),
                    ("Boundary mode", cfg.filters.boundary_mode),
                    ("Model input", f"{cfg.image.width}×{cfg.image.height}"),
                    ("Device", cfg.device),
                ])), unsafe_allow_html=True)
        with b:
            nv = cfg.filters.wiener.noise_variance
            st.markdown(card(
                card_title('Filters')
                + kv_rows([
                    ("Median kernel", f"{cfg.filters.median.kernel_size}×{cfg.filters.median.kernel_size}"),
                    ("Gaussian kernel", f"{cfg.filters.gaussian.kernel_size}×{cfg.filters.gaussian.kernel_size}"),
                    ("Gaussian sigma", f"{cfg.filters.gaussian.sigma:g}"),
                    ("Integer kernel", "yes" if cfg.filters.gaussian.integer_kernel else "no"),
                    ("Wiener variance", f"{nv:g}" if nv is not None else "per-image estimate"),
                ])), unsafe_allow_html=True)

    with t3:
        st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
        st.markdown(card(
            card_title('From configs/hardware.yaml')
            + kv_rows([
                ("Pixel width", f"{hw.stream.pixel_width}-bit"),
                ("Frame", f"{hw.stream.image_width}×{hw.stream.image_height}"),
                ("Boundary policy", hw.stream.boundary_policy),
                ("Back-pressure", "enabled" if hw.stream.backpressure else "disabled"),
                ("Simulator", hw.simulation.simulator),
                ("Synthesis vendor", esc(hw.synthesis.vendor) if hw.synthesis.vendor
                 else badge("Not set", T["warn"], dot=True)),
                ("Synthesis device", esc(hw.synthesis.device) if hw.synthesis.device
                 else badge("Not set", T["warn"], dot=True)),
            ]), max_w=640), unsafe_allow_html=True)

    with t4:
        st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
        st.markdown(card(
            card_title('AdaptiveDenoise')
            + f'<div style="font-size:var(--fs-2);color:{T["text_2"]};line-height:1.7;">'
            f'Adaptive image denoising: a CNN classifies the noise present, and the matched filter '
            f'(median, Gaussian or Wiener) is applied. The same filter selection is implemented in '
            f'SystemVerilog for FPGA acceleration, encoded as a 2-bit control code.</div>'
            f'<div style="height:1px;background:{T["border"]};margin:16px 0;"></div>'
            + kv_rows([
                ("Noise classes", ", ".join(_label(c) for c in CLASSES)),
                ("Filters", ", ".join(FILTER_META[f][1] for f in FILTER_META)),
                ("RTL modules", f"{len(rtl_inventory())} on disk"),
                ("Compute", "CPU — no FPGA board attached"),
            ]), max_w=640), unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    init()

    cfg, ds, hw = load_cfg()
    clf = load_clf(str(cfg.model_path))
    clf_ready = clf is not None

    sidebar(clf_ready, hw)
    nav = st.session_state.nav

    if nav == "Dashboard":
        page_dashboard(cfg, ds, hw, clf_ready)
    elif nav == "New Processing":
        page_processing(cfg, ds, clf, clf_ready)
    elif nav == "Projects":
        page_projects()
    elif nav == "Processing History":
        page_history()
    elif nav == "FPGA Devices":
        page_fpga(hw)
    elif nav == "API":
        page_api(cfg)
    elif nav == "Analytics":
        page_analytics()
    elif nav == "Billing":
        page_billing()
    elif nav == "Settings":
        page_settings(cfg, ds, hw, clf_ready)


import traceback as _tb
try:
    main()
except Exception as _e:
    st.error(f"**Startup error:** {_e}")
    st.code(_tb.format_exc())
