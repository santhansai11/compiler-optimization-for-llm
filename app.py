"""Streamlit UI for the LLM compiler optimizer."""

import numpy as np
import pandas as pd
import streamlit as st

from models.transformer import create_demo_transformer_graph
from pipeline import run_pipeline
from utils.metrics import compute_metrics, format_report
from utils.visualization import graph_to_dot

st.set_page_config(
    page_title="LLM Compiler Optimizer",
    page_icon="⚡",
    layout="wide",
)

CUSTOM_CSS = """
<style>
    /* ---- hide sidebar & chrome ---- */
    [data-testid="stSidebar"]        { display: none !important; }
    [data-testid="collapsedControl"] { display: none !important; }
    .block-container {
        padding-top: 4.5rem !important;
        padding-bottom: 3rem !important;
        max-width: 100% !important;
        padding-left: 3rem !important;
        padding-right: 3rem !important;
    }
    #MainMenu, footer { visibility: hidden; }

    /* ---- DARK BASE ---- */
    .stApp,
    section[data-testid="stMain"],
    div[data-testid="stAppViewContainer"] { background-color: #0a0a0a !important; }

    /* ---- Aurora background glows ---- */
    .stApp::before {
        content: ""; position: fixed; inset: 0; z-index: 0;
        pointer-events: none;
        background:
            radial-gradient(640px 420px at 10% -6%, rgba(99,102,241,.16), transparent 60%),
            radial-gradient(760px 460px at 92% -2%, rgba(168,85,247,.12), transparent 60%),
            radial-gradient(900px 520px at 50% 112%, rgba(56,189,248,.07), transparent 60%);
    }
    .stApp > div { position: relative; z-index: 1; }

    /* ---- Custom scrollbar ---- */
    ::-webkit-scrollbar { width: 10px; height: 10px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb {
        background: #27272a; border-radius: 999px;
        border: 2px solid #0a0a0a;
    }
    ::-webkit-scrollbar-thumb:hover { background: #3f3f46; }
    ::selection { background: rgba(129,140,248,.35); color: #fff; }
    .stApp > header { background: #0a0a0a !important;
                      border-bottom: 1px solid rgba(255,255,255,.07) !important; }

    /* ---- NAV ROW wrapper — flex: logo | spacer | nav items | run ---- */
    .nav-cols-row {
        background: #080808 !important;
        border-bottom: 1px solid rgba(255,255,255,0.05) !important;
        padding: 0 24px;
        margin-bottom: 0 !important;
    }
    .nav-cols-row > div[data-testid="stHorizontalBlock"] {
        display: flex !important;
        align-items: center !important;
        min-height: 52px !important;
        gap: 12px !important;
        width: 100% !important;
    }
    /* 1) Logo column */
    .nav-cols-row > div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(1) {
        flex: 0 0 auto !important;
        width: auto !important;
        display: flex !important;
        align-items: center !important;
    }
    /* 2) Spacer column */
    .nav-cols-row > div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(2) {
        flex: 1 1 auto !important;
        width: auto !important;
    }
    /* 3+) Nav items and Run */
    .nav-cols-row > div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(n+3) {
        flex: 0 0 max-content !important;
        width: max-content !important;
        min-width: max-content !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        padding: 0 8px !important;
    }

    /* ---- Logo ---- */
    .nav-logo-md {
        display: flex;
        align-items: center;
        gap: 8px;
        white-space: nowrap;
    }
    .nav-logo-md svg {
        width: 22px;
        height: 22px;
        flex-shrink: 0;
    }
    .nav-logo-md .logo-text {
        font-size: 1.4rem;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: -.02em;
        white-space: nowrap;
    }

    /* ---- Popover nav buttons — Linear style: plain text, no borders ---- */
    div[data-testid="stPopover"] button {
        background: transparent !important;
        background-color: transparent !important;
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
        border-radius: 6px !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        color: #9ca3af !important;
        padding: 5px 12px !important;
        height: auto !important;
        width: max-content !important;
        min-width: max-content !important;
        white-space: nowrap !important;
        transition: color 150ms ease, background-color 150ms ease !important;
    }
    div[data-testid="stPopover"] button p {
        color: #9ca3af !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
        margin: 0 !important;
    }
    /* Hover and Open (Active) state: subtle background and white text */
    div[data-testid="stPopover"] button:hover,
    div[data-testid="stPopover"] button[aria-expanded="true"] {
        color: #ffffff !important;
        background: rgba(255,255,255,0.08) !important;
        background-color: rgba(255,255,255,0.08) !important;
    }
    div[data-testid="stPopover"] button:hover p,
    div[data-testid="stPopover"] button[aria-expanded="true"] p {
        color: #ffffff !important;
    }
    div[data-testid="stPopover"] button:focus,
    div[data-testid="stPopover"] button:active {
        outline: none !important;
        box-shadow: none !important;
        border: none !important;
    }

    /* ---- Run — SOLID white pill like Linear "Sign up" ---- */
    div[data-testid="stButton"] button[kind="primary"] {
        background: #ffffff !important;
        background-color: #ffffff !important;
        color: #000000 !important;
        border: none !important;
        border-radius: 999px !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        padding: 5px 16px !important;
        box-shadow: none !important;
        height: auto !important;
        transition: background 120ms ease !important;
    }
    div[data-testid="stButton"] button[kind="primary"] p {
        color: #000000 !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        margin: 0 !important;
    }
    div[data-testid="stButton"] button[kind="primary"]:hover {
        background: #e5e7eb !important;
        background-color: #e5e7eb !important;
        color: #000000 !important;
    }

    /* ---- Global text on dark ---- */
    p, span, label, li { color: #d1d5db !important; }
    h1, h2, h3, h4, h5, h6 { color: #f3f4f6 !important; }
    .stAlert, [data-testid="stAlert"] {
        background: rgba(255,255,255,.05) !important;
        border: 1px solid rgba(255,255,255,.1) !important;
        border-radius: 12px !important;
    }
    [data-testid="stAlert"] p { color: #c9d1d9 !important; }
    hr { border-color: rgba(255,255,255,.08) !important; }

    /* ---- Section title ---- */
    .section-title {
        font-size: 1.05rem; font-weight: 700; color: #f4f4f5;
        letter-spacing: -.01em;
        border-left: 3px solid transparent;
        border-image: linear-gradient(180deg, #6366f1, #a855f7) 1;
        padding-left: 12px;
        margin: 28px 0 14px 0;
    }

    /* ---- Logo badge ---- */
    .logo-badge {
        width: 27px; height: 27px; border-radius: 8px; flex-shrink: 0;
        background: linear-gradient(135deg, #6366f1, #a855f7);
        display: inline-flex; align-items: center; justify-content: center;
        font-size: .85rem; line-height: 1;
        box-shadow: 0 2px 12px rgba(139,92,246,.45);
    }

    /* ---- Hero banner ---- */
    .hero {
        position: relative; overflow: hidden;
        background:
            radial-gradient(560px 260px at 88% -10%, rgba(255,255,255,.16), transparent 60%),
            linear-gradient(120deg, #312e81 0%, #5b21b6 55%, #7e22ce 100%);
        border: 1px solid rgba(255,255,255,.12);
        border-radius: 20px; padding: 22px 28px; color: #fff;
        display: flex; justify-content: space-between; align-items: center;
        flex-wrap: wrap; gap: 14px;
        box-shadow: 0 14px 48px rgba(79,70,229,.38);
    }
    .hero::after {
        content: ""; position: absolute; inset: 0; pointer-events: none;
        background-image:
            linear-gradient(rgba(255,255,255,.045) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255,255,255,.045) 1px, transparent 1px);
        background-size: 26px 26px;
        mask-image: linear-gradient(120deg, rgba(0,0,0,.9), transparent 65%);
        -webkit-mask-image: linear-gradient(120deg, rgba(0,0,0,.9), transparent 65%);
    }
    .hero-kicker { font-size: .74rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; opacity: .8; color:#fff !important; }
    .hero-big    {
        font-size: 2.9rem; font-weight: 800; line-height: 1.05;
        background: linear-gradient(90deg, #ffffff 30%, #e9d5ff 100%);
        -webkit-background-clip: text; background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero > div { position: relative; z-index: 1; }
    .hero-sub    { font-size: .9rem; opacity: .85; margin-top: 2px; color: #fff !important; }
    .hero-chips  { display: flex; gap: 8px; flex-wrap: wrap; justify-content: flex-end; max-width: 460px; }
    .hchip { background: rgba(255,255,255,.12); border: 1px solid rgba(255,255,255,.25);
             padding: 6px 12px; border-radius: 999px; font-size: .8rem; font-weight: 600; color: #fff !important; }

    /* ---- Metric cards ---- */
    .mcard {
        position: relative; overflow: hidden;
        background: linear-gradient(180deg, #16161a 0%, #121214 100%);
        border: 1px solid rgba(255,255,255,.09); border-radius: 16px;
        padding: 14px 16px 12px; height: 100%;
        box-shadow: 0 1px 4px rgba(0,0,0,.5);
        transition: transform .16s ease, border-color .16s ease, box-shadow .16s ease;
    }
    .mcard::before {
        content: ""; position: absolute; top: 0; left: 0; right: 0; height: 2px;
        background: linear-gradient(90deg, #6366f1, #a855f7, #38bdf8);
        opacity: .75;
    }
    .mcard:hover {
        transform: translateY(-2px);
        border-color: rgba(255,255,255,.18);
        box-shadow: 0 8px 28px rgba(0,0,0,.55);
    }
    .mtitle { font-size: .72rem; font-weight: 700; color: #9ca3af !important; letter-spacing: .05em; text-transform: uppercase; }
    .mvalue { font-size: 1.7rem; font-weight: 800; color: #f9fafb !important; line-height: 1.15; margin: 4px 0 8px 0; }
    .mvalue small { font-size: .92rem; font-weight: 600; color: #6b7280 !important; }
    .msub { font-size: .78rem; color: #9ca3af !important; display: flex; gap: 6px; align-items: center; flex-wrap: wrap; min-height: 22px; }

    /* ---- Chips ---- */
    .chip { padding: 2px 9px; border-radius: 999px; font-weight: 700; font-size: .73rem; white-space: nowrap; }
    .chip-good { background: #052e16; color: #4ade80 !important; border: 1px solid #166534; }
    .chip-bad  { background: #2d0a0a; color: #f87171 !important; border: 1px solid #7f1d1d; }
    .chip-flat { background: #1e1b4b; color: #a5b4fc !important; border: 1px solid #3730a3; }

    /* ---- Progress bar ---- */
    .bar { height: 6px; background: rgba(255,255,255,.08); border-radius: 999px; overflow: hidden; margin: 2px 0 8px 0; }
    .bar-fill { height: 100%; border-radius: 999px; background: linear-gradient(90deg, #6366f1, #a855f7); }

    /* ---- Legend & DataFrame ---- */
    .legend { font-size: .8rem; color: #9ca3af; line-height: 1.7; }
    .legend .dot { font-size: .95rem; }
    div[data-testid="stDataFrame"] { border-radius: 12px; }

    /* ---- Tabs ---- */
    [data-testid="stTabs"] [role="tab"]                       { color: #9ca3af !important; }
    [data-testid="stTabs"] [role="tab"][aria-selected="true"] { color: #ffffff !important; border-bottom-color: #6366f1 !important; }

    /* ---- Code ---- */
    /* ---- Linear Dashboard Content Styling ---- */
    .content-wrap {
        max-width: 1040px;
        margin: 0 auto;
        padding: 60px 32px;
    }
    .linear-hero {
        margin-bottom: 4rem;
    }
    .linear-hero h1 {
        font-size: 3.5rem !important;
        font-weight: 700 !important;
        color: #ffffff !important;
        letter-spacing: -0.03em !important;
        line-height: 1.1 !important;
        margin-bottom: 1rem !important;
        max-width: 900px;
    }
    .linear-hero p {
        font-size: 1.25rem !important;
        color: #a1a1aa !important;
        line-height: 1.5 !important;
        max-width: 700px !important;
    }
    .linear-card-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 24px;
        margin-bottom: 24px;
    }
    .linear-card {
        background: #161618;
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 32px;
        box-shadow: 0 4px 24px rgba(0,0,0,0.2);
    }
    .linear-card h2 {
        font-size: 1.1rem !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        margin-top: 0 !important;
        margin-bottom: 1.5rem !important;
        padding-bottom: 1rem !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05) !important;
    }
    .linear-card p, .linear-card li, .linear-card div {
        color: #a1a1aa !important;
        font-size: 0.95rem !important;
        line-height: 1.6 !important;
        margin-bottom: 1rem !important;
    }
    .linear-card strong {
        color: #e4e4e7 !important;
        font-weight: 600 !important;
    }
    .pass-list {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 32px 24px;
    }
    .pass-list div { margin-bottom: 0 !important; }

    /* ---- Landing page: eyebrow, gradient text, flow diagram, cards ---- */
    .hero-eyebrow {
        display: inline-flex; align-items: center; gap: 9px;
        font-size: .72rem; font-weight: 700; letter-spacing: .16em;
        color: #a5b4fc !important;
        background: rgba(99,102,241,.08);
        border: 1px solid rgba(99,102,241,.28);
        border-radius: 999px; padding: 6px 14px; margin-bottom: 20px;
    }
    .eyebrow-dot {
        width: 7px; height: 7px; border-radius: 50%;
        background: #34d399; box-shadow: 0 0 10px #34d399;
    }
    .grad-text {
        background: linear-gradient(92deg, #a5b4fc 0%, #c084fc 52%, #7dd3fc 100%);
        -webkit-background-clip: text; background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .cta-hint {
        display: inline-flex; align-items: center; gap: 8px;
        margin-top: 22px; font-size: .84rem; color: #d4d4d8 !important;
        background: rgba(255,255,255,.04);
        border: 1px solid rgba(255,255,255,.1);
        border-radius: 10px; padding: 9px 14px;
    }
    .cta-hint strong { color: #fff !important; }

    .flow-wrap { margin: 34px 0 40px 0; }
    .flow-title {
        font-size: .72rem; letter-spacing: .12em; text-transform: uppercase;
        color: #71717a !important; font-weight: 700; margin-bottom: 12px;
    }
    .flow { display: flex; align-items: stretch; gap: 10px; flex-wrap: wrap; }
    .flow-step {
        display: flex; align-items: center; gap: 10px;
        background: linear-gradient(180deg, #17171a, #131316);
        border: 1px solid rgba(255,255,255,.08); border-radius: 12px;
        padding: 10px 14px;
        transition: border-color .16s ease, transform .16s ease;
    }
    .flow-step:hover { border-color: rgba(168,85,247,.4); transform: translateY(-2px); }
    .flow-step strong { color: #f4f4f5 !important; font-size: .88rem; display: block; line-height: 1.2; }
    .flow-step small { color: #8b8b94 !important; font-size: .72rem; }
    .fnum {
        width: 22px; height: 22px; border-radius: 7px; flex-shrink: 0;
        background: linear-gradient(135deg, #312e81, #6d28d9);
        color: #fff !important; font-size: .72rem; font-weight: 700;
        display: inline-flex; align-items: center; justify-content: center;
    }
    .flow-accent {
        border-color: rgba(168,85,247,.45);
        box-shadow: 0 0 0 1px rgba(168,85,247,.15), 0 4px 20px rgba(168,85,247,.12);
    }
    .flow-arrow { align-self: center; color: #52525b !important; font-size: 1.05rem; }

    .linear-card {
        transition: transform .18s ease, border-color .18s ease, box-shadow .18s ease;
    }
    .linear-card:hover {
        transform: translateY(-3px);
        border-color: rgba(255,255,255,.12);
        box-shadow: 0 12px 44px rgba(0,0,0,.45);
    }
    .lc-head {
        display: flex; align-items: center; gap: 12px;
        margin-bottom: 1.2rem; padding-bottom: 1rem;
        border-bottom: 1px solid rgba(255,255,255,.06);
    }
    .lc-head h2 { margin: 0 !important; padding: 0 !important; border: none !important; }
    .lc-icon {
        width: 34px; height: 34px; border-radius: 10px; flex-shrink: 0;
        display: inline-flex; align-items: center; justify-content: center;
        font-size: 1rem;
    }
    .lc-red    { background: rgba(248,113,113,.12); border: 1px solid rgba(248,113,113,.28); }
    .lc-blue   { background: rgba(96,165,250,.12);  border: 1px solid rgba(96,165,250,.28); }
    .lc-purple { background: rgba(192,132,252,.12); border: 1px solid rgba(192,132,252,.28); }
    .lc-green  { background: rgba(74,222,128,.12);  border: 1px solid rgba(74,222,128,.28); }
    .lc-amber  { background: rgba(251,191,36,.12);  border: 1px solid rgba(251,191,36,.28); }

    .pass-item { display: flex; gap: 12px; align-items: flex-start; }
    .pass-num {
        font-size: .7rem; font-weight: 800; color: #818cf8 !important;
        background: rgba(99,102,241,.1); border: 1px solid rgba(99,102,241,.28);
        border-radius: 7px; padding: 2px 7px; margin-top: 2px;
        height: fit-content; white-space: nowrap;
    }

    /* ---- Tabs hover ---- */
    [data-testid="stTabs"] [role="tab"]:hover { color: #e4e4e7 !important; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def _jsonable(value):
    """Convert numpy scalars/arrays so st.json never chokes."""
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.size > 64:
            return f"<ndarray shape={value.shape} dtype={value.dtype}>"
        return value.tolist()
    return value


@st.cache_resource(show_spinner=False)
def _load_hf_graph(model_key):
    """Import a real LLM computation graph (cached across reruns)."""
    from models.import_hf import import_hf_graph

    return import_hf_graph(model_key)


def _chip(text, kind="flat"):
    return f'<span class="chip chip-{kind}">{text}</span>'


def _card(title, value, sub="", chip=None, chip_kind="flat", bar=None,
          icon=""):
    """Render one metric card as HTML (progress bar + status chip)."""
    chip_html = _chip(chip, chip_kind) if chip else ""
    bar_html = ""
    if bar is not None:
        width = max(0.0, min(100.0, float(bar)))
        bar_html = (
            f'<div class="bar"><div class="bar-fill" '
            f'style="width:{width:.0f}%"></div></div>'
        )
    return (
        f'<div class="mcard"><div class="mtitle">{icon} {title}</div>'
        f'<div class="mvalue">{value}</div>{bar_html}'
        f'<div class="msub">{chip_html}<span>{sub}</span></div></div>'
    )


# ================================================================== nav bar
st.markdown('<div class="nav-cols-row">', unsafe_allow_html=True)

# 1 Logo | 1 Spacer | 5 Popovers | 1 Run = 8 columns total
_logo_col, _spacer, _p1, _p2, _p3, _p4, _p5, _run_col = st.columns(
    [2.5, 4.0, 1.2, 0.9, 1.2, 1.3, 0.9, 0.8], gap="small"
)

with _logo_col:
    # Plain text logo
    st.markdown(
        '''
        <div class="nav-logo-md">
          <span class="logo-badge">⚡</span>
          <span class="logo-text">LLM Compiler Optimizer</span>
        </div>
        ''',
        unsafe_allow_html=True,
    )

with _spacer:
    pass


with _p1:
    with st.popover("Optimization Passes"):
        normalize = st.checkbox(
            "Graph Normalization",
            value=True,
            help="Canonicalize ops, fold constants, CSE, DCE, infer shapes.",
        )
        canonicalize = st.checkbox(
            "Attention Canonicalization",
            value=True,
            help="Fuse Q/K/V → QKᵀ → Scale → Softmax → AV into FusedAttention.",
        )
        operator_merge = st.checkbox(
            "Semantic Operator Merging",
            value=True,
            help="MatMul+Add→GEMM, +GELU→LinearGELU, Add+LayerNorm fusion.",
        )
        dags = st.checkbox(
            "DAGS Scheduling",
            value=True,
            help="Levels, critical path and a deterministic list schedule.",
        )

with _p2:
    with st.popover("Partitioning"):
        partition_mode = st.radio(
            "Mode",
            ["Off", "Graph"],
            index=0,
            label_visibility="collapsed",
        )
        num_partitions = st.select_slider(
            "Number of Partitions",
            options=[2, 3, 4],
            value=2,
            disabled=partition_mode == "Off",
        )

with _p3:
    with st.popover("Learned Search"):
        neuro_symbolic = st.checkbox(
            "Neuro-Symbolic Rewrite Search",
            value=False,
            help="GNN-guided symbolic rewrite rules search for extra fusion on "
                 "top of the selected passes.",
        )
        rl_search = st.checkbox(
            "RL Pass-Order Search (Q-learning)",
            value=False,
            help="Search the pass subset with tabular Q-learning and compile "
                 "with the recommended combination.",
        )

with _p4:
    with st.popover("Dataset & Training"):
        train_on_dataset = st.checkbox(
            "Train MLP on MNIST + compile it",
            value=False,
            help="Train a small classifier on the MNIST dataset (NumPy, manual "
                 "backprop), export the trained weights into the IR and compile "
                 "them. Cached after the first run: the first run trains (~1 "
                 "min), later runs load the cached model.",
        )

with _p5:
    with st.popover("Input Model"):
        from models.import_hf import HF_MODELS, is_available

        choices = [
            "Demo Transformer (2 blocks)",
            "Demo Transformer (4 blocks)",
        ]
        real_available = is_available()
        if real_available:
            choices += [label for label, _, _ in HF_MODELS.values()]
        model_name = st.radio(
            "Model",
            choices,
            label_visibility="collapsed",
            index=0,
        )
        hf_info = None
        hf_label = next(
            (
                label
                for label, _, _ in HF_MODELS.values()
                if label == model_name
            ),
            None,
        )
        if hf_label is not None:
            hf_key = next(
                key for key, (label, _, _) in HF_MODELS.items()
                if label == model_name
            )
            try:
                with st.spinner(
                    f"Tracing {hf_label} with torch.fx "
                    "(weights download once, then cached)…"
                ):
                    original_graph, hf_info = _load_hf_graph(hf_key)
            except Exception as exc:
                st.error(
                    f"Real-model import failed: {exc} — falling back to "
                    "the demo transformer."
                )
                original_graph = create_demo_transformer_graph(
                    num_blocks=2
                )
        else:
            num_blocks = 4 if "4 blocks" in model_name else 2
            original_graph = create_demo_transformer_graph(
                num_blocks=num_blocks
            )
        if not real_available:
            st.caption(
                "💡 Install `pip install -r requirements-real.txt` to "
                "compile real GPT-2-family graphs downloaded from "
                "HuggingFace."
            )
        st.caption(
            f"IR: {original_graph.node_count()} ops · "
            f"{original_graph.edge_count()} deps"
        )


with _run_col:
    run_clicked = st.button(
        "Run", type="primary"
    )

st.markdown('</div>', unsafe_allow_html=True)

# Content area wrapper
st.markdown('<div class="content-wrap">', unsafe_allow_html=True)

# ------------------------------------------------------------------ runner
if run_clicked:
    flags = {
        "normalize": normalize,
        "canonicalize": canonicalize,
        "merge": operator_merge,
        "schedule": dags,
        "partition": partition_mode != "Off",
    }
    rl_info = None
    if rl_search:
        from search.rl import search_pass_order

        with st.spinner("RL pass-order search…"):
            recommended, rl_info = search_pass_order(
                original_graph, episodes=40, num_partitions=num_partitions
            )
        if recommended:
            for key, value in recommended.items():
                flags[key] = flags[key] or value

    gnn_score = None
    try:
        from search.gnn import load_or_train

        gnn_score = load_or_train().score_graph(original_graph)
    except Exception:
        gnn_score = None

    training_outcome = None
    if train_on_dataset:
        from training.train import run_training

        with st.spinner("Training on MNIST (cached after the first run)…"):
            try:
                training_outcome = run_training()
            except Exception as exc:
                st.error(f"Training failed: {exc}")

    with st.spinner("Compiling and evaluating…"):
        optimized_graph, infos, compile_time = run_pipeline(
            original_graph,
            hypergraph=partition_mode == "Hypergraph",
            num_partitions=num_partitions,
            neuro_symbolic=neuro_symbolic,
            **flags,
        )
        metrics = compute_metrics(
            original_graph, optimized_graph, infos, compile_time
        )
        st.session_state["result"] = {
            "metrics": metrics,
            "infos": infos,
            "optimized": optimized_graph,
            "rl_info": rl_info,
            "gnn_score": gnn_score,
            "training": training_outcome,
            "hf_info": hf_info,
        }

result = st.session_state.get("result")

if not result:
    st.markdown(
        """
<div class="linear-hero">
<div class="hero-eyebrow"><span class="eyebrow-dot"></span>INTERACTIVE COMPILER PLAYGROUND</div>
<h1>Compiler Optimization for <span class="grad-text">LLM Computation Graphs</span></h1>
<p>Beyond XLA: semantic fusion, attention canonicalization, and learned scheduling for transformer inference graphs — running live, entirely in NumPy.</p>
<div class="cta-hint">🎛️ Configure the pipeline with the toolbar above ▸ then press <strong>Run</strong> to compile and benchmark</div>
</div>

<div class="flow-wrap">
<div class="flow-title">The pipeline you're about to run</div>
<div class="flow">
<div class="flow-step"><span class="fnum">1</span><div><strong>Normalize</strong><small>fold · CSE · DCE</small></div></div>
<div class="flow-arrow">→</div>
<div class="flow-step"><span class="fnum">2</span><div><strong>Canonicalize</strong><small>attention fusion</small></div></div>
<div class="flow-arrow">→</div>
<div class="flow-step"><span class="fnum">3</span><div><strong>Merge</strong><small>GEMM · LinearGELU</small></div></div>
<div class="flow-arrow">→</div>
<div class="flow-step"><span class="fnum">4</span><div><strong>Schedule</strong><small>DAGS levels</small></div></div>
<div class="flow-arrow">→</div>
<div class="flow-step flow-accent"><span class="fnum">5</span><div><strong>Partition</strong><small>multi-kernel</small></div></div>
</div>
</div>

<div class="linear-card-grid">
<div class="linear-card">
<div class="lc-head"><span class="lc-icon lc-red">⚠️</span><h2>The Problem</h2></div>
<p>Modern ML compilers like Google's XLA are excellent at low-level fusion — they merge slices, concatenations, and arithmetic ops into fast kernels. But research (He, 2023, <em>PLOS ONE</em>) shows XLA treats a transformer's attention mechanism as just another set of matmuls and softmaxes, missing the fact that Q/K/V projection → scaled dot-product → softmax → weighted sum is a <strong>single semantic unit</strong> that could be fused, scheduled, and partitioned as one. XLA also struggles on high-dimensional GPU kernels with many inputs/outputs, where its float4/double2 vectorization rarely triggers.</p>
<p>This project asks: <strong>what if the compiler understood transformer structure directly?</strong></p>
</div>

<div class="linear-card">
<div class="lc-head"><span class="lc-icon lc-blue">⚙️</span><h2>What Is an Optimization Pass?</h2></div>
<p>A compiler doesn't rewrite your program all at once. It runs a series of small, focused transformations called <strong>passes</strong> — each one looks at the computation graph, finds a specific pattern, and rewrites it into something faster or smaller, while preserving the exact same output. Passes run one after another, like an assembly line.</p>
<p>This dashboard lets you configure and run that pipeline yourself. Every control in the toolbar maps to a real technique below.</p>
</div>
</div>

<div class="linear-card-grid">
<div class="linear-card">
<div class="lc-head"><span class="lc-icon lc-purple">🧬</span><h2>Optimization Passes</h2></div>
<div class="pass-list">
<div class="pass-item"><span class="pass-num">01</span><div><strong>Graph Normalization</strong><br>Cleans up the graph before real optimization starts: canonicalizes node names, removes no-op identity operations, folds constant arithmetic, eliminates duplicate subexpressions (CSE), and deletes unreachable dead nodes.<br><em>42 ops → 38 ops</em></div></div>
<div class="pass-item"><span class="pass-num">02</span><div><strong>Attention Canonicalization</strong><br>Detects the 7-op Q/K/V→score→scale→softmax→AV pattern and collapses it into a single FusedAttention node — the same trick FlashAttention uses, avoiding writing the huge intermediate score matrix to memory.<br><em>38 ops → 26 ops</em></div></div>
<div class="pass-item"><span class="pass-num">03</span><div><strong>Semantic Operator Merging</strong><br>Fuses adjacent op pairs that almost always run together: MatMul+Add→GEMM, GEMM+GELU→LinearGELU, Add+LayerNorm→FusedAddLayerNorm.<br><em>26 ops → 14 ops</em></div></div>
<div class="pass-item"><span class="pass-num">04</span><div><strong>DAGS Scheduling</strong><br>Dependency-Aware Graph Scheduling. Computes each node's dependency depth and position on the critical path, then orders execution so independent work runs in parallel without stalling.</div></div>
</div>
</div>
<div class="linear-card">
<div class="lc-head"><span class="lc-icon lc-green">🧠</span><h2>Partitioning &amp; Search</h2></div>
<p><strong>Partitioning</strong><br>Splits the optimized graph into balanced partitions to simulate multi-device/multi-kernel execution. Hypergraph-based partitioning is used under the hood where relevant.</p>
<p><strong>Neuro-Symbolic Rewrite Search</strong><br>Combines a small GNN's learned cost predictions with hand-written rewrite rules.</p>
<p><strong>RL Pass-Order Search</strong><br>Uses tabular Q-learning to search over different pass sequences, converging on a near-optimal ordering.</p>
<div class="lc-head" style="margin-top:1.6rem;"><span class="lc-icon lc-amber">📦</span><h2>Dataset &amp; Models</h2></div>
<p><strong>Train MLP on MNIST</strong><br>Trains a small MLP, exports it into the IR, then runs it through the pipeline.</p>
<p><strong>Demo Transformers</strong><br>2 blocks (42 ops) or 4 blocks to see how techniques scale.</p>
</div>
</div>
""", unsafe_allow_html=True
    )
    st.stop()

metrics = result["metrics"]
infos = result["infos"]
optimized_graph = result["optimized"]
counts = metrics["counts"]
modeled = metrics["modeled"]
attention = metrics["attention_subgraphs"]
acc = metrics["accuracy"]
fusions = infos.get("semantic_merging", {}).get("fusions", 0)
speedup = metrics["speedup"]
grr = metrics["graph_reduction_ratio"]
omr = metrics["operator_merge_ratio"]
acr = metrics["attention_canonicalization_rate"]
klr = metrics["kernel_launch_reduction"]
lat0 = metrics["latency_ms"]["original"]
lat1 = metrics["latency_ms"]["optimized"]
thr0 = metrics["throughput"]["original"]
thr1 = metrics["throughput"]["optimized"]
mem0 = metrics["peak_memory_mb"]["original"]
mem1 = metrics["peak_memory_mb"]["optimized"]
compile_ms = metrics["compilation_time_s"] * 1e3

# ------------------------------------------------------------- hero banner
st.markdown(
    f"""
<div class="hero">
  <div>
    <div class="hero-kicker">Compilation result</div>
    <div class="hero-big">{speedup:.2f}×</div>
    <div class="hero-sub">measured speedup · modeled GPU
    {modeled['speedup']:.2f}×</div>
  </div>
  <div class="hero-chips">
    <span class="hchip">✓ {acc['preservation_pct']:.1f}% accuracy</span>
    <span class="hchip">−{klr * 100:.1f}% kernel launches</span>
    <span class="hchip">−{grr * 100:.1f}% graph size</span>
    <span class="hchip">{compile_ms:.1f} ms compile</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)
st.write("")

# ------------------------------------------------------- reduction & fusion
st.markdown('<div class="section-title">🎯 Graph Reduction &amp; Fusion '
            "(structural)</div>", unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(_card(
        "Graph Reduction (GRR)",
        f"{grr * 100:.1f}<small> %</small>",
        sub=f"{counts['nodes_original']} → {counts['nodes_optimized']} ops",
        chip=f"−{counts['nodes_original'] - counts['nodes_optimized']} nodes",
        chip_kind="good",
        bar=grr * 100,
        icon="📉",
    ), unsafe_allow_html=True)
with c2:
    st.markdown(_card(
        "Operator Merge (OMR)",
        f"{omr * 100:.1f}<small> %</small>",
        sub=f"{fusions} kernels fused from operator pairs",
        chip="bias folding on",
        chip_kind="flat",
        bar=omr * 100,
        icon="🧩",
    ), unsafe_allow_html=True)
with c3:
    if attention["detected"]:
        acr_sub = (f"{attention['canonicalized']} of "
                   f"{attention['detected']} attention subgraphs")
        acr_chip = "SDPA / FlashAttention-style"
    else:
        acr_sub = "no attention pattern in graph"
        acr_chip = "n/a"
    st.markdown(_card(
        "Attention Canonicalization (ACR)",
        f"{acr * 100:.0f}<small> %</small>",
        sub=acr_sub,
        chip=acr_chip,
        chip_kind="flat",
        bar=acr * 100,
        icon="⚡",
    ), unsafe_allow_html=True)
with c4:
    st.markdown(_card(
        "Kernel Launch Reduction",
        f"{klr * 100:.1f}<small> %</small>",
        sub=f"{counts['kernel_launches_original']} → "
            f"{counts['kernel_launches_optimized']} launches",
        chip="one launch per fused kernel",
        chip_kind="flat",
        bar=klr * 100,
        icon="🚀",
    ), unsafe_allow_html=True)

# -------------------------------------------------- runtime (measured)
st.markdown('<div class="section-title">⚡ Runtime Performance '
            "(measured on the NumPy reference executor)</div>",
            unsafe_allow_html=True)
r1, r2, r3, r4 = st.columns(4)
with r1:
    st.markdown(_card(
        "Inference Latency",
        f"{lat1:.2f}<small> ms</small>",
        sub=f"was {lat0:.2f} ms · batch "
            f"{metrics['throughput']['batch']}",
        chip=f"{lat1 - lat0:+.2f} ms",
        chip_kind="good" if lat1 <= lat0 else "bad",
        icon="⏱️",
    ), unsafe_allow_html=True)
with r2:
    st.markdown(_card(
        "Throughput",
        f"{thr1:.0f}<small> samples/s</small>",
        sub="steady-state, best-of-30 runs",
        chip=f"{thr1 - thr0:+.0f} samples/s",
        chip_kind="good" if thr1 >= thr0 else "bad",
        icon="🔁",
    ), unsafe_allow_html=True)
with r3:
    st.markdown(_card(
        "Speedup",
        f"{speedup:.2f}<small>×</small>",
        sub="original vs optimized graph",
        chip=f"modeled GPU {modeled['speedup']:.2f}×",
        chip_kind="flat",
        icon="📈",
    ), unsafe_allow_html=True)
with r4:
    st.markdown(_card(
        "Peak Memory (live tensors)",
        f"{mem1:.2f}<small> MB</small>",
        sub=f"was {mem0:.2f} MB during execution",
        chip=f"{mem1 - mem0:+.2f} MB",
        chip_kind="good" if mem1 <= mem0 else "bad",
        icon="🧠",
    ), unsafe_allow_html=True)

# ------------------------------------------------- accuracy & compilation
st.markdown('<div class="section-title">✅ Accuracy &amp; Compilation</div>',
            unsafe_allow_html=True)
a1, a2, a3, a4 = st.columns(4)
with a1:
    st.markdown(_card(
        "Accuracy Preservation",
        f"{acc['preservation_pct']:.1f}<small> %</small>",
        sub="original vs optimized outputs",
        chip=("outputs match" if acc["preserved"]
              else f"max |diff| {acc['max_abs_diff']:.1e}"),
        chip_kind="good" if acc["preserved"] else "bad",
        icon="🎯",
    ), unsafe_allow_html=True)
with a2:
    st.markdown(_card(
        "Compilation Time",
        f"{compile_ms:.1f}<small> ms</small>",
        sub="pipeline wall time, per pass below",
        chip=f"{len(infos['pipeline']['passes_run'])} passes",
        chip_kind="flat",
        icon="⚙️",
    ), unsafe_allow_html=True)
with a3:
    st.markdown(_card(
        "Modeled GPU Latency",
        f"{modeled['latency_optimized_us'] / 1000:.2f}<small> ms</small>",
        sub=(f"was {modeled['latency_original_us'] / 1000:.2f} ms · "
             "level-parallel DAGS"),
        chip=f"{modeled['speedup']:.2f}× modeled",
        chip_kind="flat",
        icon="🖥️",
    ), unsafe_allow_html=True)
with a4:
    st.markdown(_card(
        "Modeled GPU Memory",
        f"{modeled['memory_optimized_mb']:.0f}<small> MB</small>",
        sub=f"was {modeled['memory_original_mb']:.0f} MB without reuse",
        chip=f"−{modeled['memory_reduction_pct'] * 100:.0f}% buffer reuse",
        chip_kind="good",
        icon="💾",
    ), unsafe_allow_html=True)

# --------------------------------------------------------- summary table
st.markdown('<div class="section-title">📋 Metric Summary</div>',
            unsafe_allow_html=True)
summary = pd.DataFrame(
    [
        ("Graph reduction ratio (GRR)", "—", f"{grr * 100:.1f}%",
         f"{counts['nodes_original']} → {counts['nodes_optimized']} nodes"),
        ("Operator merge ratio (OMR)", "—", f"{omr * 100:.1f}%",
         f"{fusions} fusions"),
        ("Attention canonicalization (ACR)", "—", f"{acr * 100:.0f}%",
         f"{attention['canonicalized']}/{attention['detected']} subgraphs"),
        ("Kernel launch reduction", "—", f"{klr * 100:.1f}%",
         f"{counts['kernel_launches_original']} → "
         f"{counts['kernel_launches_optimized']}"),
        ("Inference latency (ms)", f"{lat0:.3f}", f"{lat1:.3f}",
         f"{speedup:.2f}× faster"),
        ("Throughput (samples/s)", f"{thr0:.0f}", f"{thr1:.0f}",
         f"{(thr1 / thr0 - 1) * 100:+.1f}%"),
        ("Peak memory, live tensors (MB)", f"{mem0:.2f}", f"{mem1:.2f}",
         f"{(1 - mem1 / mem0) * 100:.0f}% lower" if mem0 else "—"),
        ("Modeled GPU latency (ms)",
         f"{modeled['latency_original_us'] / 1000:.2f}",
         f"{modeled['latency_optimized_us'] / 1000:.2f}",
         f"{modeled['speedup']:.2f}× faster"),
        ("Modeled GPU memory (MB)",
         f"{modeled['memory_original_mb']:.0f}",
         f"{modeled['memory_optimized_mb']:.0f}",
         f"−{modeled['memory_reduction_pct'] * 100:.0f}%"),
        ("Accuracy preservation", "—", f"{acc['preservation_pct']:.2f}%",
         f"max |diff| {acc['max_abs_diff']:.1e}"),
        ("Compilation time", "—", f"{compile_ms:.1f} ms",
         f"{len(infos['pipeline']['passes_run'])} passes"),
    ],
    columns=["Metric", "Original", "Optimized", "Delta"],
)
st.dataframe(summary, use_container_width=True, hide_index=True)

# ------------------------------------------------------------- operator mix
st.markdown('<div class="section-title">📊 Operator Mix</div>',
            unsafe_allow_html=True)
hist = pd.DataFrame(
    {
        "Original": pd.Series(original_graph.op_type_histogram()),
        "Optimized": pd.Series(optimized_graph.op_type_histogram()),
    }
).fillna(0).astype(int)
st.bar_chart(hist, height=250)

# ------------------------------------------------------------------- graphs
st.markdown('<div class="section-title">🔀 Computation Graph</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="legend">'
    '<span class="dot" style="color:#9ec5ff">●</span> MatMul &nbsp;'
    '<span class="dot" style="color:#7fa8ff">●</span> GEMM &nbsp;'
    '<span class="dot" style="color:#ffd27f">●</span> FusedAttention &nbsp;'
    '<span class="dot" style="color:#a2d2ff">●</span> FusedAddLayerNorm &nbsp;'
    '<span class="dot" style="color:#ffb3c6">●</span> FusedScaleSoftmax &nbsp;'
    '<span class="dot" style="color:#e0b0ff">●</span> LinearGELU &nbsp;'
    '<span class="dot" style="color:#b5ead7">●</span> elementwise &nbsp;'
    '<span class="dot" style="color:#d9d9d9">●</span> I/O &nbsp;—&nbsp; '
    '<b>bold border = fused kernel</b></div>',
    unsafe_allow_html=True,
)
g1, g2 = st.columns(2)
with g1:
    st.subheader("Original")
    st.graphviz_chart(
        graph_to_dot(original_graph, show_schedule=False,
                     show_partition=False),
        use_container_width=True,
    )
with g2:
    st.subheader("Optimized")
    st.graphviz_chart(
        graph_to_dot(optimized_graph), use_container_width=True
    )

# --------------------------------------------------------------------- tabs
tab_details, tab_sched, tab_part, tab_learn, tab_train, tab_report = st.tabs(
    ["🧾 Pass Details", "🗓️ DAGS Schedule", "🧱 Partitions",
     "🧠 Learned Search", "🏋️ Training (MNIST)", "📋 Full Report"]
)

with tab_details:
    hf_result_info = result.get("hf_info")
    if hf_result_info:
        st.caption(
            f"📦 Imported from **{hf_result_info['model']}** via torch.fx · "
            f"{hf_result_info.get('parameters', 0) / 1e6:.1f}M real "
            f"parameters · {hf_result_info.get('torch.fx nodes')} fx nodes "
            f"→ {hf_result_info.get('ops_emitted')} IR ops"
        )
    for name, info in infos.items():
        st.markdown(f"**{name}**")
        st.json(_jsonable(info), expanded=False)

with tab_sched:
    schedule_info = optimized_graph.meta.get("schedule")
    if schedule_info:
        rows = [
            {
                "Order": data.get("schedule_order"),
                "Level": data.get("schedule_level"),
                "Operator": node,
                "Type": data.get("op_type"),
                "Shape": data.get("shape"),
            }
            for node, data in optimized_graph.nodes(data=True)
        ]
        rows.sort(key=lambda row: row["Order"] or 0)
        st.dataframe(pd.DataFrame(rows), use_container_width=True,
                     hide_index=True)
        st.caption(
            f"{schedule_info['levels']} levels · critical path "
            f"{len(schedule_info['critical_path'])} ops · modeled makespan "
            f"{schedule_info['makespan_parallel_us'] / 1000:.2f} ms "
            f"(sequential "
            f"{schedule_info['sequential_compute_us'] / 1000:.2f} ms)"
        )
    else:
        st.caption("DAGS was not enabled.")

with tab_part:
    partition_info = optimized_graph.meta.get("partition")
    if partition_info:
        rows = [
            {
                "Partition": data["partition"],
                "Operator": node,
                "Type": data.get("op_type"),
            }
            for node, data in optimized_graph.nodes(data=True)
            if "partition" in data
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True,
                     hide_index=True)
        st.caption(
            f"{partition_info['partitions']} partitions · sizes "
            f"{partition_info['partition_sizes']} · balance "
            f"{partition_info['balance_ratio']:.2f} · "
            + (
                f"cut edges {partition_info['cut_edges']}"
                if "cut_edges" in partition_info
                else f"cut nets {partition_info['cut_hyperedges']} · "
                     f"comm {partition_info['communication_volume']}"
            )
        )
    else:
        st.caption("Partitioning was not enabled.")

with tab_learn:
    gnn_score = result.get("gnn_score")
    if gnn_score is not None:
        st.markdown(_card(
            "GNN Predicted Speedup",
            f"{gnn_score:.2f}<small>×</small>",
            sub="SGC-style GNN surrogate · ridge head trained on a "
                "synthetic DAG corpus",
            chip=f"actual {speedup:.2f}×",
            chip_kind="flat",
            icon="🧠",
        ), unsafe_allow_html=True)
    else:
        st.caption("GNN scorer unavailable.")
    rl_info = result.get("rl_info")
    if rl_info:
        st.caption(
            "RL recommended sequence: "
            + (", ".join(rl_info["best_sequence"]) or "(none)")
            + f" · reward {rl_info['best_reward']:.3f} · "
            f"{rl_info['episodes']} episodes · "
            f"{rl_info['q_states']} Q-states explored"
        )
    ns_info = infos.get("neuro_symbolic_search")
    if ns_info and "iterations" in ns_info:
        st.caption(
            f"Neuro-symbolic search: {ns_info['iterations']} rewrites "
            f"applied · hybrid score {ns_info['initial_score']:.3f} → "
            f"{ns_info['final_score']:.3f} · {ns_info['nodes_after']} nodes"
        )
        for item in ns_info.get("applied", [])[:8]:
            st.caption(f"• {item['rule']} (score {item['score']:.3f})")
    if not rl_info and not (ns_info and "iterations" in ns_info):
        st.caption(
            "Enable the learned-search toggles in the sidebar to see the "
            "GNN prediction, the RL pass-order recommendation and the "
            "neuro-symbolic rewrite log."
        )

with tab_train:
    training_outcome = result.get("training")
    if training_outcome:
        data = training_outcome["dataset"]
        tm = training_outcome["metrics"]
        st.markdown(_card(
            "MNIST Test Accuracy",
            f"{training_outcome['test_acc'] * 100:.2f}<small> %</small>",
            sub=f"dataset: {data['name']} · MLP 784-256-64-10 · "
                "NumPy manual backprop + Adam",
            chip=f"{len(data['x_train'])} train / {len(data['x_test'])} test",
            chip_kind="good",
            icon="🏋️",
        ), unsafe_allow_html=True)
        st.caption(
            "The trained weights were exported into the IR "
            f"({training_outcome['graph'].node_count()} ops), compiled to "
            f"{training_outcome['optimized'].node_count()} ops "
            f"({tm['modeled']['speedup']:.2f}× modeled, "
            f"{tm['speedup']:.2f}× measured) and verified: "
            f"{training_outcome['pred_match_pct']:.1f}% identical "
            "predictions vs the trained model (bit-exact)."
        )
        history_df = pd.DataFrame(training_outcome["history"])
        st.subheader("Cross-entropy loss")
        st.line_chart(history_df.set_index("epoch")[["train_loss",
                                                     "val_loss"]],
                      height=240)
        st.subheader("Accuracy")
        st.line_chart(history_df.set_index("epoch")[["train_acc",
                                                     "val_acc"]],
                      height=240)
        st.subheader("Confusion matrix (test split)")
        cm_df = pd.DataFrame(training_outcome["confusion"])
        st.dataframe(cm_df, use_container_width=True)
    else:
        st.caption(
            "Enable **Train MLP on MNIST + compile it** in the sidebar, "
            "then click Run Optimization — the first run trains the model "
            "(cached afterwards) and this tab shows the training curves, "
            "the confusion matrix and the compiled-model verification."
        )

    with st.expander("🎓 How the training works — step by step"):
        st.markdown(
            "The MNIST classifier is trained with **pure NumPy and a "
            "hand-written backward pass** — no PyTorch. Six steps per "
            "mini-batch:"
        )
        st.markdown("**1 — Forward pass** (batch of 64 flattened images):")
        st.latex(
            r"z^{(l)} = a^{(l-1)} W^{(l)} + b^{(l)}, \qquad"
            r" a^{(l)} = \mathrm{ReLU}(z^{(l)}), \qquad"
            r" \hat{y} = \mathrm{softmax}(z^{(3)})"
        )
        st.markdown("**2 — Cross-entropy loss** (how wrong the predictions "
                    "are):")
        st.latex(
            r"\mathcal{L} = -\frac{1}{N}\sum_{n=1}^{N}"
            r" \sum_{c=1}^{C} y_{n,c}\, \log \hat{y}_{n,c}"
        )
        st.markdown("**3 — Backpropagation** (chain rule, layer by layer; "
                    "the softmax+CE output error simplifies to "
                    "ŷ − y):")
        st.latex(
            r"\delta^{(3)} = \hat{y} - y, \qquad"
            r" \delta^{(l)} = \left(\delta^{(l+1)} W^{(l+1)\top}\right)"
            r" \odot \mathbb{1}\!\left[z^{(l)} > 0\right]"
        )
        st.latex(
            r"\frac{\partial \mathcal{L}}{\partial W^{(l)}} ="
            r" \delta^{(l+1)\top} a^{(l)}, \qquad"
            r" \frac{\partial \mathcal{L}}{\partial b^{(l)}} ="
            r" \sum_n \delta^{(l+1)}_n"
        )
        st.markdown("**4 — Adam update** (adaptive learning rate per "
                    "parameter):")
        st.latex(
            r"m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t, \quad"
            r" v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2"
        )
        st.latex(
            r"\theta \leftarrow \theta - \alpha\,"
            r" \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)"
        )
        st.markdown(
            "**5 — Loop**: 6 epochs × mini-batches of 64 (12 000 train / "
            "2 000 test MNIST images) — the curves above are the loss and "
            "accuracy recorded at every epoch."
        )
        st.markdown(
            "**6 — Export & compile**: the *learned* weights are written "
            "into the IR as `MatMul` weights + `Constant` biases, the "
            "compiler fuses them (MatMul+Add→GEMM, +ReLU→LinearRelu), and "
            "the compiled graph is verified **bit-exact** against the "
            "trained model."
        )
        st.caption(
            "Note: the LLMs in this project are **not trained** — they use "
            "pretrained weights and we optimize their *inference* graph, "
            "exactly like production compilers (XLA, TensorRT) do. The "
            "GNN/RL models in the Learned Search tab are separate: they "
            "learn *compiler decisions*, not model weights."
        )

with tab_report:
    st.code(format_report(metrics), language="text")

st.divider()
st.caption(
    "Measured numbers come from the NumPy reference executor "
    f"(batch {metrics['throughput']['batch']}, seq 32, d_model 64, "
    "best-of-30 runs; no GPU on this machine). Modeled numbers come from "
    "the GPU-style cost model with level-parallel scheduling and "
    "buffer-reuse planning. Accuracy is the element-wise output "
    "difference between the original and the fully fused graph."
)
