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
    .block-container { padding-top: 1.4rem; padding-bottom: 2.4rem; }
    #MainMenu, footer { visibility: hidden; }
    .app-title { font-size: 2rem; font-weight: 800; color: #101828;
                 letter-spacing: -0.02em; margin: 0; }
    .app-sub { color: #667085; font-size: .93rem; margin-top: 4px; }
    .section-title {
        font-size: 1.02rem; font-weight: 700; color: #101828;
        border-left: 4px solid #4f46e5; padding-left: 10px;
        margin: 10px 0 2px 0;
    }
    .hero {
        background: linear-gradient(120deg, #4f46e5 0%, #7c3aed 55%, #a855f7 100%);
        border-radius: 20px; padding: 20px 26px; color: #ffffff;
        display: flex; justify-content: space-between; align-items: center;
        flex-wrap: wrap; gap: 14px;
        box-shadow: 0 10px 30px rgba(79, 70, 229, .32);
    }
    .hero-kicker { font-size: .74rem; font-weight: 700; letter-spacing: .14em;
                   text-transform: uppercase; opacity: .85; }
    .hero-big { font-size: 2.7rem; font-weight: 800; line-height: 1.05; }
    .hero-sub { font-size: .9rem; opacity: .92; margin-top: 2px; }
    .hero-chips { display: flex; gap: 8px; flex-wrap: wrap;
                  justify-content: flex-end; max-width: 460px; }
    .hchip { background: rgba(255,255,255,.16);
             border: 1px solid rgba(255,255,255,.35);
             padding: 6px 12px; border-radius: 999px;
             font-size: .8rem; font-weight: 600; }
    .mcard {
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #e5eaf1; border-radius: 16px;
        padding: 14px 16px 12px 16px; height: 100%;
        box-shadow: 0 1px 2px rgba(16,24,40,.05),
                    0 1px 3px rgba(16,24,40,.08);
    }
    .mtitle { font-size: .72rem; font-weight: 700; color: #667085;
              letter-spacing: .05em; text-transform: uppercase; }
    .mvalue { font-size: 1.7rem; font-weight: 800; color: #101828;
              line-height: 1.15; margin: 4px 0 8px 0; }
    .mvalue small { font-size: .92rem; font-weight: 600; color: #98a2b3; }
    .msub { font-size: .78rem; color: #667085; display: flex; gap: 6px;
            align-items: center; flex-wrap: wrap; min-height: 22px; }
    .chip { padding: 2px 9px; border-radius: 999px; font-weight: 700;
            font-size: .73rem; white-space: nowrap; }
    .chip-good { background: #ecfdf3; color: #067647;
                 border: 1px solid #abefc6; }
    .chip-bad  { background: #fef3f2; color: #b42318;
                 border: 1px solid #fecdc9; }
    .chip-flat { background: #eff4ff; color: #3538cd;
                 border: 1px solid #c7d7fe; }
    .bar { height: 6px; background: #eef2f6; border-radius: 999px;
           overflow: hidden; margin: 2px 0 8px 0; }
    .bar-fill { height: 100%; border-radius: 999px;
                background: linear-gradient(90deg, #6366f1, #a855f7); }
    .legend { font-size: .8rem; color: #475467; line-height: 1.7; }
    .legend .dot { font-size: .95rem; }
    div[data-testid="stDataFrame"] { border-radius: 12px; }
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
        return value.tolist()
    return value


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


st.markdown('<div class="app-title">⚡ LLM Compiler Optimizer</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="app-sub">A graph-based compiler optimization framework '
    "for LLMs — normalization, attention canonicalization, semantic "
    "fusion, DAGS scheduling and partitioning, evaluated with the full "
    "output-metric suite.</div>",
    unsafe_allow_html=True,
)
st.divider()

# ------------------------------------------------------------------ sidebar
st.sidebar.header("🧩 Optimization Passes")

normalize = st.sidebar.checkbox(
    "Graph Normalization",
    value=True,
    help="Canonicalize ops, fold constants, CSE, DCE, infer shapes.",
)
canonicalize = st.sidebar.checkbox(
    "Attention Graph Canonicalization",
    value=True,
    help="Fuse Q/K/V → QKᵀ → Scale → Softmax → AV into FusedAttention.",
)
operator_merge = st.sidebar.checkbox(
    "Semantic Operator Merging",
    value=True,
    help="MatMul+Add→GEMM, +GELU→LinearGELU, Add+LayerNorm fusion.",
)
dags = st.sidebar.checkbox(
    "Dependency-Aware Scheduling (DAGS)",
    value=True,
    help="Levels, critical path and a deterministic list schedule.",
)
partition_mode = st.sidebar.radio(
    "Partitioning",
    ["Off", "Graph", "Hypergraph"],
    index=0,
    horizontal=True,
)
num_partitions = st.sidebar.slider(
    "Number of Partitions", 2, 4, 2,
    disabled=partition_mode == "Off",
)

st.sidebar.divider()
st.sidebar.subheader("🧠 Learned Search")
neuro_symbolic = st.sidebar.checkbox(
    "Neuro-Symbolic Rewrite Search",
    value=False,
    help="GNN-guided symbolic rewrite rules search for extra fusion on "
         "top of the selected passes.",
)
rl_search = st.sidebar.checkbox(
    "RL Pass-Order Search (Q-learning)",
    value=False,
    help="Search the pass subset with tabular Q-learning and compile "
         "with the recommended combination.",
)

st.sidebar.divider()
st.sidebar.subheader("🧠 Input Model")
model_name = st.sidebar.selectbox(
    "Model",
    ["Demo Transformer (2 blocks)", "Demo Transformer (4 blocks)"],
)
num_blocks = 4 if "4 blocks" in model_name else 2

original_graph = create_demo_transformer_graph(num_blocks=num_blocks)
st.sidebar.caption(
    f"IR: {original_graph.node_count()} ops · "
    f"{original_graph.edge_count()} dependencies"
)

run_clicked = st.sidebar.button(
    "🚀 Run Optimization", type="primary", use_container_width=True
)

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
        }

result = st.session_state.get("result")

if not result:
    st.info(
        "Select optimization passes in the sidebar and click "
        "**🚀 Run Optimization** to compile the model and produce the "
        "full output-metric report."
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
tab_details, tab_sched, tab_part, tab_learn, tab_report = st.tabs(
    ["🧾 Pass Details", "🗓️ DAGS Schedule", "🧱 Partitions",
     "🧠 Learned Search", "📋 Full Report"]
)

with tab_details:
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
