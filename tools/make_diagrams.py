"""Generate the architecture / block / pass diagrams used by the GUI and report.

Run:  python tools/make_diagrams.py
Writes PNG (+SVG) files to docs/diagrams/ and frontend/public/diagrams/.
Graph figures are drawn from the *real* graphs produced by the compiler, so
they cannot drift from the code.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.operations import attention_demo, transpose_demo  # noqa: E402
from models.transformer import create_demo_transformer_graph  # noqa: E402
from passes.canonicalize import canonicalize_attention  # noqa: E402
from passes.normalize import normalize_graph  # noqa: E402
from passes.partition import partition_graph, partition_hypergraph  # noqa: E402
from pipeline import run_pipeline  # noqa: E402

OUT = [ROOT / "docs" / "diagrams", ROOT / "frontend" / "public" / "diagrams"]

INK, MUTED = "#1f2433", "#5b6478"
C = {"input": "#e8eefc", "ir": "#ede7fb", "pass": "#dff3ee", "eval": "#fdf0d9",
     "out": "#fbe5ea", "ui": "#e6f0fa", "fused": "#f9d9a8", "plain": "#ffffff"}
EDGE = "#3b3f58"
PART = ["#cfe3ff", "#ffd9c9", "#d8f2d0", "#f1d6f5", "#fff1b8", "#d4f0f0"]


def box(ax, x, y, w, h, text, fc="#fff", fs=9, bold=False, ec=EDGE, lw=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=INK, fontweight="bold" if bold else "normal", linespacing=1.25)


def arrow(ax, p, q, color=EDGE, rad=0.0, lw=1.3, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=11, lw=lw,
                                 color=color, connectionstyle=f"arc3,rad={rad}",
                                 linestyle=ls, shrinkA=0, shrinkB=0))


def canvas(w, h, title=None):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    if title:
        ax.text(w / 2, h - 0.25, title, ha="center", va="top", fontsize=13,
                fontweight="bold", color=INK)
    return fig, ax


def save(fig, name):
    for d in OUT:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / f"{name}.png", dpi=170, bbox_inches="tight", facecolor="white")
        fig.savefig(d / f"{name}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", name)


def layered_positions(graph):
    g = graph.graph
    level = {}
    for n in graph.topological_order():
        level[n] = max((level[p] for p in g.predecessors(n)), default=-1) + 1
    cols = {}
    for n, lv in level.items():
        cols.setdefault(lv, []).append(n)
    pos = {}
    for lv, names in cols.items():
        for i, n in enumerate(sorted(names)):
            pos[n] = (lv, -(i - (len(names) - 1) / 2))
    return pos


def draw_dag(ax, graph, title, color=None, fs=7.5):
    pos = layered_positions(graph)
    g = graph.graph
    xs = [p[0] for p in pos.values()] or [0]
    ys = [p[1] for p in pos.values()] or [0]
    for u, v in g.edges():
        (x1, y1), (x2, y2) = pos[u], pos[v]
        arrow(ax, (x1 + 0.34, y1), (x2 - 0.34, y2), color="#8a90a6", lw=1.1)
    for n, (x, y) in pos.items():
        d = g.nodes[n]
        op = str(d.get("op_type", "?"))
        if color:
            fc = color(n, d)
        else:
            fc = C["fused"] if d.get("fused") or op.startswith(("Fused", "Linear")) else C["plain"]
        ax.add_patch(FancyBboxPatch((x - 0.34, y - 0.2), 0.68, 0.4,
                                    boxstyle="round,pad=0.01,rounding_size=0.06", fc=fc, ec=EDGE, lw=1))
        ax.text(x, y, op, ha="center", va="center", fontsize=fs, color=INK)
    ax.set_xlim(min(xs) - 0.6, max(xs) + 0.6)
    ax.set_ylim(min(ys) - 0.6, max(ys) + 0.6)
    ax.set_title(f"{title}\n({graph.node_count()} nodes)", fontsize=10.5, color=INK)
    ax.axis("off")


def fig_system_architecture():
    fig, ax = canvas(13.5, 7.4, "System architecture: LLM computation-graph compiler")
    cols = [
        (0.25, 2.55, "1  Model sources", C["input"], [
            "Architecture builders\nPost-LN · Pre-LN · ReLU\nLLaMA-style · MLP",
            "Operation lab\ntranspose · scale chain\nMLP block · attention",
            "Real LLM import\nGPT-2 via torch.fx\n(HuggingFace)",
            "PyTorch attention\nmodule (FX trace)"]),
        (3.15, 2.2, "2  Graph IR", C["ir"], [
            "ComputationGraph\nDAG: ops = nodes\ntensor deps = edges",
            "Node attributes\nop_type · shape · weight\nfused_from · level"]),
        (5.7, 2.9, "3  Compiler passes", C["pass"], [
            "Graph normalization\nCSE · const-fold · transpose algebra",
            "Attention canonicalization\nQ,K,V·QKᵀ·scale·softmax·AV",
            "Semantic operator merging\nMatMul+Add+GELU → LinearGELU",
            "DAGS scheduling\nlevels + critical path",
            "Partitioning\ngraph cut / hypergraph"]),
        (8.95, 2.3, "4  Evaluation", C["eval"], [
            "NumPy reference\nexecutor (measured)",
            "GPU cost model\n(modeled latency/memory)",
            "Equivalence check\noriginal vs optimized",
            "JAX / OpenXLA\n(CPU) baseline"]),
        (11.6, 1.7, "5  Interface", C["ui"], [
            "REST API\nPython",
            "React GUI\n(Vite)",
            "Excel report\n(openpyxl)"]),
    ]
    for x, w, title, fc, items in cols:
        ax.text(x + w / 2, 6.65, title, ha="center", fontsize=10.5, fontweight="bold", color=INK)
        n = len(items)
        h = min(1.12, 5.6 / n - 0.18)
        for i, t in enumerate(items):
            y = 6.35 - (i + 1) * (5.6 / n) + (5.6 / n - h) / 2
            box(ax, x, y, w, h, t, fc=fc, fs=8)
    for a, b in [(2.8, 3.15), (5.35, 5.7), (8.6, 8.95), (11.25, 11.6)]:
        arrow(ax, (a, 3.6), (b, 3.6), lw=2)
    ax.text(6.75, 0.55, "Ten output metrics: latency · throughput · speedup · peak GPU memory · GRR · "
            "accuracy preservation · OMR · ACR · kernel-launch reduction · compilation time",
            ha="center", fontsize=9, color=MUTED, style="italic")
    save(fig, "fig_system_architecture")


def fig_block_diagram():
    fig, ax = canvas(13.5, 5.6, "Block diagram: compiler pass pipeline")
    steps = [
        ("Input graph\nG₀ (DAG)", C["input"]),
        ("Normalize\nidentity · CSE\nconst-fold\ntranspose algebra", C["pass"]),
        ("Attention\ncanonicalize\n→ FusedAttention", C["pass"]),
        ("Semantic merge\nGEMM · LinearGELU\nFusedAddLayerNorm", C["pass"]),
        ("DAGS\nlevels +\ncritical path", C["pass"]),
        ("Partition\nk-way cut\n(hypergraph)", C["pass"]),
        ("Optimized\ngraph G*", C["out"]),
    ]
    w, gap, x = 1.55, 0.35, 0.3
    centers = []
    for t, fc in steps:
        box(ax, x, 2.55, w, 1.7, t, fc=fc, fs=8.5)
        centers.append(x + w / 2)
        x += w + gap
    for i in range(len(steps) - 1):
        arrow(ax, (centers[i] + w / 2, 3.4), (centers[i + 1] - w / 2, 3.4), lw=1.8)
    box(ax, 1.4, 0.55, 3.2, 1.0, "Reference executor\nruns G₀ and G*", C["eval"], fs=9)
    box(ax, 5.2, 0.55, 3.2, 1.0, "Equivalence check\nmax |y₀ − y*| ≤ tolerance", C["eval"], fs=9)
    box(ax, 9.0, 0.55, 3.8, 1.0, "Metrics\n10 outputs → GUI / Excel", C["ui"], fs=9)
    arrow(ax, (centers[0], 2.55), (2.4, 1.55), color=MUTED, ls="--")
    arrow(ax, (centers[-1], 2.55), (4.2, 1.55), color=MUTED, ls="--", rad=-0.1)
    arrow(ax, (4.6, 1.05), (5.2, 1.05))
    arrow(ax, (8.4, 1.05), (9.0, 1.05))
    ax.text(6.75, 4.75, "each pass: graph in → graph out, semantics preserved", ha="center",
            fontsize=9.5, color=MUTED, style="italic")
    save(fig, "fig_block_diagram")


def fig_attention_fusion():
    g = attention_demo()
    opt, _ = canonicalize_attention(g)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [3, 1.3]})
    draw_dag(axes[0], g, "Before: unfused attention head")
    draw_dag(axes[1], opt, "After: FusedAttention")
    fig.suptitle("Attention graph canonicalization (real compiler output)", fontsize=12.5,
                 fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, "fig_attention_fusion")


def fig_transpose_algebra():
    g = transpose_demo()
    opt, _ = normalize_graph(g)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), gridspec_kw={"width_ratios": [1.6, 1.1]})
    draw_dag(axes[0], g, "Before: (Qᵀ)ᵀ and Kᵀ materialised",
             color=lambda n, d: "#f6c9c9" if d.get("op_type") == "Transpose" else C["plain"])
    draw_dag(axes[1], opt, "After: Q·Kᵀ as one transposed GEMM",
             color=lambda n, d: C["fused"] if d.get("transpose_rhs") else C["plain"])
    fig.suptitle("Graph normalization: transpose algebra  (Aᵀ)ᵀ = A,  A·Bᵀ → GEMM(transpose_rhs)",
                 fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, "fig_transpose_algebra")


def fig_dags_schedule():
    g = create_demo_transformer_graph(num_blocks=1)
    opt, _, _ = run_pipeline(g, schedule=True)
    sched = opt.meta["schedule"]
    crit = set(sched["critical_path"])
    fig, ax = plt.subplots(figsize=(12, 5))
    gg = opt.graph
    levels = {}
    for n, d in gg.nodes(data=True):
        levels.setdefault(d["schedule_level"], []).append(n)
    pos = {}
    for lv, names in levels.items():
        for i, n in enumerate(sorted(names)):
            pos[n] = (lv, -(i - (len(names) - 1) / 2))
    for u, v in gg.edges():
        on = u in crit and v in crit
        arrow(ax, (pos[u][0] + 0.36, pos[u][1]), (pos[v][0] - 0.36, pos[v][1]),
              color="#d1542f" if on else "#9aa0b5", lw=2.2 if on else 1.0)
    for n, (x, y) in pos.items():
        ax.add_patch(FancyBboxPatch((x - 0.36, y - 0.22), 0.72, 0.44,
                                    boxstyle="round,pad=0.01,rounding_size=0.06",
                                    fc="#ffe3d6" if n in crit else C["plain"], ec=EDGE, lw=1))
        ax.text(x, y, gg.nodes[n]["op_type"], ha="center", va="center", fontsize=7, color=INK)
    top = max(p[1] for p in pos.values())
    for lv in levels:
        ax.text(lv, top + 0.8, f"level {lv}", ha="center", fontsize=8.5, color=MUTED)
    ax.set_xlim(-0.7, max(levels) + 0.7)
    ax.set_ylim(min(p[1] for p in pos.values()) - 0.7, top + 1.2)
    ax.axis("off")
    ax.set_title(f"Dependency-Aware Graph Scheduling (DAGS): {sched['levels']} dependency levels, "
                 f"avg parallelism {sched['avg_parallelism']:.2f}; critical path in orange",
                 fontsize=11, color=INK)
    save(fig, "fig_dags_schedule")


def fig_partitioning():
    g = create_demo_transformer_graph(num_blocks=1)
    fig, axes = plt.subplots(2, 1, figsize=(13, 9))
    for ax, (label, fn) in zip(axes, [("Graph min-cut (k=3)", partition_graph),
                                      ("Hypergraph partition (k=3)", partition_hypergraph)]):
        opt, _, _ = run_pipeline(g, canonicalize=False, merge=False, schedule=False)
        part, info = fn(opt, 3)

        def col(n, d, p=part):
            return PART[int(p.graph.nodes[n].get("partition", 0)) % len(PART)]

        cut = (f"cut hyperedges = {info['cut_hyperedges']}" if "cut_hyperedges" in info
               else f"cut edges = {info['cut_edges']}")
        draw_dag(ax, part, f"{label}: {cut}, sizes = {info['partition_sizes']}", color=col, fs=6)
    fig.suptitle("Graph partitioning: colours = partitions (devices)", fontsize=12.5,
                 fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, "fig_partitioning")


def fig_architectures():
    fig, ax = canvas(13.5, 6.2, "Supported transformer block variants")
    variants = [
        ("Post-LN (Vaswani)", ["Q K V projections", "QKᵀ → scale → softmax → ·V", "Add (residual)", "LayerNorm",
                               "FFN: Linear + GELU + Linear", "Add (residual)", "LayerNorm"]),
        ("Pre-LN (GPT-2 style)", ["LayerNorm", "Q K V projections", "QKᵀ → scale → softmax → ·V", "Add (residual)",
                                  "LayerNorm", "FFN: Linear + GELU + Linear", "Add (residual)"]),
        ("LLaMA-style", ["RMSNorm", "Q K V projections", "QKᵀ → scale → softmax → ·V", "O projection + Add",
                         "RMSNorm", "SwiGLU: silu(gate)·up → down", "Add (residual)"]),
    ]
    w = 3.9
    for k, (name, blocks) in enumerate(variants):
        x = 0.35 + k * (w + 0.35)
        ax.text(x + w / 2, 5.5, name, ha="center", fontsize=11, fontweight="bold", color=INK)
        for i, t in enumerate(blocks):
            y = 4.65 - i * 0.68
            if "Norm" in t:
                fc = C["pass"]
            elif "FFN" in t or "SwiGLU" in t:
                fc = C["eval"]
            elif "QK" in t:
                fc = C["ir"]
            else:
                fc = C["input"]
            box(ax, x, y, w, 0.5, t, fc=fc, fs=8)
            if i < len(blocks) - 1:
                arrow(ax, (x + w / 2, y), (x + w / 2, y - 0.18), lw=1)
    save(fig, "fig_architectures")


if __name__ == "__main__":
    for fn in (fig_system_architecture, fig_block_diagram, fig_attention_fusion,
               fig_transpose_algebra, fig_dags_schedule, fig_partitioning, fig_architectures):
        fn()
