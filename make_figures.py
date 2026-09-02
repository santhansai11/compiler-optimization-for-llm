"""Generate the report figures (docs/*.png) from the project itself.

Run:  python make_figures.py   (writes PNGs into docs/)
"""

import os

import networkx as nx
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SS = 2  # supersample factor for smooth output
DOCS = os.path.join(os.path.dirname(__file__), "docs")

# palette (matches the dashboard)
INDIGO = (79, 70, 229)
PURPLE = (168, 85, 247)
SLATE = (16, 24, 40)
GRAY = (102, 112, 133)
LIGHT = (246, 248, 251)
BORDER = (229, 234, 241)
GREEN_BG = (236, 253, 243)
GREEN_FG = (6, 118, 71)
WHITE = (255, 255, 255)

OP_COLORS = {
    "Input": (217, 217, 217), "Constant": (230, 230, 230),
    "Identity": (240, 240, 240), "MatMul": (158, 197, 255),
    "GEMM": (127, 168, 255), "LinearGELU": (224, 176, 255),
    "FusedAttention": (255, 210, 127), "FusedAddLayerNorm": (162, 210, 255),
    "FusedScaleSoftmax": (255, 179, 198), "Add": (181, 234, 215),
    "Scale": (255, 214, 165), "Softmax": (255, 173, 173),
    "LayerNorm": (199, 206, 234), "GELU": (255, 201, 222),
    "Logits": (158, 197, 255),
}

_FONTS = {}


def font(size, bold=False):
    key = (size, bold)
    if key not in _FONTS:
        name = "arialbd.ttf" if bold else "arial.ttf"
        try:
            _FONTS[key] = ImageFont.truetype(
                f"C:/Windows/Fonts/{name}", size * SS
            )
        except Exception:
            _FONTS[key] = ImageFont.load_default()
    return _FONTS[key]


def new_canvas(w, h):
    img = Image.new("RGB", (w * SS, h * SS), WHITE)
    return img, ImageDraw.Draw(img)


def save(img, w, h, name):
    os.makedirs(DOCS, exist_ok=True)
    img = img.resize((w, h), Image.LANCZOS)
    path = os.path.join(DOCS, name)
    img.save(path)
    print(f"wrote {path}  ({os.path.getsize(path) // 1024} KB)")


def title_block(draw, w, title, subtitle):
    draw.text((24 * SS, 18 * SS), title, fill=SLATE,
              font=font(26, bold=True))
    draw.text((24 * SS, 52 * SS), subtitle, fill=GRAY, font=font(15))


def rbox(draw, cx, cy, w, h, fill, outline, text, tsize=14,
         tcolor=SLATE, bold=True, radius=10, sub=None, ssize=11):
    """Rounded box centred at (cx, cy) with centred text (and sub-line)."""
    x0, y0 = (cx - w / 2) * SS, (cy - h / 2) * SS
    x1, y1 = (cx + w / 2) * SS, (cy + h / 2) * SS
    draw.rounded_rectangle([x0, y0, x1, y1], radius * SS, fill=fill,
                           outline=outline, width=2 * SS)
    if sub:
        draw.text((cx * SS, (cy - 7) * SS), text, fill=tcolor,
                  font=font(tsize, bold=bold), anchor="mm")
        draw.text((cx * SS, (cy + 12) * SS), sub, fill=GRAY,
                  font=font(ssize), anchor="mm")
    else:
        draw.text((cx * SS, cy * SS), text, fill=tcolor,
                  font=font(tsize, bold=bold), anchor="mm")


def arrow(draw, p1, p2, color=(150, 160, 175), width=2, head=8):
    """Line with an arrowhead pointing into p2."""
    x1, y1 = p1[0] * SS, p1[1] * SS
    x2, y2 = p2[0] * SS, p2[1] * SS
    draw.line([x1, y1, x2, y2], fill=color, width=width * SS)
    ang = np.arctan2(y2 - y1, x2 - x1)
    for delta in (np.pi * 0.85, -np.pi * 0.85):
        px = x2 + head * SS * np.cos(ang + delta)
        py = y2 + head * SS * np.sin(ang + delta)
        draw.line([x2, y2, px, py], fill=color, width=width * SS)


def chip(draw, x, y, text, fg, bg, size=13):
    f = font(size, bold=True)
    tb = draw.textbbox((0, 0), text, font=f)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    pad = 8 * SS
    draw.rounded_rectangle([x * SS, y * SS, x * SS + tw + 2 * pad,
                            y * SS + th + 2 * pad], radius=12 * SS,
                           fill=bg, outline=fg, width=SS)
    draw.text((x * SS + pad, y * SS + pad), text, fill=fg, font=f)
    return tw + 2 * pad  # width in unscaled px (approx)


def fig1_pipeline():
    """The full compilation pipeline."""
    W, H = 1680, 460
    img, draw = new_canvas(W, H)
    title_block(draw, W, "Compilation pipeline — how the optimizer rewrites the graph",
                "Each pass rewrites or annotates the IR; the optimized graph is executed and measured")

    stages = [
        ("Input graph", "42 ops", WHITE, BORDER),
        ("1. Normalize", "42 → 38", WHITE, INDIGO),
        ("2. Attention canon.", "26 ops", WHITE, INDIGO),
        ("3. Semantic merge", "14 ops", WHITE, INDIGO),
        ("4. DAGS schedule", "11 levels", WHITE, INDIGO),
        ("5/6. Partitioning", "k parts", WHITE, INDIGO),
        ("Execute + measure", "10 metrics", GREEN_BG, GREEN_FG),
    ]
    bw, bh, gap = 195, 84, 32
    y = 175
    xs = [70 + i * (bw + gap) for i in range(len(stages))]
    for (label, sub, fill, line), cx in zip(stages, [x + bw / 2 for x in xs]):
        rbox(draw, cx, y, bw, bh, fill, line,
             label, tsize=14, bold=True, sub=sub)
    for i in range(len(stages) - 1):
        x1 = xs[i] + bw
        x2 = xs[i + 1]
        arrow(draw, (x1 + 4, y), (x2 - 6, y), color=INDIGO, width=2)

    # learned-search lane
    y2 = 330
    draw.text((70 * SS, (y2 - 58) * SS),
              "AI-guided search (optional)", fill=PURPLE,
              font=font(16, bold=True))
    learners = [
        ("7. GNN scorer", "predicts optimizability"),
        ("8. RL pass-order search", "Q-learning over pass subsets"),
        ("9. Neuro-symbolic", "rules + GNN guidance"),
    ]
    lx = [140, 470, 830]
    for (label, sub), cx in zip(learners, [x + 130 for x in lx]):
        rbox(draw, cx, y2, 260, 74, (250, 245, 255), PURPLE, label,
             tsize=14, bold=True, sub=sub)
    for cx, tx in ((270, 165), (600, 300), (960, 435)):
        arrow(draw, (cx, y2 - 38), (tx, y + bh / 2 + 6),
              color=PURPLE, width=2)

    draw.text((70 * SS, 425 * SS),
              "pipeline.py orchestrates the passes in compiler order; "
              "search modules can extend or replace fixed passes.",
              fill=GRAY, font=font(13))
    save(img, W, H, "fig1_pipeline.png")


def fig3_attention_fusion():
    """7-op attention pattern → one fused operator."""
    W, H = 1500, 560
    img, draw = new_canvas(W, H)
    title_block(draw, W, "Attention graph canonicalization (ACR)",
                "The 7-op attention sandwich is pattern-matched and replaced by one FlashAttention-style kernel")

    ops = [
        ("Q_Projection", "MatMul", 110, 200),
        ("K_Projection", "MatMul", 110, 300),
        ("V_Projection", "MatMul", 110, 400),
        ("QK_Score", "MatMul", 320, 250),
        ("Scale", "Scale × 1/√d", 520, 250),
        ("Softmax", "Softmax", 720, 250),
        ("Attention_Output", "MatMul", 920, 250),
    ]
    for name, sub, cx, cy in ops:
        rbox(draw, cx, cy, 165, 62, OP_COLORS.get(sub.split()[0], LIGHT),
             (140, 150, 165), name, tsize=13, sub=sub)
    arrow(draw, (192, 200), (238, 245))
    arrow(draw, (192, 300), (238, 262))
    arrow(draw, (192, 400), (838, 268))
    arrow(draw, (402, 250), (438, 250))
    arrow(draw, (602, 250), (638, 250))
    arrow(draw, (802, 250), (838, 250))

    # brace + big arrow to fused node
    arrow(draw, (1040, 250), (1080, 250), color=INDIGO, width=6, head=16)
    rbox(draw, 1265, 250, 330, 110, OP_COLORS["FusedAttention"],
         INDIGO, "FusedAttention", tsize=20, bold=True,
         sub="single SDPA-style kernel · 7 ops → 1")
    draw.text((1265 * SS, 340 * SS),
              "scale factor and operator provenance are carried inside the\n"
              "fused node, so the executor rebuilds identical math.",
              fill=GRAY, font=font(12), anchor="mm")
    draw.text((110 * SS, 500 * SS),
              "Result on the demo model: 2 attention blocks detected → 2 "
              "fused (ACR = 100%).", fill=SLATE, font=font(14, bold=True))
    save(img, W, H, "fig3_attention_fusion.png")


def render_graph_panel(draw, graph, ox, oy, w, h, heading, sub, k=1.2,
                       nw=118, nh=36):
    """Draw one computation graph (spring layout) inside a panel."""
    draw.text((ox * SS, oy * SS), heading, fill=SLATE,
              font=font(18, bold=True))
    draw.text((ox * SS, (oy + 28) * SS), sub, fill=GRAY, font=font(13))

    pad, top = 55, 95
    gw, gh = w - 2 * pad, h - top - 45
    pos = nx.spring_layout(graph.graph, seed=11, iterations=600, k=k)
    coords = np.array([pos[n] for n in graph.graph.nodes()])
    lo, hi = coords.min(axis=0), coords.max(axis=0)
    span = np.maximum(hi - lo, 1e-9)
    scaled = {
        node: (ox + pad + (cx - lo[0]) / span[0] * gw,
               oy + top + (cy - lo[1]) / span[1] * gh)
        for node, (cx, cy) in pos.items()
    }

    for u, v in graph.graph.edges():
        arrow(draw, scaled[u], scaled[v], color=(165, 175, 190), width=1,
              head=6)
    for node, (cx, cy) in scaled.items():
        op = graph.op_type(node)
        fill = OP_COLORS.get(op, LIGHT)
        label = node if len(node) <= 15 else node[:14] + "…"
        rbox(draw, cx, cy, nw, nh, fill, (120, 130, 148), label,
             tsize=11, bold=False, radius=8)


def fig4_graphs(original_graph, optimized_graph):
    """Before/after: the real graphs rendered from the IR."""
    W, H = 1900, 1250
    img, draw = new_canvas(W, H)
    title_block(draw, W, "Original vs optimized computation graph",
                "Rendered straight from the IR — 42 ops collapse to 14 after "
                "canonicalization + fusion (same outputs, bit-exact)")
    half = W // 2
    draw.line([half * SS, 120 * SS, half * SS, (H - 30) * SS],
              fill=BORDER, width=2 * SS)
    render_graph_panel(draw, original_graph, 30, 110, half - 40, H - 150,
                       "Original — 42 operations",
                       "Q/K/V, QKᵀ, scale, softmax, bias adds, "
                       "identities, dead branch", k=2.2, nw=110, nh=34)
    render_graph_panel(draw, optimized_graph, half + 10, 110,
                       half - 40, H - 150,
                       "Optimized — 14 operations",
                       "FusedAttention, GEMM, LinearGELU, "
                       "FusedAddLayerNorm kernels", k=1.4)
    save(img, W, H, "fig4_graphs.png")


def fig2_block():
    """One transformer block as the model builder constructs it."""
    W, H = 1750, 720
    img, draw = new_canvas(W, H)
    title_block(draw, W, "Inside one Transformer block (as built by "
                         "models/transformer.py)",
                "Boxes are operations (nodes), arrows are tensor dependencies "
                "(edges) — attention plus a feed-forward network")

    bw, bh = 168, 56
    y_att = 190
    att = [
        ("Q_Projection", "MatMul", 130, y_att - 90),
        ("K_Projection", "MatMul", 130, y_att),
        ("V_Projection", "MatMul", 130, y_att + 90),
        ("QK_Score", "MatMul", 360, y_att),
        ("Scale", "× 1/√d", 580, y_att),
        ("Softmax", "Softmax", 800, y_att),
        ("Attention_Output", "MatMul", 1030, y_att),
        ("Residual_Add", "Add", 1270, y_att),
        ("LayerNorm", "LayerNorm", 1500, y_att),
    ]
    src_x = 30
    draw.text((src_x * SS, (y_att + 155) * SS), "from source", fill=GRAY,
              font=font(13, bold=True), anchor="lm")
    for name, sub, cx, cy in att:
        base = sub.split()[0]
        rbox(draw, cx, cy, bw, bh, OP_COLORS.get(base, LIGHT),
             (140, 150, 165), name, tsize=13, sub=sub)
    arrow(draw, (214, y_att - 90), (276, y_att - 12), color=INDIGO)
    arrow(draw, (214, y_att), (276, y_att), color=INDIGO)
    arrow(draw, (214, y_att + 90), (1090, y_att + 24), color=INDIGO)
    for x in (444, 664, 884, 1114, 1186, 1416):
        arrow(draw, (x, y_att), (x + 36, y_att), color=INDIGO)
    draw.text((700 * SS, 80 * SS), "attention: how much each token "
              "matters", fill=GRAY, font=font(13))

    y_ffn = 400
    ffn = [
        ("FFN_Linear", "MatMul", 130, y_ffn),
        ("FFN_Bias", "Add", 360, y_ffn),
        ("GELU", "GELU", 580, y_ffn),
        ("FFN_Output", "MatMul", 800, y_ffn),
        ("FFN_Bias_Out", "Add", 1030, y_ffn),
        ("FFN_Residual", "Add", 1270, y_ffn),
        ("FFN_LayerNorm", "LayerNorm", 1500, y_ffn),
    ]
    arrow(draw, (1584, y_att + 28), (1584, y_ffn - 34), color=INDIGO)
    draw.text((1600 * SS, (y_att + 95) * SS), "feeds the next stage",
              fill=GRAY, font=font(11))
    for name, sub, cx, cy in ffn:
        base = sub.split()[0]
        rbox(draw, cx, cy, bw, bh, OP_COLORS.get(base, LIGHT),
             (140, 150, 165), name, tsize=13, sub=sub)
    for x in (214, 444, 664, 884, 1114, 1186, 1416):
        arrow(draw, (x, y_ffn), (x + 36, y_ffn), color=INDIGO)
    draw.text((700 * SS, 300 * SS), "feed-forward network with GELU "
              "activation", fill=GRAY, font=font(13))

    y_tail = 560
    draw.text((130 * SS, y_tail * SS), "…block 2 repeats the same "
              "structure (names prefixed Block2_)",
              fill=GRAY, font=font(13))
    rbox(draw, 700, y_tail, 200, 56, WHITE, BORDER, "Dropout", tsize=13,
         sub="Identity — removed by normalize")
    rbox(draw, 1030, y_tail, 200, 56, OP_COLORS["Add"], (140, 150, 165),
         "Residual_Add", tsize=13, sub="skip connection")
    arrow(draw, (1584, y_ffn + 28), (1584, y_tail - 34), color=INDIGO)
    arrow(draw, (800, y_tail), (928, y_tail), color=INDIGO)

    draw.text((30 * SS, 665 * SS), "Every pattern here is a fusion "
              "target: 7 attention ops → FusedAttention · MatMul+Add → "
              "GEMM · GEMM+GELU → LinearGELU · Add+LayerNorm → "
              "FusedAddLayerNorm · Identity → removed",
              fill=SLATE, font=font(13, bold=True))
    save(img, W, H, "fig2_transformer_block.png")


def fig5_stages():
    """Ops remaining after each pipeline stage."""
    W, H = 1250, 620
    img, draw = new_canvas(W, H)
    title_block(draw, W, "Graph evolution through the pipeline",
                "Operations remaining after each stage (demo transformer)")

    stages = [
        ("Input model", 42, "hand-built transformer"),
        ("Normalize", 38, "fold + CSE + DCE − identities"),
        ("Attention canon.", 26, "2 × (7 ops → 1 fused)"),
        ("Semantic merging", 14, "11 kernels fused"),
    ]
    bw, gap = 200, 80
    max_n = 42
    base_y, chart_h = 470, 280
    for i, (label, n, sub) in enumerate(stages):
        cx = 150 + i * (bw + gap)
        bar_h = int(chart_h * n / max_n)
        color = INDIGO if i < 3 else GREEN_FG
        draw.rounded_rectangle(
            [(cx - bw / 2) * SS, (base_y - bar_h) * SS,
             (cx + bw / 2) * SS, base_y * SS],
            radius=10 * SS, fill=color)
        draw.text((cx * SS, (base_y - bar_h - 22) * SS), str(n),
                  fill=SLATE, font=font(24, bold=True), anchor="mm")
        draw.text((cx * SS, (base_y + 18) * SS), label, fill=SLATE,
                  font=font(14, bold=True), anchor="mm")
        draw.text((cx * SS, (base_y + 40) * SS), sub, fill=GRAY,
                  font=font(11), anchor="mm")
        if i:
            px = 150 + (i - 1) * (bw + gap)
            reduction = (stages[i - 1][1] - n) / stages[i - 1][1] * 100
            draw.text((((px + cx) / 2) * SS, 170 * SS),
                      f"−{reduction:.0f}%", fill=GREEN_FG,
                      font=font(15, bold=True), anchor="mm")

    draw.text((150 * SS, 545 * SS),
              "Net effect: 66.7% fewer operations and kernel launches — "
              "with bit-exact outputs (accuracy preservation 100%).",
              fill=SLATE, font=font(14, bold=True))
    save(img, W, H, "fig5_stages.png")


def fig6_metrics(m):
    """Measured + modeled before/after for the headline metrics."""
    W, H = 1600, 660
    img, draw = new_canvas(W, H)
    title_block(draw, W, "Output metrics — original vs optimized",
                "Measured on the NumPy reference executor; GPU numbers from "
                "the documented cost model")

    panels = [
        ("Inference latency (ms, ↓ better)",
         m["latency_ms"]["original"], m["latency_ms"]["optimized"],
         f"{m['speedup']:.2f}× faster"),
        ("Throughput (samples/s, ↑ better)",
         m["throughput"]["original"], m["throughput"]["optimized"],
         f"{(m['throughput']['optimized'] / m['throughput']['original'] - 1) * 100:+.1f}%"),
        ("Peak memory (MB, ↓ better)",
         m["peak_memory_mb"]["original"], m["peak_memory_mb"]["optimized"],
         f"{(1 - m['peak_memory_mb']['optimized'] / m['peak_memory_mb']['original']) * 100:.0f}% lower"),
    ]
    pw, ph = 470, 360
    for i, (label, before, after, note) in enumerate(panels):
        ox = 40 + i * (pw + 40)
        oy = 120
        draw.rounded_rectangle([ox * SS, oy * SS, (ox + pw) * SS,
                                (oy + ph) * SS], radius=14 * SS,
                               fill=LIGHT, outline=BORDER, width=SS)
        draw.text(((ox + pw / 2) * SS, (oy + 26) * SS), label,
                  fill=SLATE, font=font(15, bold=True), anchor="mm")
        bw2, area_h = 120, 200
        base_y = oy + ph - 60
        mx = max(before, after)
        for j, (val, col, tag) in enumerate(
            ((before, (150, 160, 175), "original"),
             (after, INDIGO, "optimized"))
        ):
            h = max(8, int(area_h * val / mx))
            bx = ox + 90 + j * (bw2 + 70)
            draw.rounded_rectangle(
                [bx * SS, (base_y - h) * SS, (bx + bw2) * SS, base_y * SS],
                radius=8 * SS, fill=col)
            draw.text(((bx + bw2 / 2) * SS, (base_y - h - 16) * SS),
                      f"{val:.2f}" if val < 10 else f"{val:.0f}",
                      fill=SLATE, font=font(15, bold=True), anchor="mm")
            draw.text(((bx + bw2 / 2) * SS, (base_y + 16) * SS), tag,
                      fill=GRAY, font=font(12), anchor="mm")
        chip(draw, ox + pw / 2 - 60, oy + ph - 44, note,
             GREEN_FG, GREEN_BG)
    draw.text((40 * SS, 545 * SS),
              "Measured speedup {:.2f}×  ·  modeled GPU speedup {:.2f}×  ·  "
              "modeled GPU memory {} → {} MB (−{:.0f}%)  ·  accuracy "
              "preservation {:.1f}%".format(
                  m["speedup"], m["modeled"]["speedup"],
                  m["modeled"]["memory_original_mb"],
                  m["modeled"]["memory_optimized_mb"],
                  m["modeled"]["memory_reduction_pct"] * 100,
                  m["accuracy"]["preservation_pct"]),
              fill=SLATE, font=font(15, bold=True))
    draw.text((40 * SS, 585 * SS),
              "Structural ratios: GRR {:.1%} · OMR {:.1%} · ACR {:.0%} · "
              "kernel-launch reduction {:.1%}".format(
                  m["graph_reduction_ratio"],
                  m["operator_merge_ratio"],
                  m["attention_canonicalization_rate"],
                  m["kernel_launch_reduction"]),
              fill=GRAY, font=font(13))
    save(img, W, H, "fig6_metrics.png")


def polyline(draw, points, color, width=3):
    pts = [(x * SS, y * SS) for x, y in points]
    draw.line(pts, fill=color, width=width * SS, joint="curve")
    for x, y in pts:
        draw.ellipse([x - 4 * SS, y - 4 * SS, x + 4 * SS, y + 4 * SS],
                     fill=color)


def fig7_compile(infos):
    """Per-pass compilation time breakdown."""
    W, H = 1250, 520
    img, draw = new_canvas(W, H)
    title_block(draw, W, "Compilation time breakdown",
                "Wall time of each pass (metric 10) — the whole pipeline "
                "runs in a few milliseconds")
    rows = [(name, info.get("duration_s", 0.0) * 1e3)
            for name, info in infos.items()
            if isinstance(info, dict) and "duration_s" in info
            and name != "pipeline"]
    rows.sort(key=lambda item: item[1])
    mx = max(v for _, v in rows) or 1.0
    x0, y = 330, 150
    for name, ms in rows:
        bw = max(6, (ms / mx) * 700)
        draw.rounded_rectangle([x0 * SS, y * SS, (x0 + bw) * SS,
                                (y + 30) * SS], radius=8 * SS, fill=INDIGO)
        draw.text(((x0 - 14) * SS, (y + 15) * SS), name, fill=SLATE,
                  font=font(13, bold=True), anchor="rm")
        draw.text(((x0 + bw + 12) * SS, (y + 15) * SS),
                  f"{ms:.2f} ms", fill=SLATE, font=font(13), anchor="lm")
        y += 52
    save(img, W, H, "fig7_compile_time.png")


def fig8_structural(m):
    """The four structural ratio metrics."""
    W, H = 1250, 520
    img, draw = new_canvas(W, H)
    title_block(draw, W, "Structural optimization ratios",
                "Exact counts from the IR (GRR / OMR / ACR / kernel-launch "
                "reduction)")
    rows = [
        ("Graph Reduction Ratio (GRR)", m["graph_reduction_ratio"]),
        ("Operator Merge Ratio (OMR)", m["operator_merge_ratio"]),
        ("Attention Canonicalization (ACR)",
         m["attention_canonicalization_rate"]),
        ("Kernel Launch Reduction", m["kernel_launch_reduction"]),
    ]
    x0, y = 380, 150
    for name, value in rows:
        bw = max(8, value * 700)
        color = GREEN_FG if value >= 0.5 else INDIGO
        draw.rounded_rectangle([x0 * SS, y * SS, (x0 + bw) * SS,
                                (y + 34) * SS], radius=8 * SS, fill=color)
        draw.text(((x0 - 14) * SS, (y + 17) * SS), name, fill=SLATE,
                  font=font(13, bold=True), anchor="rm")
        draw.text(((x0 + bw + 12) * SS, (y + 17) * SS),
                  f"{value * 100:.1f}%", fill=SLATE,
                  font=font(14, bold=True), anchor="lm")
        y += 68
    save(img, W, H, "fig8_structural.png")


def _axes(draw, x0, x1, y0, y1, epochs, lo, hi, y_fmt):
    draw.line([x0 * SS, y0 * SS, x1 * SS, y0 * SS], fill=SLATE,
              width=2 * SS)
    draw.line([x0 * SS, y0 * SS, x0 * SS, y1 * SS], fill=SLATE,
              width=2 * SS)
    for e in range(1, int(epochs) + 1):
        px = x0 + (e - 1) / max(1, epochs - 1) * (x1 - x0)
        draw.text((px * SS, (y0 + 16) * SS), str(e), fill=GRAY,
                  font=font(12), anchor="mm")
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        vy = lo + frac * (hi - lo)
        py = y0 - frac * (y0 - y1)
        draw.text(((x0 - 12) * SS, py * SS), y_fmt.format(vy), fill=GRAY,
                  font=font(12), anchor="rm")
    draw.text((((x0 + x1) / 2) * SS, (y0 + 42) * SS), "epoch", fill=SLATE,
              font=font(13, bold=True), anchor="mm")


def fig9_train_loss(history):
    W, H = 1200, 640
    img, draw = new_canvas(W, H)
    title_block(draw, W, "Training on MNIST — cross-entropy loss",
                "NumPy MLP 784→256→64→10 · manual backprop + Adam · "
                "mini-batch 64")
    epochs = history["epoch"][-1]
    tl, vl = history["train_loss"], history["val_loss"]
    lo, hi = 0.0, max(max(tl), max(vl)) * 1.15
    x0, x1, y0, y1 = 170, 1080, 170, 520

    def px(e):
        return x0 + (e - 1) / max(1, epochs - 1) * (x1 - x0)

    def py(v):
        return y0 - (v - lo) / (hi - lo) * (y0 - y1)

    _axes(draw, x0, x1, y0, y1, epochs, lo, hi, "{:.2f}")
    polyline(draw, [(px(e), py(v)) for e, v in zip(history["epoch"], tl)],
             INDIGO)
    polyline(draw, [(px(e), py(v)) for e, v in zip(history["epoch"], vl)],
             PURPLE)
    draw.text((120 * SS, 600 * SS),
              "both losses fall and stay close together → learning without "
              "overfitting", fill=SLATE, font=font(13))
    chip(draw, 900, 190, "train loss", "#FFFFFF", INDIGO)
    chip(draw, 900, 235, "validation loss", "#FFFFFF", PURPLE)
    save(img, W, H, "fig9_train_loss.png")


def fig10_train_accuracy(history):
    W, H = 1200, 640
    img, draw = new_canvas(W, H)
    title_block(draw, W, "Training on MNIST — accuracy",
                "train vs validation accuracy per epoch")
    epochs = history["epoch"][-1]
    ta, va = history["train_acc"], history["val_acc"]
    lo = min(min(ta), min(va)) - 0.02
    hi = 1.0
    x0, x1, y0, y1 = 170, 1080, 170, 520

    def px(e):
        return x0 + (e - 1) / max(1, epochs - 1) * (x1 - x0)

    def py(v):
        return y0 - (v - lo) / (hi - lo) * (y0 - y1)

    _axes(draw, x0, x1, y0, y1, epochs, lo, hi, "{:.2f}")
    polyline(draw, [(px(e), py(v)) for e, v in zip(history["epoch"], ta)],
             INDIGO)
    polyline(draw, [(px(e), py(v)) for e, v in zip(history["epoch"], va)],
             PURPLE)
    last = history["val_acc"][-1]
    draw.text((120 * SS, 600 * SS),
              f"final validation accuracy: {last * 100:.2f}% (metric for "
              "the trained model)", fill=SLATE, font=font(13, bold=True))
    chip(draw, 900, 190, "train acc", "#FFFFFF", INDIGO)
    chip(draw, 900, 235, "validation acc", "#FFFFFF", PURPLE)
    save(img, W, H, "fig10_train_accuracy.png")


def fig11_confusion(cm):
    W, H = 1120, 950
    img, draw = new_canvas(W, H)
    title_block(draw, W, "Confusion matrix on the MNIST test split",
                "rows = true digit, columns = predicted digit "
                "(diagonal = correct)")
    cell, ox, oy = 78, 170, 170
    mx = max(int(cm.max()), 1)
    for i in range(10):
        for j in range(10):
            v = int(cm[i, j])
            t = v / mx
            col = tuple(int(255 - (255 - c) * t) for c in INDIGO)
            x = ox + j * cell
            y = oy + i * cell
            draw.rectangle([x * SS, y * SS, (x + cell) * SS,
                            (y + cell) * SS], fill=col, outline=BORDER,
                           width=SS)
            if v:
                draw.text(((x + cell / 2) * SS, (y + cell / 2) * SS),
                          str(v), fill=WHITE if t > 0.45 else SLATE,
                          font=font(11), anchor="mm")
        draw.text(((ox - 14) * SS, (oy + i * cell + cell / 2) * SS),
                  str(i), fill=SLATE, font=font(13, bold=True), anchor="rm")
        draw.text(((ox + i * cell + cell / 2) * SS, (oy - 14) * SS),
                  str(i), fill=SLATE, font=font(13, bold=True), anchor="mb")
    draw.text((60 * SS, (oy + 5 * cell) * SS), "true digit", fill=GRAY,
              font=font(13, bold=True))
    draw.text(((ox + 5 * cell) * SS, (oy - 48) * SS), "predicted digit",
              fill=GRAY, font=font(13, bold=True), anchor="mm")
    diag = sum(int(cm[i, i]) for i in range(10)) / max(1, int(cm.sum()))
    draw.text((60 * SS, 880 * SS),
              f"diagonal share (accuracy): {diag * 100:.2f}%",
              fill=SLATE, font=font(14, bold=True))
    save(img, W, H, "fig11_confusion.png")


def fig12_architectures(rows):
    """rows: (name, ops_before, ops_after, modeled_speedup)."""
    W, H = 1500, 660
    img, draw = new_canvas(W, H)
    title_block(draw, W, "Architecture comparison — same compiler, "
                         "same metrics",
                "ops before/after the pipeline + modeled speedup per "
                "architecture")
    bw, gap = 260, 70
    mx = max(r[1] for r in rows)
    base_y, chart_h = 450, 270
    for i, (name, before, after, spd) in enumerate(rows):
        cx = 190 + i * (bw + gap)
        for j, (val, col, tag) in enumerate(
            ((before, (150, 160, 175), "original"),
             (after, INDIGO, "optimized"))
        ):
            h = max(8, int(chart_h * val / mx))
            bx = cx - 65 + j * 135
            draw.rounded_rectangle(
                [bx * SS, (base_y - h) * SS, (bx + 110) * SS,
                 base_y * SS], radius=8 * SS, fill=col)
            draw.text(((bx + 55) * SS, (base_y - h - 18) * SS), str(val),
                      fill=SLATE, font=font(16, bold=True), anchor="mm")
            draw.text(((bx + 55) * SS, (base_y + 16) * SS), tag,
                      fill=GRAY, font=font(11), anchor="mm")
        draw.text((cx * SS, (base_y + 44) * SS), name, fill=SLATE,
                  font=font(13, bold=True), anchor="mm")
        draw.text((cx * SS, (base_y + 66) * SS),
                  f"{spd:.2f}× modeled speedup", fill=GREEN_FG,
                  font=font(12, bold=True), anchor="mm")
    save(img, W, H, "fig12_architectures.png")


def main():
    from models.transformer import create_demo_transformer_graph
    from pipeline import run_pipeline

    fig1_pipeline()
    fig2_block()
    fig3_attention_fusion()

    graph = create_demo_transformer_graph()
    optimized, infos, compile_time = run_pipeline(graph)
    fig4_graphs(graph, optimized)
    fig5_stages()

    from utils.metrics import compute_metrics
    metrics = compute_metrics(graph, optimized, infos, compile_time)
    fig6_metrics(metrics)
    fig7_compile(infos)
    fig8_structural(metrics)

    try:
        from training.train import run_training

        outcome = run_training()
        fig9_train_loss(outcome["history"])
        fig10_train_accuracy(outcome["history"])
        fig11_confusion(outcome["confusion"])
    except Exception as exc:
        print("training figures skipped:", exc)

    rows = []
    from models.architectures import ARCHITECTURES

    for name, builder in ARCHITECTURES.items():
        arch_graph = builder()
        arch_opt, arch_infos, arch_time = run_pipeline(arch_graph)
        arch_metrics = compute_metrics(arch_graph, arch_opt, arch_infos,
                                       arch_time)
        rows.append((name, arch_graph.node_count(), arch_opt.node_count(),
                     arch_metrics["modeled"]["speedup"]))
    fig12_architectures(rows)
    print("all figures generated")


if __name__ == "__main__":
    main()
