"""Graphviz rendering with op-type coloring, fusion and partition marks."""

OP_COLORS = {
    "Input": "#d9d9d9",
    "Constant": "#e6e6e6",
    "Identity": "#f0f0f0",
    "MatMul": "#9ec5ff",
    "GEMM": "#7fa8ff",
    "LinearGELU": "#e0b0ff",
    "FusedAttention": "#ffd27f",
    "FusedAddLayerNorm": "#a2d2ff",
    "FusedScaleSoftmax": "#ffb3c6",
    "Add": "#b5ead7",
    "Mul": "#c1e1c1",
    "Sub": "#c1e1c1",
    "Scale": "#ffd6a5",
    "Softmax": "#ffadad",
    "LayerNorm": "#c7ceea",
    "GELU": "#ffc9de",
    "Relu": "#c8f7dc",
    "LinearRelu": "#b9f0ff",
    "Gather": "#ffe0b3",
    "Slice": "#e6e0f8",
    "Reshape": "#eeeeee",
    "Transpose": "#e0e0e0",
    "MaskedFill": "#ffd9d9",
    "Broadcast": "#eeeeee",
    "Cast": "#eeeeee",
    "Tanh": "#ffc9de",
    "Pow": "#ffc9de",
    "Operation": "#cfe2ff",
}

FUSED_OPS = {
    "GEMM",
    "LinearGELU",
    "FusedAttention",
    "FusedAddLayerNorm",
    "FusedScaleSoftmax",
}


def graph_to_dot(graph, show_schedule=True, show_partition=True):
    """Render the computation graph as a colored Graphviz DOT string."""
    lines = [
        "digraph G {",
        "    rankdir=TB;",
        '    bgcolor="white";',
        '    graph [pad="0.3", nodesep="0.4", ranksep="0.6", splines=polyline];',
        '    node [shape=box, style="rounded,filled", fontname="Arial", '
        'fontsize=10, margin="0.15,0.08"];',
        '    edge [color="gray40", arrowsize=0.7];',
    ]

    partitions = {}
    for node, data in graph.graph.nodes(data=True):
        op_type = data.get("op_type", "Operation")
        color = OP_COLORS.get(op_type, "#cfe2ff")
        marks = []
        if show_schedule and "schedule_order" in data:
            marks.append(
                f"#{data['schedule_order']}L{data['schedule_level']}"
            )
        if show_partition and "partition" in data:
            marks.append(f"P{data['partition']}")
        label = f"{node}\\n({op_type})"
        if marks:
            label += "\\n[" + " ".join(marks) + "]"
        style = "rounded,filled"
        if data.get("fused") or op_type in FUSED_OPS:
            style = "rounded,filled,bold"
        attrs = f'label="{label}", fillcolor="{color}", style="{style}"'
        partition = data.get("partition") if show_partition else None
        partitions.setdefault(partition, []).append(
            f'        "{node}" [{attrs}];'
        )

    if len(partitions) == 1 or None in partitions:
        for nodes in partitions.values():
            lines.extend(nodes)
    else:
        for partition in sorted(partitions):
            lines.append(f"    subgraph cluster_{partition} {{")
            lines.append(f'        label="Partition {partition}";')
            lines.append('        style="rounded,dashed"; color="#888888";')
            lines.extend(partitions[partition])
            lines.append("    }")

    for source, target in graph.graph.edges():
        lines.append(f'    "{source}" -> "{target}";')

    lines.append("}")
    return "\n".join(lines)
