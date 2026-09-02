"""Demo model graphs used by the optimizer UI, CLI and tests."""

import numpy as np

from ir.graph import ComputationGraph


def _add_transformer_block(graph, prefix, source, d_model):
    """Append one post-LN transformer block; returns the block output node."""
    p = prefix
    scale_factor = float(1.0 / np.sqrt(d_model))

    graph.add_operation(f"{p}_Q_Projection", "MatMul")
    graph.add_operation(f"{p}_K_Projection", "MatMul")
    graph.add_operation(f"{p}_V_Projection", "MatMul")
    graph.add_operation(f"{p}_QK_Score", "MatMul")
    graph.add_operation(f"{p}_Scale", "Scale", factor=scale_factor)
    graph.add_operation(f"{p}_Softmax", "Softmax")
    graph.add_operation(f"{p}_Attention_Output", "MatMul")
    graph.add_operation(f"{p}_Residual_Add", "Add")
    graph.add_operation(f"{p}_LayerNorm", "LayerNorm")
    graph.add_operation(f"{p}_FFN_Linear", "MatMul")
    graph.add_operation(f"{p}_FFN_Bias", "Add")
    graph.add_operation(f"{p}_GELU", "GELU")
    graph.add_operation(f"{p}_FFN_Output", "MatMul")
    graph.add_operation(f"{p}_FFN_Bias_Output", "Add")
    graph.add_operation(f"{p}_FFN_Residual_Add", "Add")
    graph.add_operation(f"{p}_FFN_LayerNorm", "LayerNorm")
    graph.add_operation(f"{p}_Dropout", "Identity")

    deps = [
        (source, f"{p}_Q_Projection"),
        (source, f"{p}_K_Projection"),
        (source, f"{p}_V_Projection"),
        (f"{p}_Q_Projection", f"{p}_QK_Score"),
        (f"{p}_K_Projection", f"{p}_QK_Score"),
        (f"{p}_QK_Score", f"{p}_Scale"),
        (f"{p}_Scale", f"{p}_Softmax"),
        (f"{p}_Softmax", f"{p}_Attention_Output"),
        (f"{p}_V_Projection", f"{p}_Attention_Output"),
        (f"{p}_Attention_Output", f"{p}_Residual_Add"),
        (source, f"{p}_Residual_Add"),
        (f"{p}_Residual_Add", f"{p}_LayerNorm"),
        (f"{p}_LayerNorm", f"{p}_FFN_Linear"),
        (f"{p}_FFN_Linear", f"{p}_FFN_Bias"),
        (f"{p}_FFN_Bias", f"{p}_GELU"),
        (f"{p}_GELU", f"{p}_FFN_Output"),
        (f"{p}_FFN_Output", f"{p}_FFN_Bias_Output"),
        (f"{p}_FFN_Bias_Output", f"{p}_FFN_Residual_Add"),
        (f"{p}_LayerNorm", f"{p}_FFN_Residual_Add"),
        (f"{p}_FFN_Residual_Add", f"{p}_FFN_LayerNorm"),
        (f"{p}_FFN_LayerNorm", f"{p}_Dropout"),
    ]
    for src, dst in deps:
        graph.add_dependency(src, dst)

    return f"{p}_Dropout"


def create_demo_transformer_graph(num_blocks=2, batch=4, seq=32, d_model=64):
    """Build a small transformer computation graph for demo purposes."""
    graph = ComputationGraph("demo_transformer")
    graph.meta["config"] = {"batch": batch, "seq": seq, "d_model": d_model}

    rng = np.random.default_rng(42)

    graph.add_operation("Input", "Input")

    pos = rng.normal(0.0, 0.02, size=(seq, d_model)).astype(np.float32)
    bias = rng.normal(0.0, 0.01, size=(d_model,)).astype(np.float32)

    graph.add_operation(
        "Const_Pos_Embed", "Constant", value=pos, shape=(seq, d_model)
    )
    graph.add_operation(
        "Const_Logit_Bias", "Constant", value=bias, shape=(d_model,)
    )

    graph.add_operation("Embed_Scale", "Scale", factor=0.5)
    graph.add_dependency("Const_Pos_Embed", "Embed_Scale")

    graph.add_operation("Embed_Add", "Add")
    graph.add_dependency("Embed_Scale", "Embed_Add")
    graph.add_dependency("Input", "Embed_Add")

    node = "Embed_Add"
    for index in range(1, num_blocks + 1):
        node = _add_transformer_block(graph, f"Block{index}", node, d_model)

    graph.add_operation("Logits", "MatMul")
    graph.add_dependency(node, "Logits")

    graph.add_operation("Logits_Bias", "Add", is_output=True)
    graph.add_dependency("Logits", "Logits_Bias")
    graph.add_dependency("Const_Logit_Bias", "Logits_Bias")

    # Dead debug branch: exercises constant folding + dead-code elimination.
    graph.add_operation("Unused_Debug", "Add")
    graph.add_dependency("Const_Logit_Bias", "Unused_Debug")
    graph.add_dependency("Const_Pos_Embed", "Unused_Debug")

    return graph
