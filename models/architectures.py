"""Model architecture variants for the architecture comparison study.

All builders emit the same ComputationGraph IR so the identical
pipeline (and its metrics) applies to every architecture:

* ``create_postln_transformer_graph``  — classic post-LN Transformer
  (the reference; delegates to models.transformer).
* ``create_preln_transformer_graph``   — GPT-style pre-LN Transformer
  (LayerNorm inside the residual branch, final LayerNorm on top).
* ``create_relu_transformer_graph``    — post-LN with ReLU activations
  (exercises the LinearRelu fusion path).
* ``create_mlp_classifier_graph``      — a plain MLP classifier (also
  used to export the trained MNIST model).
"""

import numpy as np

from ir.graph import ComputationGraph
from models.transformer import create_demo_transformer_graph


def create_postln_transformer_graph(num_blocks=2, batch=4, seq=32,
                                    d_model=64):
    """Reference architecture (delegates to the demo builder)."""
    return create_demo_transformer_graph(num_blocks=num_blocks,
                                         batch=batch, seq=seq,
                                         d_model=d_model)


def create_preln_transformer_graph(num_blocks=2, batch=4, seq=32,
                                   d_model=64):
    """GPT-style pre-LN Transformer: LayerNorm inside each branch."""
    graph = ComputationGraph(f"preln_transformer_{num_blocks}b")
    graph.meta["config"] = {"batch": batch, "seq": seq, "d_model": d_model}
    scale_factor = float(1.0 / np.sqrt(d_model))

    graph.add_operation("Input", "Input")
    node = "Input"
    for i in range(1, num_blocks + 1):
        p = f"Block{i}"
        graph.add_operation(f"{p}_LN1", "LayerNorm")
        graph.add_dependency(node, f"{p}_LN1")
        for role in ("Q", "K", "V"):
            graph.add_operation(f"{p}_{role}_Projection", "MatMul")
            graph.add_dependency(f"{p}_LN1", f"{p}_{role}_Projection")
        graph.add_operation(f"{p}_QK_Score", "MatMul")
        graph.add_dependency(f"{p}_Q_Projection", f"{p}_QK_Score")
        graph.add_dependency(f"{p}_K_Projection", f"{p}_QK_Score")
        graph.add_operation(f"{p}_Scale", "Scale", factor=scale_factor)
        graph.add_dependency(f"{p}_QK_Score", f"{p}_Scale")
        graph.add_operation(f"{p}_Softmax", "Softmax")
        graph.add_dependency(f"{p}_Scale", f"{p}_Softmax")
        graph.add_operation(f"{p}_Attention_Output", "MatMul")
        graph.add_dependency(f"{p}_Softmax", f"{p}_Attention_Output")
        graph.add_dependency(f"{p}_V_Projection", f"{p}_Attention_Output")
        graph.add_operation(f"{p}_Attn_Residual", "Add")
        graph.add_dependency(f"{p}_Attention_Output", f"{p}_Attn_Residual")
        graph.add_dependency(node, f"{p}_Attn_Residual")

        graph.add_operation(f"{p}_LN2", "LayerNorm")
        graph.add_dependency(f"{p}_Attn_Residual", f"{p}_LN2")
        graph.add_operation(f"{p}_FFN_Linear", "MatMul")
        graph.add_dependency(f"{p}_LN2", f"{p}_FFN_Linear")
        graph.add_operation(f"{p}_FFN_Bias", "Add")
        graph.add_dependency(f"{p}_FFN_Linear", f"{p}_FFN_Bias")
        graph.add_operation(f"{p}_GELU", "GELU")
        graph.add_dependency(f"{p}_FFN_Bias", f"{p}_GELU")
        graph.add_operation(f"{p}_FFN_Output", "MatMul")
        graph.add_dependency(f"{p}_GELU", f"{p}_FFN_Output")
        graph.add_operation(f"{p}_FFN_Bias_Output", "Add")
        graph.add_dependency(f"{p}_FFN_Output", f"{p}_FFN_Bias_Output")
        graph.add_operation(f"{p}_FFN_Residual", "Add")
        graph.add_dependency(f"{p}_FFN_Bias_Output", f"{p}_FFN_Residual")
        graph.add_dependency(f"{p}_Attn_Residual", f"{p}_FFN_Residual")
        node = f"{p}_FFN_Residual"

    graph.add_operation("Final_LayerNorm", "LayerNorm")
    graph.add_dependency(node, "Final_LayerNorm")
    graph.add_operation("Logits", "MatMul")
    graph.add_dependency("Final_LayerNorm", "Logits")
    graph.add_operation("Logits_Bias", "Add", is_output=True)
    graph.add_dependency("Logits", "Logits_Bias")
    return graph


def create_relu_transformer_graph(num_blocks=2, batch=4, seq=32,
                                  d_model=64):
    """Post-LN Transformer with ReLU activations (LinearRelu fusion)."""
    return create_demo_transformer_graph(num_blocks=num_blocks,
                                         batch=batch, seq=seq,
                                         d_model=d_model, act="relu")


def create_mlp_classifier_graph(sizes=(784, 256, 64, 10), params=None,
                                act="relu", name="mlp_classifier",
                                batch=256):
    """Plain MLP classifier graph; ``params`` = trained [W, b] pairs."""
    graph = ComputationGraph(name)
    graph.meta["config"] = {"batch": batch, "input_shape": [sizes[0]]}
    rng = np.random.default_rng(7)

    graph.add_operation("Input", "Input")
    node = "Input"
    for i, (fan_in, fan_out) in enumerate(zip(sizes[:-1], sizes[1:])):
        if params is not None:
            W, b = params[i]
        else:
            W = rng.normal(0.0, np.sqrt(2.0 / fan_in),
                           (fan_in, fan_out)).astype(np.float32)
            b = np.zeros(fan_out, dtype=np.float32)
        mat = f"Dense{i}_MatMul"
        graph.add_operation(mat, "MatMul", weight=W)
        graph.add_dependency(node, mat)
        const = f"Dense{i}_BiasConst"
        graph.add_operation(const, "Constant", value=b,
                            shape=tuple(b.shape))
        add = f"Dense{i}_Bias"
        graph.add_operation(add, "Add")
        graph.add_dependency(mat, add)
        graph.add_dependency(const, add)
        node = add
        if i < len(sizes) - 2:
            act_op = "GELU" if act == "gelu" else "Relu"
            act_name = f"{act.capitalize()}{i}"
            graph.add_operation(act_name, act_op)
            graph.add_dependency(node, act_name)
            node = act_name

    graph.add_operation("Probabilities", "Softmax", is_output=True)
    graph.add_dependency(node, "Probabilities")
    return graph


ARCHITECTURES = {
    "Post-LN Transformer": create_postln_transformer_graph,
    "Pre-LN Transformer (GPT-style)": create_preln_transformer_graph,
    "Post-LN + ReLU FFN": create_relu_transformer_graph,
    "MLP classifier": create_mlp_classifier_graph,
}
