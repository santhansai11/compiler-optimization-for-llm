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
        graph.add_operation(f"{p}_QK_Score", "MatMul", transpose_rhs=True)
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


def create_llama_style_graph(num_blocks=2, batch=4, seq=32, d_model=64):
    """LLaMA-style decoder: RMSNorm + attention + SwiGLU feed-forward.

    Differences from GPT-2 style: RMSNorm instead of LayerNorm, a gated
    SwiGLU feed-forward (``down(silu(gate(x)) * up(x))``) and no biases.
    Rotary embeddings / causal masking are omitted from this IR sketch.
    """
    graph = ComputationGraph(f"llama_style_{num_blocks}b")
    graph.meta["config"] = {"batch": batch, "seq": seq, "d_model": d_model}
    scale_factor = float(1.0 / np.sqrt(d_model))

    graph.add_operation("Input", "Input")
    node = "Input"
    for i in range(1, num_blocks + 1):
        p = f"Block{i}"
        graph.add_operation(f"{p}_RMSNorm1", "RMSNorm")
        graph.add_dependency(node, f"{p}_RMSNorm1")
        for role in ("Q", "K", "V"):
            graph.add_operation(f"{p}_{role}_Projection", "MatMul")
            graph.add_dependency(f"{p}_RMSNorm1", f"{p}_{role}_Projection")
        graph.add_operation(f"{p}_QK_Score", "MatMul", transpose_rhs=True)
        graph.add_dependency(f"{p}_Q_Projection", f"{p}_QK_Score")
        graph.add_dependency(f"{p}_K_Projection", f"{p}_QK_Score")
        graph.add_operation(f"{p}_Scale", "Scale", factor=scale_factor)
        graph.add_dependency(f"{p}_QK_Score", f"{p}_Scale")
        graph.add_operation(f"{p}_Softmax", "Softmax")
        graph.add_dependency(f"{p}_Scale", f"{p}_Softmax")
        graph.add_operation(f"{p}_Attention_Output", "MatMul")
        graph.add_dependency(f"{p}_Softmax", f"{p}_Attention_Output")
        graph.add_dependency(f"{p}_V_Projection", f"{p}_Attention_Output")
        graph.add_operation(f"{p}_O_Projection", "MatMul")
        graph.add_dependency(f"{p}_Attention_Output", f"{p}_O_Projection")
        graph.add_operation(f"{p}_Attn_Residual", "Add")
        graph.add_dependency(f"{p}_O_Projection", f"{p}_Attn_Residual")
        graph.add_dependency(node, f"{p}_Attn_Residual")

        graph.add_operation(f"{p}_RMSNorm2", "RMSNorm")
        graph.add_dependency(f"{p}_Attn_Residual", f"{p}_RMSNorm2")
        graph.add_operation(f"{p}_Gate_Proj", "MatMul")
        graph.add_operation(f"{p}_Up_Proj", "MatMul")
        graph.add_dependency(f"{p}_RMSNorm2", f"{p}_Gate_Proj")
        graph.add_dependency(f"{p}_RMSNorm2", f"{p}_Up_Proj")
        graph.add_operation(f"{p}_SiLU", "SiLU")
        graph.add_dependency(f"{p}_Gate_Proj", f"{p}_SiLU")
        graph.add_operation(f"{p}_SwiGLU_Mul", "Mul")
        graph.add_dependency(f"{p}_SiLU", f"{p}_SwiGLU_Mul")
        graph.add_dependency(f"{p}_Up_Proj", f"{p}_SwiGLU_Mul")
        graph.add_operation(f"{p}_Down_Proj", "MatMul")
        graph.add_dependency(f"{p}_SwiGLU_Mul", f"{p}_Down_Proj")
        graph.add_operation(f"{p}_FFN_Residual", "Add")
        graph.add_dependency(f"{p}_Down_Proj", f"{p}_FFN_Residual")
        graph.add_dependency(f"{p}_Attn_Residual", f"{p}_FFN_Residual")
        node = f"{p}_FFN_Residual"

    graph.add_operation("Final_RMSNorm", "RMSNorm")
    graph.add_dependency(node, "Final_RMSNorm")
    graph.add_operation("Logits", "MatMul", is_output=True)
    graph.add_dependency("Final_RMSNorm", "Logits")
    return graph


def create_moe_graph(num_blocks=2, batch=4, seq=32, d_model=64, experts=4):
    """Dense mixture-of-experts Transformer (Mixtral-like layout).

    Each block replaces the single feed-forward network with ``experts``
    parallel expert FFNs whose outputs are averaged. Routing/top-k gating
    is omitted from this IR sketch (all experts run, equal weights), but
    the *wide parallel structure* is what matters to the optimizer.
    """
    graph = ComputationGraph(f"moe_{num_blocks}b_{experts}e")
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
        graph.add_operation(f"{p}_QK_Score", "MatMul", transpose_rhs=True)
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
        graph.add_operation(f"{p}_Experts_Sum", "Add")
        for e in range(experts):
            fc1, act, fc2 = (f"{p}_E{e}_FC1", f"{p}_E{e}_GELU", f"{p}_E{e}_FC2")
            graph.add_operation(fc1, "MatMul")
            graph.add_operation(act, "GELU")
            graph.add_operation(fc2, "MatMul")
            graph.add_dependency(f"{p}_LN2", fc1)
            graph.add_dependency(fc1, act)
            graph.add_dependency(act, fc2)
            graph.add_dependency(fc2, f"{p}_Experts_Sum")
        graph.add_operation(f"{p}_Experts_Mean", "Scale", factor=1.0 / experts)
        graph.add_dependency(f"{p}_Experts_Sum", f"{p}_Experts_Mean")
        graph.add_operation(f"{p}_FFN_Residual", "Add")
        graph.add_dependency(f"{p}_Experts_Mean", f"{p}_FFN_Residual")
        graph.add_dependency(f"{p}_Attn_Residual", f"{p}_FFN_Residual")
        node = f"{p}_FFN_Residual"

    graph.add_operation("Final_LayerNorm", "LayerNorm")
    graph.add_dependency(node, "Final_LayerNorm")
    graph.add_operation("Logits", "MatMul", is_output=True)
    graph.add_dependency("Final_LayerNorm", "Logits")
    return graph


def create_multihead_graph(num_blocks=2, batch=4, seq=32, d_model=64, heads=4):
    """Pre-LN Transformer with ``heads`` explicit parallel attention heads.

    Every head has its own Q/K/V projections and attention chain; head
    outputs are summed (a stand-in for concat + output projection).
    """
    graph = ComputationGraph(f"multihead_{num_blocks}b_{heads}h")
    graph.meta["config"] = {"batch": batch, "seq": seq, "d_model": d_model}
    scale_factor = float(1.0 / np.sqrt(d_model))
    graph.add_operation("Input", "Input")
    node = "Input"
    for i in range(1, num_blocks + 1):
        p = f"Block{i}"
        graph.add_operation(f"{p}_LN1", "LayerNorm")
        graph.add_dependency(node, f"{p}_LN1")
        graph.add_operation(f"{p}_Heads_Sum", "Add")
        for h in range(heads):
            q, k, v, sc, ss, sm, av = (f"{p}_H{h}_{x}" for x in
                                       ("Q", "K", "V", "QK", "Scale", "Softmax", "AV"))
            for name in (q, k, v):
                graph.add_operation(name, "MatMul")
                graph.add_dependency(f"{p}_LN1", name)
            graph.add_operation(sc, "MatMul", transpose_rhs=True)
            graph.add_dependency(q, sc)
            graph.add_dependency(k, sc)
            graph.add_operation(ss, "Scale", factor=scale_factor)
            graph.add_dependency(sc, ss)
            graph.add_operation(sm, "Softmax")
            graph.add_dependency(ss, sm)
            graph.add_operation(av, "MatMul")
            graph.add_dependency(sm, av)
            graph.add_dependency(v, av)
            graph.add_dependency(av, f"{p}_Heads_Sum")
        graph.add_operation(f"{p}_Attn_Residual", "Add")
        graph.add_dependency(f"{p}_Heads_Sum", f"{p}_Attn_Residual")
        graph.add_dependency(node, f"{p}_Attn_Residual")
        graph.add_operation(f"{p}_LN2", "LayerNorm")
        graph.add_dependency(f"{p}_Attn_Residual", f"{p}_LN2")
        graph.add_operation(f"{p}_FFN_Linear", "MatMul")
        graph.add_dependency(f"{p}_LN2", f"{p}_FFN_Linear")
        graph.add_operation(f"{p}_GELU", "GELU")
        graph.add_dependency(f"{p}_FFN_Linear", f"{p}_GELU")
        graph.add_operation(f"{p}_FFN_Output", "MatMul")
        graph.add_dependency(f"{p}_GELU", f"{p}_FFN_Output")
        graph.add_operation(f"{p}_FFN_Residual", "Add")
        graph.add_dependency(f"{p}_FFN_Output", f"{p}_FFN_Residual")
        graph.add_dependency(f"{p}_Attn_Residual", f"{p}_FFN_Residual")
        node = f"{p}_FFN_Residual"
    graph.add_operation("Final_LayerNorm", "LayerNorm")
    graph.add_dependency(node, "Final_LayerNorm")
    graph.add_operation("Logits", "MatMul", is_output=True)
    graph.add_dependency("Final_LayerNorm", "Logits")
    return graph


def create_parallel_block_graph(num_blocks=2, batch=4, seq=32, d_model=64):
    """GPT-J / PaLM-style block: attention and FFN run IN PARALLEL on the same
    LayerNorm output and are summed with the residual (3-input Add).

    Deliberately NOT part of ``ARCHITECTURES``: it is the held-out "never
    tuned on" architecture used by ``benchmarks/unseen_architecture_test.py``.
    """
    graph = ComputationGraph(f"parallel_block_{num_blocks}b")
    graph.meta["config"] = {"batch": batch, "seq": seq, "d_model": d_model}
    scale_factor = float(1.0 / np.sqrt(d_model))
    graph.add_operation("Input", "Input")
    node = "Input"
    for i in range(1, num_blocks + 1):
        p = f"Block{i}"
        graph.add_operation(f"{p}_LN", "LayerNorm")
        graph.add_dependency(node, f"{p}_LN")
        for role in ("Q", "K", "V"):
            graph.add_operation(f"{p}_{role}", "MatMul")
            graph.add_dependency(f"{p}_LN", f"{p}_{role}")
        graph.add_operation(f"{p}_QK", "MatMul", transpose_rhs=True)
        graph.add_dependency(f"{p}_Q", f"{p}_QK")
        graph.add_dependency(f"{p}_K", f"{p}_QK")
        graph.add_operation(f"{p}_Scale", "Scale", factor=scale_factor)
        graph.add_dependency(f"{p}_QK", f"{p}_Scale")
        graph.add_operation(f"{p}_Softmax", "Softmax")
        graph.add_dependency(f"{p}_Scale", f"{p}_Softmax")
        graph.add_operation(f"{p}_AV", "MatMul")
        graph.add_dependency(f"{p}_Softmax", f"{p}_AV")
        graph.add_dependency(f"{p}_V", f"{p}_AV")
        graph.add_operation(f"{p}_FFN1", "MatMul")
        graph.add_dependency(f"{p}_LN", f"{p}_FFN1")
        graph.add_operation(f"{p}_GELU", "GELU")
        graph.add_dependency(f"{p}_FFN1", f"{p}_GELU")
        graph.add_operation(f"{p}_FFN2", "MatMul")
        graph.add_dependency(f"{p}_GELU", f"{p}_FFN2")
        graph.add_operation(f"{p}_Sum", "Add")
        graph.add_dependency(f"{p}_AV", f"{p}_Sum")
        graph.add_dependency(f"{p}_FFN2", f"{p}_Sum")
        graph.add_dependency(node, f"{p}_Sum")
        node = f"{p}_Sum"
    graph.add_operation("Final_LN", "LayerNorm")
    graph.add_dependency(node, "Final_LN")
    graph.add_operation("Logits", "MatMul", is_output=True)
    graph.add_dependency("Final_LN", "Logits")
    return graph


UNSEEN_ARCHITECTURES = {"Parallel-block (GPT-J style)": create_parallel_block_graph}

ARCHITECTURES = {
    "Post-LN Transformer": create_postln_transformer_graph,
    "Pre-LN Transformer (GPT-style)": create_preln_transformer_graph,
    "Post-LN + ReLU FFN": create_relu_transformer_graph,
    "LLaMA-style (RMSNorm + SwiGLU)": create_llama_style_graph,
    "MoE Transformer (parallel experts)": create_moe_graph,
    "Multi-head Transformer (parallel heads)": create_multihead_graph,
    "MLP classifier": create_mlp_classifier_graph,
}
