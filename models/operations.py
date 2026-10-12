"""Small, easy-to-explain operation workloads ("operation lab").

Each builder emits a tiny computation graph that isolates ONE optimisation
idea, so the effect of a pass can be read directly off the DAG:

* ``transpose_demo``  - matrix transpose algebra: (A^T)^T -> A and
  A @ B^T -> a single transposed-GEMM (no materialised transpose).
* ``scale_chain_demo``- chained element-wise scalings collapse to one.
* ``mlp_block_demo``  - MatMul + Add + GELU -> one fused LinearGELU kernel.
* ``attention_demo``  - one attention head: the full QK^T-Scale-Softmax-AV
  sandwich becomes a single FusedAttention node.

All inputs are random float32 tensors of shape (batch, seq, d_model), so any
dataset can stand in for them: the optimiser only sees shapes and ops.
"""

import numpy as np

from ir.graph import ComputationGraph


def _config(graph, batch, seq, d_model):
    graph.meta["config"] = {"batch": batch, "seq": seq, "d_model": d_model}


def transpose_demo(batch=4, seq=32, d_model=64):
    """Scores = Q @ K^T, with a redundant (Q^T)^T emitted by the framework."""
    g = ComputationGraph("transpose_demo")
    _config(g, batch, seq, d_model)
    g.add_operation("Input", "Input")
    g.add_operation("Q", "MatMul")
    g.add_operation("K", "MatMul")
    g.add_dependency("Input", "Q")
    g.add_dependency("Input", "K")

    # A redundant double transpose (A^T)^T that a framework might emit.
    g.add_operation("Q_T", "Transpose", dims=(0, 2, 1))
    g.add_operation("Q_TT", "Transpose", dims=(0, 2, 1))
    g.add_dependency("Q", "Q_T")
    g.add_dependency("Q_T", "Q_TT")

    # K^T feeding a matmul: the classic Q @ K^T of attention scores.
    g.add_operation("K_T", "Transpose", dims=(0, 2, 1))
    g.add_dependency("K", "K_T")
    g.add_operation("Scores", "MatMul")
    g.add_dependency("Q_TT", "Scores")
    g.add_dependency("K_T", "Scores")

    g.add_operation("Output", "Scale", factor=float(1.0 / np.sqrt(d_model)),
                    is_output=True)
    g.add_dependency("Scores", "Output")
    return g


def scale_chain_demo(batch=4, seq=32, d_model=64):
    """x * a * b * c with redundant identities and a duplicated branch."""
    g = ComputationGraph("scale_chain_demo")
    _config(g, batch, seq, d_model)
    g.add_operation("Input", "Input")
    g.add_operation("Scale_a", "Scale", factor=0.5)
    g.add_operation("Id1", "Identity")
    g.add_operation("Scale_b", "Scale", factor=0.25)
    g.add_operation("Scale_c", "Scale", factor=8.0)
    g.add_dependency("Input", "Scale_a")
    g.add_dependency("Scale_a", "Id1")
    g.add_dependency("Id1", "Scale_b")
    g.add_dependency("Scale_b", "Scale_c")

    # Two identical Softmax branches (common sub-expression), each used
    # by a different consumer so they can be shared without changing math.
    g.add_operation("Branch1", "Softmax")
    g.add_operation("Branch2", "Softmax")
    g.add_dependency("Scale_c", "Branch1")
    g.add_dependency("Scale_c", "Branch2")
    g.add_operation("Out1", "Scale", factor=2.0)
    g.add_operation("Out2", "Scale", factor=3.0)
    g.add_dependency("Branch1", "Out1")
    g.add_dependency("Branch2", "Out2")
    g.add_operation("Output", "Add", is_output=True)
    g.add_dependency("Out1", "Output")
    g.add_dependency("Out2", "Output")
    return g


def mlp_block_demo(batch=4, seq=32, d_model=64):
    """Feed-forward block: linear -> bias -> GELU -> linear -> bias."""
    g = ComputationGraph("mlp_block_demo")
    _config(g, batch, seq, d_model)
    rng = np.random.default_rng(3)
    g.add_operation("Input", "Input")
    prev = "Input"
    for i in (1, 2):
        g.add_operation(f"Linear{i}", "MatMul")
        g.add_dependency(prev, f"Linear{i}")
        bias = rng.normal(0, 0.02, d_model).astype(np.float32)
        g.add_operation(f"Bias{i}_const", "Constant", value=bias,
                        shape=bias.shape)
        g.add_operation(f"Bias{i}", "Add")
        g.add_dependency(f"Linear{i}", f"Bias{i}")
        g.add_dependency(f"Bias{i}_const", f"Bias{i}")
        prev = f"Bias{i}"
        if i == 1:
            g.add_operation("Act", "GELU")
            g.add_dependency(prev, "Act")
            prev = "Act"
    g.graph.nodes[prev]["is_output"] = True
    return g


def attention_demo(batch=4, seq=32, d_model=64):
    """A single attention head (the unit FusedAttention replaces)."""
    g = ComputationGraph("attention_demo")
    _config(g, batch, seq, d_model)
    g.add_operation("Input", "Input")
    for role in ("Q", "K", "V"):
        g.add_operation(f"{role}_Projection", "MatMul")
        g.add_dependency("Input", f"{role}_Projection")
    g.add_operation("QK_Score", "MatMul", transpose_rhs=True)
    g.add_dependency("Q_Projection", "QK_Score")
    g.add_dependency("K_Projection", "QK_Score")
    g.add_operation("Scale", "Scale", factor=float(1.0 / np.sqrt(d_model)))
    g.add_dependency("QK_Score", "Scale")
    g.add_operation("Softmax", "Softmax")
    g.add_dependency("Scale", "Softmax")
    g.add_operation("Attention_Output", "MatMul", is_output=True)
    g.add_dependency("Softmax", "Attention_Output")
    g.add_dependency("V_Projection", "Attention_Output")
    return g


OPERATION_DEMOS = {
    "op_transpose": ("Op lab - Transpose algebra", transpose_demo,
                     "(Aᵀ)ᵀ→A and A·Bᵀ folded into one transposed GEMM"),
    "op_scale_chain": ("Op lab - Scale chain + CSE", scale_chain_demo,
                       "Chained scalings collapse, duplicate branches merge"),
    "op_mlp_block": ("Op lab - MLP block", mlp_block_demo,
                     "MatMul+Bias+GELU become one LinearGELU kernel"),
    "op_attention": ("Op lab - Single attention head", attention_demo,
                     "Q/K/V·QKᵀ·Scale·Softmax·AV become FusedAttention"),
}
