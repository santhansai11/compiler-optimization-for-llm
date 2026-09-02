"""Export a trained model's weights into the computation-graph IR.

The forward pass is emitted the way a real framework export (ONNX /
Torch FX) would look: separate MatMul and bias-Add nodes with the
trained weights stored as IR attributes — exactly the structure the
compiler's fusion passes are designed to optimize.
"""

from ir.graph import ComputationGraph


def create_mlp_classifier_graph(sizes=(784, 256, 64, 10), params=None,
                                act="relu", name="mlp_classifier",
                                batch=256):
    """Build an MLP-classifier graph; ``params`` carries trained weights.

    ``params`` is the list of [W, b] pairs from ``training.model.MLP``;
    when ``None``, deterministic seeded weights are generated so the
    architecture can be used for structure-only comparisons too.
    """
    import numpy as np

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


def export_mlp_graph(params, act="relu", name="trained_mlp", batch=256):
    """Export trained parameters as a compilable computation graph."""
    sizes = [params[0][0].shape[0]] + [W.shape[1] for W, _ in params]
    return create_mlp_classifier_graph(sizes=tuple(sizes), params=params,
                                       act=act, name=name, batch=batch)
