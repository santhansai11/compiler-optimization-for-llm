import torch
import torch.fx as fx

from ir.operators import fused_attention


def is_fx_node(node):
    return isinstance(node, fx.Node)


def is_call_function(node, name):
    return (
        is_fx_node(node)
        and node.op == "call_function"
        and getattr(node.target, "__name__", "") == name
    )


def is_call_method(node, name):
    return (
        is_fx_node(node)
        and node.op == "call_method"
        and node.target == name
    )


def is_matmul(node):
    return is_call_function(node, "matmul")


def find_attention_pattern(graph_module):

    nodes = list(graph_module.graph.nodes)

    for output_node in nodes:

        if not is_matmul(output_node):
            continue

        if len(output_node.args) != 2:
            continue

        softmax_node = output_node.args[0]
        v_node = output_node.args[1]

        if not is_call_function(
            softmax_node,
            "softmax"
        ):
            continue

        if not is_matmul(v_node):
            continue

        x_v, wv = v_node.args

        scale_node = softmax_node.args[0]

        if not is_call_function(
            scale_node,
            "truediv"
        ):
            continue

        scores_node, scale_value = scale_node.args

        if not is_matmul(scores_node):
            continue

        q_node, kt_node = scores_node.args

        if not is_call_method(
            kt_node,
            "transpose"
        ):
            continue

        k_node = kt_node.args[0]

        if not is_matmul(q_node):
            continue

        if not is_matmul(k_node):
            continue

        x_q, wq = q_node.args
        x_k, wk = k_node.args

        # Relationship check:
        # Q, K and V all originate from X.
        if x_q is not x_k:
            continue

        if x_q is not x_v:
            continue

        return {
            "x": x_q,
            "wq": wq,
            "wk": wk,
            "wv": wv,
            "scale": scale_value,

            "q": q_node,
            "k": k_node,
            "v": v_node,
            "kt": kt_node,
            "scores": scores_node,
            "scale_node": scale_node,
            "softmax": softmax_node,
            "output": output_node,
        }

    return None


def semantic_merge_attention(graph_module):

    pattern = find_attention_pattern(graph_module)

    if pattern is None:
        return False

    graph = graph_module.graph

    # All inputs required by fused_attention must already exist.
    # Insert after the latest dependency.
    dependencies = [
        pattern["x"],
        pattern["wq"],
        pattern["wk"],
        pattern["wv"],
        pattern["scale"],
    ]

    # Find the latest dependency in topological order.
    nodes = list(graph.nodes)

    last_dependency = max(
        dependencies,
        key=lambda node: nodes.index(node)
    )

    with graph.inserting_after(last_dependency):

        fused = graph.call_function(
            fused_attention,
            args=(
                pattern["x"],
                pattern["wq"],
                pattern["wk"],
                pattern["wv"],
                pattern["scale"],
            ),
        )

    # Replace final attention result with fused result.
    pattern["output"].replace_all_uses_with(fused)

    # Remove the old attention computation.
    removable = [
        pattern["output"],
        pattern["softmax"],
        pattern["scale_node"],
        pattern["scores"],
        pattern["kt"],
        pattern["q"],
        pattern["k"],
        pattern["v"],
    ]

    for node in removable:
        if len(node.users) == 0:
            graph.erase_node(node)

    graph.eliminate_dead_code()

    graph.lint()
    graph_module.recompile()

    return True
