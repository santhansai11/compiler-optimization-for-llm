import torch.fx as fx


def extract_graph(model):
    model.eval()
    return fx.symbolic_trace(model)


def node_label(node):
    if node.op == "placeholder":
        return "INPUT"

    if node.op == "output":
        return "OUTPUT"

    if node.op == "call_module":
        return f"{node.name}\n{node.target}"

    if node.op == "call_method":
        return f"{node.name}\n{node.target}"

    if node.op == "call_function":
        target = getattr(node.target, "__name__", str(node.target))
        return f"{node.name}\n{target}"

    if node.op == "get_attr":
        return f"{node.name}\n{node.target}"

    return node.name


def print_graph(graph_module, title):
    print(f"\n===== {title} =====")

    for i, node in enumerate(graph_module.graph.nodes):

        target = node.target

        if callable(target):
            target = getattr(
                target,
                "__name__",
                str(target)
            )

        print(
            f"{i:02d} | "
            f"{node.op:15} | "
            f"{node.name:20} | "
            f"{target}"
        )


def count_compute_nodes(graph_module):

    ignored = {
        "placeholder",
        "output",
        "get_attr",
    }

    return sum(
        node.op not in ignored
        for node in graph_module.graph.nodes
    )