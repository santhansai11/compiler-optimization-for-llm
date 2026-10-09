import matplotlib.pyplot as plt
import networkx as nx


def make_graph(graph_module):

    g = nx.DiGraph()

    for node in graph_module.graph.nodes:

        if node.op == "placeholder":
            label = "INPUT"

        elif node.op == "output":
            label = "OUTPUT"

        elif node.op == "get_attr":
            label = node.name

        elif node.op == "call_method":
            label = node.target

        elif node.op == "call_function":
            label = getattr(
                node.target,
                "__name__",
                str(node.target)
            )

        else:
            label = node.name

        g.add_node(
            node.name,
            label=label
        )

    for node in graph_module.graph.nodes:

        for user in node.users:

            g.add_edge(
                node.name,
                user.name
            )

    return g


def draw_graph(
    graph_module,
    title,
    filename,
):
    g = make_graph(graph_module)

    position = nx.spring_layout(
        g,
        seed=42,
        k=2.0
    )

    labels = nx.get_node_attributes(
        g,
        "label"
    )

    plt.figure(figsize=(14, 8))

    nx.draw(
        g,
        position,
        labels=labels,
        with_labels=True,
        node_size=2800,
        arrows=True,
        font_size=8,
    )

    plt.title(title)

    plt.savefig(
        filename,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()
