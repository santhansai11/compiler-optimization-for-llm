import matplotlib.pyplot as plt
import networkx as nx


IGNORE_OPS = {
    "get_attr",
    "placeholder",
    "output",
}


def fx_to_networkx(graph_module):
    graph = nx.DiGraph()

    fx_graph = graph_module.graph

    for node in fx_graph.nodes:

        if node.op in IGNORE_OPS:
            continue

        graph.add_node(
            node.name,
            label=node.name
        )

    for node in fx_graph.nodes:

        if node.op in IGNORE_OPS:
            continue

        for user in node.users:

            if user.op in IGNORE_OPS:
                continue

            graph.add_edge(
                node.name,
                user.name
            )

    return graph


def draw_graph(
    graph_module,
    title,
    save_path
):
    graph = fx_to_networkx(graph_module)

    positions = nx.spring_layout(
        graph,
        seed=42,
        k=2.0
    )

    labels = nx.get_node_attributes(
        graph,
        "label"
    )

    plt.figure(figsize=(14, 8))

    nx.draw(
        graph,
        positions,
        labels=labels,
        with_labels=True,
        node_size=2500,
        node_color="lightblue",
        arrows=True,
        font_size=8
    )

    plt.title(title)

    plt.savefig(
        save_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()