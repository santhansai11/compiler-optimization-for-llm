"""Computation graph intermediate representation for the LLM compiler."""

import networkx as nx


class ComputationGraph:
    """A directed acyclic graph of tensor operations.

    Nodes represent operators and carry an ``op_type`` plus arbitrary
    attributes (``value`` for constants, ``factor`` for scaling ops,
    ``fused_from`` provenance, ``schedule_order`` / ``schedule_level``
    from DAGS, ``partition`` from partitioning, ``shape`` / ``dtype``
    from normalization). Edges represent tensor dependencies.
    """

    def __init__(self, name="computation_graph"):
        self.name = name
        self.graph = nx.DiGraph()
        self.meta = {}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def add_operation(self, name, op_type, **attrs):
        """Add an operator node and return its name."""
        if name in self.graph:
            raise ValueError(f"Duplicate operation name: {name!r}")
        self.graph.add_node(name, op_type=op_type, **attrs)
        return name

    def add_dependency(self, source, target, **attrs):
        """Add a tensor dependency between two operations."""
        if source not in self.graph:
            raise KeyError(f"Unknown operation: {source!r}")
        if target not in self.graph:
            raise KeyError(f"Unknown operation: {target!r}")
        self.graph.add_edge(source, target, **attrs)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def has_operation(self, name):
        return name in self.graph

    def op_type(self, name):
        return self.graph.nodes[name].get("op_type", "Operation")

    def predecessors(self, name):
        return list(self.graph.predecessors(name))

    def successors(self, name):
        return list(self.graph.successors(name))

    def nodes(self, data=False):
        return self.graph.nodes(data=data)

    def edges(self, data=False):
        return self.graph.edges(data=data)

    def node_count(self):
        return self.graph.number_of_nodes()

    def edge_count(self):
        return self.graph.number_of_edges()

    def topological_order(self):
        return list(nx.topological_sort(self.graph))

    def is_dag(self):
        return nx.is_directed_acyclic_graph(self.graph)

    def sink_nodes(self):
        return [node for node, degree in self.graph.out_degree() if degree == 0]

    def op_type_histogram(self):
        counts = {}
        for _, data in self.graph.nodes(data=True):
            op_type = data.get("op_type", "Operation")
            counts[op_type] = counts.get(op_type, 0) + 1
        return counts

    # ------------------------------------------------------------------
    # Transforms
    # ------------------------------------------------------------------
    def copy(self):
        """Return a deep-enough clone (graph structure + attrs + meta)."""
        clone = ComputationGraph(self.name)
        clone.graph = nx.DiGraph(self.graph)
        clone.meta = dict(self.meta)
        return clone
