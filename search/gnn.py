"""Graph Neural Network scorer for computation graphs (NumPy).

A simplified spectral-style GNN (SGC flavour): node features are one-hot
op types plus structural/cost scalars, propagated over the normalized
adjacency for two rounds, mean-pooled into a graph embedding, and scored
by a ridge-regression head trained to predict the relative speedup the
optimizer can achieve on a graph. Weights are trained on a synthetic DAG
corpus (labels from the cost model) and cached in ``gnn_weights.npz``.
"""

import os

import numpy as np

from ir.graph import ComputationGraph
from utils.cost_model import get_op_latency

OP_VOCAB = [
    "Input", "Constant", "Identity", "MatMul", "GEMM", "LinearGELU",
    "LinearRelu", "FusedAttention", "FusedAddLayerNorm",
    "FusedScaleSoftmax", "Scale", "Softmax", "Add", "Mul", "Sub",
    "LayerNorm", "GELU", "Relu", "Operation",
]

FEATURE_EXTRA = 4
HIDDEN_DIM = 16
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "gnn_weights.npz")


def node_features(graph):
    """Per-node feature matrix: one-hot op type + structural scalars."""
    nodes = list(graph.graph.nodes())
    index = {node: i for i, node in enumerate(nodes)}
    width = len(OP_VOCAB) + FEATURE_EXTRA
    features = np.zeros((len(nodes), width), dtype=np.float64)
    for node in nodes:
        row = features[index[node]]
        op_type = graph.op_type(node)
        vocab_index = OP_VOCAB.index(op_type if op_type in OP_VOCAB
                                     else "Operation")
        row[vocab_index] = 1.0
        row[len(OP_VOCAB) + 0] = np.log1p(graph.graph.in_degree(node)) / 4.0
        row[len(OP_VOCAB) + 1] = np.log1p(graph.graph.out_degree(node)) / 4.0
        row[len(OP_VOCAB) + 2] = 1.0 if op_type.startswith("Fused") else 0.0
        row[len(OP_VOCAB) + 3] = get_op_latency(op_type) / 1000.0
    return nodes, index, features


def normalized_adjacency(graph, index):
    """Symmetrically normalized adjacency with self-loops (D^-1/2 A D^-1/2)."""
    n = len(index)
    adj = np.eye(n)
    for u, v in graph.graph.edges():
        adj[index[u], index[v]] = 1.0
        adj[index[v], index[u]] = 1.0
    degrees = adj.sum(axis=1)
    inv_sqrt = 1.0 / np.sqrt(np.maximum(degrees, 1e-9))
    return adj * inv_sqrt[:, None] * inv_sqrt[None, :]


class GraphScorer:
    """Two-round message-passing encoder + learned ridge readout head."""

    def __init__(self):
        rng = np.random.default_rng(0)
        width = len(OP_VOCAB) + FEATURE_EXTRA
        self.W1 = rng.normal(0.0, 0.4, (width, HIDDEN_DIM))
        self.W2 = rng.normal(0.0, 0.4, (HIDDEN_DIM, HIDDEN_DIM))
        self.head = np.zeros(HIDDEN_DIM)
        self.bias = 0.0

    def embed(self, graph):
        nodes, index, features = node_features(graph)
        adj = normalized_adjacency(graph, index)
        hidden = np.maximum(features @ self.W1, 0.0)
        for _ in range(2):
            hidden = np.maximum(hidden + adj @ (hidden @ self.W2), 0.0)
        return hidden.mean(axis=0)

    def score_graph(self, graph):
        """Predicted relative speedup the optimizer can reach (higher = more
        optimizable)."""
        return float(self.embed(graph) @ self.head + self.bias)

    def fit(self, embeddings, labels, l2=1e-3):
        x = np.asarray(embeddings, dtype=np.float64)
        y = np.asarray(labels, dtype=np.float64)
        center = float(np.mean(y))
        gram = x.T @ x + l2 * np.eye(x.shape[1])
        try:
            self.head = np.linalg.solve(gram, x.T @ (y - center))
        except np.linalg.LinAlgError:
            self.head = np.linalg.lstsq(x, y - center, rcond=None)[0]
        self.bias = center
        return self

    def save(self, path=WEIGHTS_PATH):
        np.savez(path, W1=self.W1, W2=self.W2, head=self.head,
                 bias=np.float64(self.bias))

    @classmethod
    def load(cls, path=WEIGHTS_PATH):
        data = np.load(path)
        scorer = cls()
        scorer.W1 = data["W1"]
        scorer.W2 = data["W2"]
        scorer.head = data["head"]
        scorer.bias = float(data["bias"])
        return scorer


def load_or_train(path=WEIGHTS_PATH, corpus_size=36, seed=0):
    """Load cached weights, or train once and cache them."""
    if os.path.exists(path):
        try:
            return GraphScorer.load(path)
        except Exception:
            pass
    return train(path=path, corpus_size=corpus_size, seed=seed)


def train(path=WEIGHTS_PATH, corpus_size=36, seed=0):
    """Train the readout head on a structured synthetic DAG corpus.

    Labels come from the cost model (modeled speedup of the full fixed
    pipeline). Each corpus graph contributes TWO samples - the unfused
    input graph and its fused pipeline output, sharing the same label -
    so the feature space the head sees at search time (fused operators)
    matches the training distribution.
    """
    from pipeline import run_pipeline  # local import avoids a cycle
    from utils.cost_model import scheduled_latency_us, sequential_latency_us

    rng = np.random.default_rng(seed)
    scorer = GraphScorer()
    embeddings, labels = [], []

    def add_sample(sample_graph, label):
        embeddings.append(scorer.embed(sample_graph))
        labels.append(label)

    for i in range(corpus_size):
        graph = _random_dag(rng, f"synthetic_{i}")
        try:
            optimized, _, _ = run_pipeline(
                graph, normalize=True, canonicalize=True, merge=True,
                schedule=True, partition=False,
            )
            original_cost = sequential_latency_us(graph)
            optimized_cost = scheduled_latency_us(optimized)
            label = original_cost / optimized_cost if optimized_cost else 1.0
        except Exception:
            label = 1.0
        add_sample(graph, label)
        add_sample(optimized, label)

    scorer.fit(embeddings, labels)
    try:
        scorer.save(path)
    except Exception:
        pass
    return scorer


def _add_attention_motif(graph, rng, prefix, source):
    """Append a canonical attention sub-graph; returns its output node."""
    del rng  # motifs are fully determined; kept for signature symmetry
    q, k, v = (f"{prefix}_q", f"{prefix}_k", f"{prefix}_v")
    for op_name in (q, k, v):
        graph.add_operation(op_name, "MatMul")
        graph.add_dependency(source, op_name)
    qk = f"{prefix}_qk"
    graph.add_operation(qk, "MatMul")
    graph.add_dependency(q, qk)
    graph.add_dependency(k, qk)
    scale = f"{prefix}_scale"
    graph.add_operation(scale, "Scale", factor=0.125)
    graph.add_dependency(qk, scale)
    softmax = f"{prefix}_softmax"
    graph.add_operation(softmax, "Softmax")
    graph.add_dependency(scale, softmax)
    av = f"{prefix}_av"
    graph.add_operation(av, "MatMul")
    graph.add_dependency(softmax, av)
    graph.add_dependency(v, av)
    return av


def _random_dag(rng, name):
    """Random DAG mixing attention motifs and random base ops."""
    graph = ComputationGraph(name)
    graph.meta["config"] = {"batch": 4, "seq": 32, "d_model": 64}
    vocab = ["MatMul", "Add", "Scale", "Softmax", "GELU", "LayerNorm",
             "MatMul", "Add"]
    graph.add_operation("Input", "Input")
    current = "Input"
    nodes = ["Input"]

    for b in range(int(rng.integers(0, 3))):
        prefix = f"{name}_m{b}"
        current = _add_attention_motif(graph, rng, prefix, current)
        nodes.extend([f"{prefix}_{part}"
                      for part in ("q", "k", "v", "qk", "scale",
                                   "softmax", "av")])

    for i in range(int(rng.integers(2, 9))):
        op_type = str(vocab[int(rng.integers(0, len(vocab)))])
        node_name = f"{name}_r{i}_{op_type}"
        graph.add_operation(node_name, op_type)
        graph.add_dependency(current, node_name)
        if len(nodes) > 1 and rng.random() < 0.5:
            extra = str(rng.choice(nodes))
            if extra != current:
                graph.add_dependency(extra, node_name)
        nodes.append(node_name)
        current = node_name
    return graph
