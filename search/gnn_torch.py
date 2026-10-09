"""PyTorch message-passing scorer trained to predict graph optimizability."""

from pathlib import Path

import torch
from torch import nn

from search.gnn import OP_VOCAB, _random_dag

FEATURES = len(OP_VOCAB) + 4
WEIGHTS_PATH = Path(__file__).with_name("gnn_torch_weights.pt")


def graph_tensors(graph):
    """Convert a ComputationGraph DAG to node features and directed edges."""
    nodes = list(graph.graph.nodes())
    index = {node: i for i, node in enumerate(nodes)}
    x = torch.zeros((len(nodes), FEATURES), dtype=torch.float32)
    edges = []
    for node in nodes:
        row = x[index[node]]
        op_type = graph.op_type(node)
        vocab = op_type if op_type in OP_VOCAB else "Operation"
        row[OP_VOCAB.index(vocab)] = 1.0
        row[len(OP_VOCAB)] = min(graph.graph.in_degree(node), 16) / 16
        row[len(OP_VOCAB) + 1] = min(graph.graph.out_degree(node), 16) / 16
        row[len(OP_VOCAB) + 2] = float(op_type.startswith("Fused"))
        row[len(OP_VOCAB) + 3] = min(
            float(graph.graph.nodes[node].get("estimated_latency_us", 10.0)),
            1000.0,
        ) / 1000.0
    for source, target in graph.graph.edges():
        edges.append((index[source], index[target]))
    edge_index = (torch.tensor(edges, dtype=torch.long).t().contiguous()
                  if edges else torch.empty((2, 0), dtype=torch.long))
    return x, edge_index


class GraphRegressor(nn.Module):
    """Two rounds of directed message passing and pooled regression."""

    def __init__(self, hidden=32):
        super().__init__()
        self.input = nn.Linear(FEATURES, hidden)
        self.message = nn.ModuleList(nn.Linear(hidden * 2, hidden)
                                     for _ in range(2))
        self.head = nn.Sequential(nn.Linear(hidden * 2 + 3, hidden), nn.ReLU(),
                                  nn.Linear(hidden, 1))

    def forward(self, x, edge_index):
        h = torch.relu(self.input(x))
        src, dst = edge_index
        for layer in self.message:
            aggregate = torch.zeros_like(h)
            degree = torch.zeros((h.shape[0], 1), dtype=h.dtype,
                                 device=h.device)
            if src.numel():
                aggregate.index_add_(0, dst, h[src])
                degree.index_add_(0, dst, torch.ones_like(degree[dst]))
            aggregate = aggregate / degree.clamp_min(1.0)
            h = torch.relu(layer(torch.cat((h, aggregate), dim=-1)))
        pooled = torch.cat((h.mean(0), h.max(0).values))
        structural = x[:, -3:-1].mean(0).tolist()
        # Include log graph size as a stable global signal.
        global_features = torch.cat((pooled, x.new_tensor([
            torch.log1p(x.new_tensor(float(x.shape[0]))).item(),
            *structural,
        ])))
        return self.head(global_features).squeeze(-1)


class TorchGraphScorer:
    """Adapter compatible with neuro-symbolic candidate scoring."""

    def __init__(self, model=None):
        self.model = model or GraphRegressor()
        self.model.eval()

    def score_graph(self, graph):
        x, edges = graph_tensors(graph)
        with torch.no_grad():
            return float(self.model(x, edges).item())

    def save(self, path=WEIGHTS_PATH):
        torch.save(self.model.state_dict(), path)

    @classmethod
    def load(cls, path=WEIGHTS_PATH):
        model = GraphRegressor()
        model.load_state_dict(torch.load(path, map_location="cpu",
                                         weights_only=True))
        return cls(model)


def train(path=WEIGHTS_PATH, corpus_size=96, seed=0, epochs=160):
    """Train graph scorer using optimizer speedup labels from the cost model."""
    from pipeline import run_pipeline
    from utils.cost_model import scheduled_latency_us, sequential_latency_us

    torch.manual_seed(seed)
    rng = __import__("numpy").random.default_rng(seed)
    samples, labels = [], []
    for i in range(corpus_size):
        graph = _random_dag(rng, f"torch_train_{i}")
        try:
            optimized, _, _ = run_pipeline(
                graph, normalize=True, canonicalize=True, merge=True,
                schedule=True, partition=False,
            )
            original_cost = sequential_latency_us(graph)
            optimized_cost = scheduled_latency_us(optimized)
            label = original_cost / optimized_cost if optimized_cost else 1.0
            samples.extend((graph, optimized))
            labels.extend((label, label))
        except Exception:
            samples.append(graph)
            labels.append(1.0)

    model = GraphRegressor()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3,
                                  weight_decay=1e-4)
    targets = torch.tensor(labels, dtype=torch.float32)
    # Fixed held-out tail makes the training diagnostic reproducible.
    split = max(1, int(len(samples) * 0.8))
    tensors = [graph_tensors(graph) for graph in samples]
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        predictions = torch.stack([
            model(*tensors[index]) for index in range(split)
        ])
        loss = torch.nn.functional.smooth_l1_loss(predictions, targets[:split])
        loss.backward()
        optimizer.step()
    model.eval()
    scorer = TorchGraphScorer(model)
    scorer.save(path)
    with torch.no_grad():
        heldout = torch.stack([model(*tensors[i])
                               for i in range(split, len(samples))])
        val_loss = (torch.nn.functional.smooth_l1_loss(
            heldout, targets[split:]).item() if len(samples) > split else None)
    scorer.training_info = {
        "samples": len(samples), "epochs": epochs,
        "train_loss": float(loss.item()), "validation_loss": val_loss,
        "label_source": "modeled optimizer speedup",
    }
    return scorer


def load_or_train(path=WEIGHTS_PATH, corpus_size=96, seed=0):
    try:
        if Path(path).exists():
            return TorchGraphScorer.load(path)
    except (OSError, RuntimeError, ValueError):
        pass
    return train(path=path, corpus_size=corpus_size, seed=seed)
