"""Learned optimizer: a GNN that chooses the compiler plan for a graph.

The rewrite rules themselves stay symbolic (so every rewrite is provably
output-preserving). What is *learned* is the decision of **which passes to
run and across how many devices to place the graph**:

    action = (normalize?, canonicalize?, merge?) x (devices k = 1..4)    # 32

    reward = deployment speedup over the unoptimized single-device graph
             - DEVICE_PENALTY * (devices used - 1)     # extra hardware costs
             - PASS_PENALTY   * (#passes run)          # tie-break, compile risk

Rewards are *modeled* with ``utils.cost_model.deployment_latency_us`` (a
hand-written cost table plus an assumed transfer cost), not measured on a
GPU. Training set: randomly generated graphs from 7 architecture families
with injected redundancy, each labelled by actually running all 32 plans.
Evaluation: leave-one-family-out (train on 6 families, test on the 7th).

    python -m search.learned_optimizer train     # LOFO eval + final model
"""

import json
import math
import pickle
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import torch
from torch import nn

from utils.environment import collect as collect_environment

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_PATH = Path(__file__).with_name("learned_optimizer.pt")
CACHE_PATH = ROOT / "training" / "artifacts" / "learned_dataset.pkl"
RESULTS_PATH = ROOT / "docs" / "results" / "learned_results.json"

DEVICE_PENALTY = 0.15
PASS_PENALTY = 0.005
MAX_DEVICES = 4
PASS_SUBSETS = list(product([False, True], repeat=3))  # normalize, canon, merge
ACTIONS = [(*subset, k) for subset in PASS_SUBSETS
           for k in range(1, MAX_DEVICES + 1)]
FAMILIES = ["postln", "preln", "relu", "llama", "moe", "multihead", "mlp"]


def action_flags(action):
    n, c, m, k = action
    return {"normalize": n, "canonicalize": c, "merge": m,
            "schedule": True, "partition": k > 1}, k


def action_label(action):
    n, c, m, k = action
    names = [name for name, on in
             zip(("normalize", "canonicalize", "merge"), (n, c, m)) if on]
    return f"{'+'.join(names) or 'no passes'} · {k} device{'s' if k > 1 else ''}"


# ----------------------------------------------------------------------
# Dataset generation
# ----------------------------------------------------------------------
def _inject_redundancy(graph, rng):
    """Insert semantics-preserving junk: identities and Scale(a)->Scale(1/a).

    The junk is spliced into an existing edge WITHOUT changing the operand
    order of the consumer (operand roles are encoded by in-edge order).
    """
    from passes.normalize import _replace_input

    g = graph.graph
    for i in range(int(rng.integers(0, 9))):
        edges = [(u, v) for u, v in g.edges()
                 if g.nodes[u].get("op_type") not in ("Constant",)]
        if not edges:
            return graph
        u, v = edges[int(rng.integers(len(edges)))]
        if rng.random() < 0.5:
            graph.add_operation(f"inj{i}_id", "Identity")
            graph.add_dependency(u, f"inj{i}_id")
            last = f"inj{i}_id"
        else:
            a = float(rng.choice([0.5, 2.0, 4.0]))
            graph.add_operation(f"inj{i}_s1", "Scale", factor=a)
            graph.add_operation(f"inj{i}_s2", "Scale", factor=1.0 / a)
            graph.add_dependency(u, f"inj{i}_s1")
            graph.add_dependency(f"inj{i}_s1", f"inj{i}_s2")
            last = f"inj{i}_s2"
        _replace_input(g, v, u, last)
    return graph


def generate_graph(family, rng):
    from models.architectures import (
        create_llama_style_graph, create_moe_graph, create_mlp_classifier_graph,
        create_multihead_graph, create_postln_transformer_graph,
        create_preln_transformer_graph, create_relu_transformer_graph)

    blocks = int(rng.integers(1, 7))
    if family == "postln":
        g = create_postln_transformer_graph(blocks)
    elif family == "preln":
        g = create_preln_transformer_graph(blocks)
    elif family == "relu":
        g = create_relu_transformer_graph(blocks)
    elif family == "llama":
        g = create_llama_style_graph(blocks)
    elif family == "moe":
        g = create_moe_graph(int(rng.integers(1, 4)), experts=int(rng.integers(2, 9)))
    elif family == "multihead":
        g = create_multihead_graph(int(rng.integers(1, 4)), heads=int(rng.integers(2, 9)))
    elif family == "mlp":
        depth = int(rng.integers(2, 9))
        g = create_mlp_classifier_graph(sizes=tuple([64] * depth + [10]))
    else:
        raise ValueError(family)
    return _inject_redundancy(g, rng)


def evaluate_actions(graph):
    """True reward of every plan, by actually running the pipeline."""
    from pipeline import run_pipeline
    from utils.cost_model import deployment_latency_us

    base = deployment_latency_us(graph)
    rewards = []
    for action in ACTIONS:
        flags, k = action_flags(action)
        opt, _, _ = run_pipeline(graph, num_partitions=k,
                                 partition_mode="latency", **flags)
        devices = opt.meta.get("partition", {}).get("partitions", 1) if k > 1 else 1
        rewards.append(base / deployment_latency_us(opt)
                       - DEVICE_PENALTY * (devices - 1)
                       - PASS_PENALTY * sum(action[:3]))
    return rewards


def _constants():
    from utils import cost_model
    return (DEVICE_PENALTY, PASS_PENALTY, cost_model.COMM_US)


def build_dataset(per_family=100, seed=0, use_cache=True):
    if use_cache and CACHE_PATH.exists():
        data = pickle.loads(CACHE_PATH.read_bytes())
        from utils.cost_model import CALIBRATED
        if (data["per_family"] == per_family and data["seed"] == seed
                and data.get("version") == FEATURE_VERSION
                and data.get("calibrated") == CALIBRATED
                and data.get("constants") == _constants()):
            return data["samples"]
    rng = np.random.default_rng(seed)
    samples = []
    t0 = time.time()
    for family in FAMILIES:
        for i in range(per_family):
            g = generate_graph(family, rng)
            samples.append({"family": family, "features": featurize(g),
                            "rewards": evaluate_actions(g), "nodes": g.node_count()})
        print(f"  labelled {family:10s} ({time.time() - t0:.0f}s)", flush=True)
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_bytes(pickle.dumps({"per_family": per_family, "seed": seed,
                                         "version": FEATURE_VERSION,
                                         "constants": _constants(),
                                         "calibrated": __import__("utils.cost_model", fromlist=["x"]).CALIBRATED,
                                         "samples": samples}))
    return samples


# ----------------------------------------------------------------------
# Features + model
# ----------------------------------------------------------------------
def _op_vocab():
    from utils.cost_model import OP_LATENCY_US
    return sorted(OP_LATENCY_US) + ["Other"]


OPS = _op_vocab()
NODE_DIM = len(OPS) + 4
GLOBAL_DIM = 8
FEATURE_VERSION = 3  # bump when features OR the cost model change


def featurize(graph):
    """Graph -> (node features, edge list, global features)."""
    from utils.cost_model import get_op_latency

    g = graph.graph
    nodes = list(graph.topological_order())
    index = {n: i for i, n in enumerate(nodes)}
    level = {}
    for n in nodes:
        level[n] = max((level[p] for p in g.predecessors(n)), default=-1) + 1
    depth = max(level.values()) + 1
    widths = np.bincount(list(level.values()))
    x = np.zeros((len(nodes), NODE_DIM), dtype=np.float32)
    for n in nodes:
        op = g.nodes[n].get("op_type", "Other")
        x[index[n], OPS.index(op) if op in OPS else len(OPS) - 1] = 1.0
        x[index[n], len(OPS)] = min(g.in_degree(n), 8) / 8
        x[index[n], len(OPS) + 1] = min(g.out_degree(n), 8) / 8
        x[index[n], len(OPS) + 2] = get_op_latency(op) / 150.0
        x[index[n], len(OPS) + 3] = level[n] / max(1, depth - 1)
    edges = np.array([(index[u], index[v]) for u, v in g.edges()], dtype=np.int64)
    lat = {n: get_op_latency(g.nodes[n].get("op_type")) + 12.0 for n in nodes}
    crit = {}
    for n in reversed(nodes):
        crit[n] = lat[n] + max((crit[s] for s in g.successors(n)), default=0.0)
    total, critical = sum(lat.values()), max(crit.values())
    glob = np.array([math.log(len(nodes)) / 5, depth / 60, widths.max() / 30,
                     widths.mean() / 8, len(nodes) / depth / 8,
                     total / critical / 4,            # available parallelism
                     math.log(total) / 12,            # total modeled work
                     (total - critical) / total],     # parallelizable fraction
                    dtype=np.float32)
    return x, edges.reshape(-1, 2), glob


def collate(feats):
    xs, srcs, dsts, bat, globs = [], [], [], [], []
    offset = 0
    for gi, (x, e, glob) in enumerate(feats):
        xs.append(x)
        srcs.append(e[:, 0] + offset)
        dsts.append(e[:, 1] + offset)
        bat.append(np.full(len(x), gi))
        globs.append(glob)
        offset += len(x)
    return (torch.from_numpy(np.concatenate(xs)),
            torch.from_numpy(np.concatenate(srcs)),
            torch.from_numpy(np.concatenate(dsts)),
            torch.from_numpy(np.concatenate(bat)),
            torch.from_numpy(np.stack(globs)))


class PlanGNN(nn.Module):
    """Directed message passing (forward + backward) -> reward per plan."""

    def __init__(self, hidden=64, layers=3):
        super().__init__()
        self.inp = nn.Linear(NODE_DIM, hidden)
        self.fwd = nn.ModuleList(nn.Linear(2 * hidden, hidden) for _ in range(layers))
        self.bwd = nn.ModuleList(nn.Linear(2 * hidden, hidden) for _ in range(layers))
        self.head = nn.Sequential(nn.Linear(2 * hidden + GLOBAL_DIM, 128), nn.ReLU(),
                                  nn.Linear(128, len(ACTIONS)))

    @staticmethod
    def _mean_agg(h, src, dst):
        agg = torch.zeros_like(h)
        deg = torch.zeros(h.shape[0], 1)
        agg.index_add_(0, dst, h[src])
        deg.index_add_(0, dst, torch.ones(len(dst), 1))
        return agg / deg.clamp_min(1)

    def forward(self, x, src, dst, batch, glob):
        h = torch.relu(self.inp(x))
        for f, b in zip(self.fwd, self.bwd):
            hf = torch.relu(f(torch.cat([h, self._mean_agg(h, src, dst)], -1)))
            hb = torch.relu(b(torch.cat([h, self._mean_agg(h, dst, src)], -1)))
            h = h + hf + hb
        n_graphs = glob.shape[0]
        mean = torch.zeros(n_graphs, h.shape[1]).index_add_(0, batch, h)
        counts = torch.zeros(n_graphs, 1).index_add_(0, batch, torch.ones(len(batch), 1))
        mean = mean / counts.clamp_min(1)
        mx = torch.full((n_graphs, h.shape[1]), -1e9).scatter_reduce(
            0, batch[:, None].expand_as(h), h, "amax", include_self=True)
        return self.head(torch.cat([mean, mx, glob], -1))


def fit(samples, epochs=300, seed=0, lr=3e-3):
    torch.manual_seed(seed)
    model = PlanGNN()
    batch = collate([s["features"] for s in samples])
    y = torch.tensor([s["rewards"] for s in samples], dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for _ in range(epochs):
        opt.zero_grad()
        loss = nn.functional.mse_loss(model(*batch), y)
        loss.backward()
        opt.step()
        sched.step()
    model.eval()
    return model, float(loss.item())


def predict(model, feats):
    with torch.no_grad():
        return model(*collate(feats)).numpy()


# ----------------------------------------------------------------------
# Evaluation + training entry point
# ----------------------------------------------------------------------
def _fixed(name):
    return ACTIONS.index({"none": (False, False, False, 1),
                          "fixed": (True, True, True, 1),
                          "fixed4": (True, True, True, 4)}[name])


def evaluate_heldout(model, samples):
    rewards = np.array([s["rewards"] for s in samples])
    pred = predict(model, [s["features"] for s in samples])
    choice = pred.argmax(1)
    rng = np.random.default_rng(0)
    oracle = rewards.max(1)
    got = rewards[np.arange(len(samples)), choice]
    return {
        "graphs": len(samples),
        "no_passes": float(rewards[:, _fixed("none")].mean()),
        "fixed_pipeline_1dev": float(rewards[:, _fixed("fixed")].mean()),
        "fixed_pipeline_4dev": float(rewards[:, _fixed("fixed4")].mean()),
        "random_plan": float(rewards[np.arange(len(samples)),
                                     rng.integers(0, len(ACTIONS), len(samples))].mean()),
        "learned": float(got.mean()),
        "oracle": float(oracle.mean()),
        "regret_pct": float(((oracle - got) / oracle).mean() * 100),
        "within_2pct_of_oracle": float((got >= oracle * 0.98).mean() * 100),
        "exact_plan_match": float((choice == rewards.argmax(1)).mean() * 100),
        "device_count_match": float(np.mean(
            [ACTIONS[c][3] == ACTIONS[o][3] for c, o in zip(choice, rewards.argmax(1))]) * 100),
    }


def train_and_evaluate(per_family=100, epochs=300):
    print("building dataset ...", flush=True)
    samples = build_dataset(per_family)
    lofo = {}
    for fam in FAMILIES:
        train = [s for s in samples if s["family"] != fam]
        test = [s for s in samples if s["family"] == fam]
        model, tr_loss = fit(train, epochs)
        lofo[fam] = evaluate_heldout(model, test)
        lofo[fam]["train_mse"] = tr_loss
        r = lofo[fam]
        print(f"held-out {fam:10s} learned {r['learned']:.3f} fixed {r['fixed_pipeline_1dev']:.3f} "
              f"oracle {r['oracle']:.3f} regret {r['regret_pct']:.1f}%", flush=True)
    # final model: all families; random 20% split only to report in-distribution fit
    rng = np.random.default_rng(1)
    idx = rng.permutation(len(samples))
    cut = int(0.8 * len(samples))
    tr = [samples[i] for i in idx[:cut]]
    te = [samples[i] for i in idx[cut:]]
    m80, _ = fit(tr, epochs)
    random_split = evaluate_heldout(m80, te)
    final, loss = fit(samples, epochs)
    torch.save(final.state_dict(), WEIGHTS_PATH)
    mean = lambda key: float(np.mean([lofo[f][key] for f in FAMILIES]))
    summary = {k: mean(k) for k in lofo[FAMILIES[0]] if k not in ("graphs", "train_mse")}
    res = {"families": FAMILIES, "graphs_per_family": per_family, "epochs": epochs,
           "actions": [action_label(a) for a in ACTIONS],
           "device_penalty": DEVICE_PENALTY, "pass_penalty": PASS_PENALTY,
           "leave_one_family_out": lofo, "lofo_mean": summary,
           "random_split_20pct": random_split, "final_train_mse": loss,
           "environment": collect_environment()}
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(res, indent=1))
    print("LOFO mean:", json.dumps(summary, indent=1))
    make_figure(res)
    return res


def make_figure(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fams = res["families"]
    keys = [("no_passes", "No passes", "#b8bfd0"),
            ("fixed_pipeline_1dev", "Fixed pipeline (1 device)", "#8fa3e8"),
            ("fixed_pipeline_4dev", "Fixed pipeline (4 devices)", "#a6d6c8"),
            ("learned", "Learned (held-out family)", "#d1542f"),
            ("oracle", "Oracle (best plan)", "#3b3f58")]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    w = 0.16
    for j, (k, label, col) in enumerate(keys):
        vals = [res["leave_one_family_out"][f][k] for f in fams]
        ax.bar(np.arange(len(fams)) + (j - 2) * w, vals, w, label=label, color=col)
    ax.set_xticks(np.arange(len(fams)))
    ax.set_xticklabels(fams)
    ax.set_ylabel("mean reward (modeled speedup − penalties)")
    ax.set_title("Learned optimizer, leave-one-family-out evaluation")
    ax.legend(fontsize=7.5, ncol=3, loc="upper left")
    ax.grid(axis="y", alpha=.3)
    fig.tight_layout()
    for d in (ROOT / "docs" / "diagrams", ROOT / "frontend" / "public" / "diagrams"):
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / "fig_learned_optimizer.png", dpi=170)
    plt.close(fig)


# ----------------------------------------------------------------------
# Inference
# ----------------------------------------------------------------------
class LearnedOptimizer:
    def __init__(self, model):
        self.model = model

    @classmethod
    def load(cls, path=WEIGHTS_PATH):
        model = PlanGNN()
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        model.eval()
        return cls(model)

    def plan(self, graph):
        """Predicted reward for every plan on ``graph`` and the best plan."""
        pred = predict(self.model, [featurize(graph)])[0]
        best = int(pred.argmax())
        return {"action": ACTIONS[best], "label": action_label(ACTIONS[best]),
                "predicted": pred.tolist()}


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        train_and_evaluate()
    else:
        print(__doc__)
