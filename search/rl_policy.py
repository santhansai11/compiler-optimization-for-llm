"""Reinforcement learning: a sequential pass-selection policy (DQN).

The compiler is cast as an MDP:

  state   the CURRENT graph (after the passes applied so far) + a mask of
          which passes were already applied
  action  apply `normalize` | `canonicalize` | `merge` | place on k=2/3/4
          devices (terminal) | `stop` (terminal)
  reward  shaped difference of the modeled deployment reward before/after the
          action, so the episode return equals the final plan reward used by
          ``search.learned_optimizer``:
              speedup - 0.15*(devices-1) - 0.005*(#passes)

A graph-neural Q-network reads the current graph and picks the next action;
it is trained with temporal-difference targets (target network, replay of
every transition of the training graphs) and evaluated by greedy rollouts on
architecture families it never saw (leave-one-family-out).

The environment is deterministic and small, so its transition table (the
graph after every pass ordering + the reward of each placement) is computed
once per training graph with the real pass implementations and cached;
rollouts then look states up instead of re-running passes. This is an offline
DQN on a tabulated exact simulator - not interaction with real hardware.

    python -m search.rl_policy train
"""

import json
import pickle
import sys
import time
from itertools import permutations
from pathlib import Path

import numpy as np
import torch
from torch import nn

from search import learned_optimizer as lo

ROOT = lo.ROOT
CACHE = ROOT / "training" / "artifacts" / "rl_tables.pkl"
WEIGHTS = Path(__file__).with_name("rl_policy.pt")
RESULTS = ROOT / "docs" / "results" / "rl_results.json"

PASSES = ("normalize", "canonicalize", "merge")
ACTIONS = list(PASSES) + ["place2", "place3", "place4", "stop"]
N_ACT = len(ACTIONS)


def apply_pass(graph, name):
    from passes.canonicalize import canonicalize_attention
    from passes.merge import merge_operators
    from passes.normalize import normalize_graph

    fn = {"normalize": normalize_graph, "canonicalize": canonicalize_attention,
          "merge": merge_operators}[name]
    return fn(graph)[0]


def _reward(base, graph, n_passes, devices=1):
    from utils.cost_model import deployment_latency_us

    return (base / deployment_latency_us(graph)
            - lo.DEVICE_PENALTY * (devices - 1) - lo.PASS_PENALTY * n_passes)


def tabulate(graph):
    """All pass orderings -> {sequence: features, stay reward, placement rewards}."""
    from passes.partition import partition_latency_aware
    from utils.cost_model import deployment_latency_us

    base = deployment_latency_us(graph)
    graphs = {(): graph}
    for r in range(1, len(PASSES) + 1):
        for seq in permutations(PASSES, r):
            graphs[seq] = apply_pass(graphs[seq[:-1]], seq[-1])
    table = {}
    for seq, g in graphs.items():
        place = []
        for k in (2, 3, 4):
            placed, info = partition_latency_aware(g, k)
            place.append(_reward(base, placed, len(seq), info["partitions"]))
        table[seq] = {"features": lo.featurize(g), "stay": _reward(base, g, len(seq)),
                      "place": place}
    return table


def build_tables(per_family=100, seed=0, use_cache=True):
    from utils.cost_model import CALIBRATED

    if use_cache and CACHE.exists():
        data = pickle.loads(CACHE.read_bytes())
        if (data["per_family"], data["seed"], data["calibrated"]) == (per_family, seed, CALIBRATED):
            return data["items"]
    rng = np.random.default_rng(seed)  # same generator order as learned_optimizer
    items, t0 = [], time.time()
    for family in lo.FAMILIES:
        for _ in range(per_family):
            g = lo.generate_graph(family, rng)
            items.append({"family": family, "table": tabulate(g)})
        print(f"  tabulated {family:10s} ({time.time() - t0:.0f}s)", flush=True)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_bytes(pickle.dumps({"per_family": per_family, "seed": seed,
                                    "calibrated": CALIBRATED, "items": items}))
    return items


# ---------------------------------------------------------------- Q-network
class QNet(nn.Module):
    """Same directed message passing as the plan selector; Q-values per action."""

    def __init__(self, hidden=48, layers=3):
        super().__init__()
        self.inp = nn.Linear(lo.NODE_DIM, hidden)
        self.fwd = nn.ModuleList(nn.Linear(2 * hidden, hidden) for _ in range(layers))
        self.bwd = nn.ModuleList(nn.Linear(2 * hidden, hidden) for _ in range(layers))
        self.head = nn.Sequential(
            nn.Linear(2 * hidden + lo.GLOBAL_DIM + len(PASSES), 128), nn.ReLU(),
            nn.Linear(128, N_ACT))

    def forward(self, x, src, dst, batch, glob, mask):
        h = torch.relu(self.inp(x))
        agg = lo.PlanGNN._mean_agg
        for f, b in zip(self.fwd, self.bwd):
            h = h + torch.relu(f(torch.cat([h, agg(h, src, dst)], -1))) \
                  + torch.relu(b(torch.cat([h, agg(h, dst, src)], -1)))
        n = glob.shape[0]
        mean = torch.zeros(n, h.shape[1]).index_add_(0, batch, h)
        cnt = torch.zeros(n, 1).index_add_(0, batch, torch.ones(len(batch), 1))
        mx = torch.full((n, h.shape[1]), -1e9).scatter_reduce(
            0, batch[:, None].expand_as(h), h, "amax", include_self=True)
        return self.head(torch.cat([mean / cnt.clamp_min(1), mx, glob, mask], -1))


def _mask_vec(seq):
    return [1.0 if p in seq else 0.0 for p in PASSES]


def _legal(seq):
    return [a for a in ACTIONS if a not in seq]


def transitions(items):
    """Every (state, action, reward, next-state | terminal) of the tabulated MDP."""
    out = []
    for gi, item in enumerate(items):
        t = item["table"]
        for seq, row in t.items():
            for a in _legal(seq):
                if a in PASSES:
                    nxt = seq + (a,)
                    out.append((gi, seq, ACTIONS.index(a), t[nxt]["stay"] - row["stay"], nxt))
                elif a == "stop":
                    out.append((gi, seq, ACTIONS.index(a), 0.0, None))
                else:
                    k = int(a[-1])
                    out.append((gi, seq, ACTIONS.index(a), row["place"][k - 2] - row["stay"], None))
    return out


def _batch(items, pairs):
    feats = [items[g]["table"][s]["features"] for g, s in pairs]
    mask = torch.tensor([_mask_vec(s) for _, s in pairs], dtype=torch.float32)
    return (*lo.collate(feats), mask)


def _legal_mask(seqs):
    m = torch.full((len(seqs), N_ACT), -1e9)
    for i, s in enumerate(seqs):
        for a in _legal(s):
            m[i, ACTIONS.index(a)] = 0.0
    return m


def train_dqn(items, steps=2500, batch=96, lr=1e-3, seed=0, sync_every=150):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    trans = transitions(items)
    q, target = QNet(), QNet()
    target.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=lr)
    losses = []
    for step in range(1, steps + 1):
        idx = rng.integers(0, len(trans), batch)
        tr = [trans[i] for i in idx]
        qs = q(*_batch(items, [(g, s) for g, s, _, _, _ in tr]))
        pred = qs[torch.arange(batch), torch.tensor([a for _, _, a, _, _ in tr])]
        with torch.no_grad():
            nxt_rows = [(i, (g, n)) for i, (g, _, _, _, n) in enumerate(tr) if n is not None]
            best = torch.zeros(batch)
            if nxt_rows:
                ids, pairs = zip(*nxt_rows)
                qn = target(*_batch(items, list(pairs))) + _legal_mask([p[1] for p in pairs])
                best[list(ids)] = qn.max(1).values
            y = torch.tensor([r for _, _, _, r, _ in tr]) + best
        loss = nn.functional.smooth_l1_loss(pred, y)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(q.parameters(), 5.0)
        opt.step()
        losses.append(loss.item())
        if step % sync_every == 0:
            target.load_state_dict(q.state_dict())
    q.eval()
    return q, float(np.mean(losses[-200:]))


def rollout(q, table):
    """Greedy policy rollout on a tabulated graph; returns (return, action list)."""
    seq, actions = (), []
    for _ in range(8):
        row = table[seq]
        with torch.no_grad():
            values = q(*_batch([{"table": table}], [(0, seq)]))[0] + _legal_mask([seq])[0]
        a = ACTIONS[int(values.argmax())]
        actions.append(a)
        if a in PASSES:
            seq = seq + (a,)
        elif a == "stop":
            return row["stay"], actions
        else:
            return row["place"][int(a[-1]) - 2], actions
    return table[seq]["stay"], actions


def evaluate(q, items, seed=0):
    rng = np.random.default_rng(seed)
    got, fixed, none, oracle, rand = [], [], [], [], []
    for it in items:
        t = it["table"]
        got.append(rollout(q, t)[0])
        none.append(t[()]["stay"])
        fixed.append(t[PASSES]["stay"])
        finals = [row["stay"] for row in t.values()] + [r for row in t.values() for r in row["place"]]
        oracle.append(max(finals))
        # random policy: uniformly random legal action until terminal
        seq = ()
        while True:
            a = _legal(seq)[int(rng.integers(len(_legal(seq))))]
            if a in PASSES:
                seq += (a,)
            elif a == "stop":
                rand.append(t[seq]["stay"])
                break
            else:
                rand.append(t[seq]["place"][int(a[-1]) - 2])
                break
    got, oracle = np.array(got), np.array(oracle)
    return {"graphs": len(items), "no_passes": float(np.mean(none)),
            "fixed_pipeline_1dev": float(np.mean(fixed)),
            "random_policy": float(np.mean(rand)), "rl": float(got.mean()),
            "oracle": float(oracle.mean()),
            "regret_pct": float(((oracle - got) / oracle).mean() * 100),
            "within_2pct_of_oracle": float((got >= oracle * 0.98).mean() * 100)}


def main(per_family=100, steps=2500):
    print("tabulating the pass-ordering MDP ...", flush=True)
    items = build_tables(per_family)
    lofo = {}
    for fam in lo.FAMILIES:
        train = [it for it in items if it["family"] != fam]
        test = [it for it in items if it["family"] == fam]
        q, loss = train_dqn(train, steps)
        lofo[fam] = evaluate(q, test)
        lofo[fam]["td_loss"] = loss
        r = lofo[fam]
        print(f"held-out {fam:10s} RL {r['rl']:.3f} fixed {r['fixed_pipeline_1dev']:.3f} "
              f"random {r['random_policy']:.3f} oracle {r['oracle']:.3f} regret {r['regret_pct']:.1f}%",
              flush=True)
    final, _ = train_dqn(items, steps)
    torch.save(final.state_dict(), WEIGHTS)
    keys = [k for k in lofo[lo.FAMILIES[0]] if k not in ("graphs", "td_loss")]
    summary = {k: float(np.mean([lofo[f][k] for f in lo.FAMILIES])) for k in keys}
    res = {"families": lo.FAMILIES, "actions": ACTIONS, "steps": steps,
           "leave_one_family_out": lofo, "lofo_mean": summary}
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(res, indent=1))
    print("LOFO mean:", json.dumps(summary, indent=1))


class RLPolicy:
    @classmethod
    def load(cls, path=WEIGHTS):
        q = QNet()
        q.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        q.eval()
        p = cls()
        p.q = q
        return p

    def plan(self, graph):
        """Run the learned policy on a real graph; returns (final graph, actions)."""
        from passes.partition import partition_latency_aware

        seq, actions, g = (), [], graph
        for _ in range(8):
            feats = lo.featurize(g)
            with torch.no_grad():
                v = self.q(*lo.collate([feats]), torch.tensor([_mask_vec(seq)]))[0] \
                    + _legal_mask([seq])[0]
            a = ACTIONS[int(v.argmax())]
            actions.append(a)
            if a in PASSES:
                g, seq = apply_pass(g, a), seq + (a,)
            elif a == "stop":
                break
            else:
                g, _ = partition_latency_aware(g, int(a[-1]))
                break
        return g, actions


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        main()
    else:
        print(__doc__)
