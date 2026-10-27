"""Evaluate the trained Tiny Shakespeare LM on the ENTIRE validation split:
PyTorch vs the IR (original) vs the IR compiled by our passes."""

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.tiny_lm import WEIGHTS_PATH, build_graph, make_torch_model  # noqa: E402
from pipeline import run_pipeline  # noqa: E402
from utils.executor import execute  # noqa: E402
from utils.environment import collect as environment  # noqa: E402

SEQ, BATCH = 64, 32


def nll_top1(logits, y):
    m = logits.max(-1, keepdims=True)
    lp = logits - m - np.log(np.exp(logits - m).sum(-1, keepdims=True))
    return float(-np.take_along_axis(lp, y[..., None], -1).mean()), float((logits.argmax(-1) == y).mean())


def main():
    w = np.load(WEIGHTS_PATH)
    chars = [str(c) for c in w["chars"]]
    stoi = {c: i for i, c in enumerate(chars)}
    text = (ROOT / "training" / "data" / "tinyshakespeare.txt").read_text()
    data = np.array([stoi[c] for c in text])
    val = data[int(0.9 * len(data)):]
    n = (len(val) - 1) // SEQ
    xs = val[: n * SEQ].reshape(n, SEQ)
    ys = val[1: n * SEQ + 1].reshape(n, SEQ)
    vocab, d = w["tok.weight"].shape
    model = make_torch_model(vocab, d, 2, w["pos"].shape[0])
    model.load_state_dict({k: torch.from_numpy(w[k]) for k in model.state_dict()})
    model.eval()
    g = build_graph(batch=BATCH, seq=SEQ)
    opt, infos, _ = run_pipeline(g, partition=False)
    acc = {k: [0.0, 0.0, 0] for k in ("pytorch", "ir_original", "ir_compiled")}
    max_diff = {"ir_original": 0.0, "ir_compiled": 0.0}
    agree = 0
    total = 0
    for i in range(0, n - BATCH + 1, BATCH):
        x, y = xs[i:i + BATCH], ys[i:i + BATCH]
        with torch.no_grad():
            ref = model(torch.from_numpy(x)).numpy()
        outs = {"pytorch": ref,
                "ir_original": execute(g, feed=x.astype(np.float32), repeats=1, warmup=0)["output"],
                "ir_compiled": execute(opt, feed=x.astype(np.float32), repeats=1, warmup=0)["output"]}
        for k, o in outs.items():
            nl, t1 = nll_top1(o, y)
            acc[k][0] += nl * x.size
            acc[k][1] += t1 * x.size
            acc[k][2] += x.size
            if k != "pytorch":
                max_diff[k] = max(max_diff[k], float(np.abs(o - ref).max()))
        agree += int((outs["ir_compiled"].argmax(-1) == ref.argmax(-1)).sum())
        total += x.size
    res = {"validation_characters": total, "sequences": total // SEQ}
    for k, (nl, t1, c) in acc.items():
        res[k] = {"nll": nl / c, "perplexity": float(np.exp(nl / c)), "top1_acc": t1 / c}
    res["max_logit_diff_vs_pytorch"] = max_diff
    res["compiled_same_top1_as_pytorch_pct"] = 100.0 * agree / total
    res["graph_nodes"] = [g.node_count(), opt.node_count()]
    res["environment"] = environment()
    (ROOT / "docs" / "results" / "lm_full_eval.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
