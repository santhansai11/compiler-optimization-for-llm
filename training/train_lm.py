"""Train a tiny char-level Transformer on Tiny Shakespeare, then compile it.

    python training/train_lm.py            # train (~few min on CPU) + compile + verify

Outputs: models/tiny_lm_weights.npz, docs/results/lm_results.json and
docs/diagrams/fig_lm_training.png.
"""

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from models.tiny_lm import build_graph, export_weights, make_torch_model  # noqa: E402
from pipeline import run_pipeline  # noqa: E402
from utils.executor import execute  # noqa: E402
from utils.metrics import compute_metrics  # noqa: E402

DATA = ROOT / "training" / "data" / "tinyshakespeare.txt"
SEQ, BATCH, STEPS, LR = 64, 64, 2500, 3e-3


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    text = DATA.read_text()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    split = int(0.9 * len(data))
    train, val = data[:split], data[split:]

    def batch(src, n=BATCH, gen=None):
        ix = torch.randint(len(src) - SEQ - 1, (n,), generator=gen)
        x = torch.stack([src[i:i + SEQ] for i in ix])
        y = torch.stack([src[i + 1:i + SEQ + 1] for i in ix])
        return x, y

    model = make_torch_model(len(chars), 64, 2, SEQ)
    params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
    print(f"chars={len(text):,} vocab={len(chars)} params={params:,}")

    @torch.no_grad()
    def eval_loss(src, iters=20):
        model.eval()
        g = torch.Generator().manual_seed(1)
        tot = sum(F.cross_entropy(model(x).flatten(0, 1), y.flatten()).item()
                  for x, y in (batch(src, 64, g) for _ in range(iters)))
        model.train()
        return tot / iters

    hist = []
    t0 = time.time()
    for step in range(1, STEPS + 1):
        x, y = batch(train)
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 100 == 0 or step == 1:
            vl = eval_loss(val)
            hist.append((step, loss.item(), vl))
            print(f"step {step:5d}  train {loss.item():.3f}  val {vl:.3f}  ({time.time() - t0:.0f}s)")
    model.eval()
    train_loss, val_loss = eval_loss(train), eval_loss(val)
    export_weights(model, chars="".join(chars))

    # ---- sample text from the trained model ----
    ids = torch.tensor([[stoi["\n"]]])
    with torch.no_grad():
        for _ in range(240):
            logits = model(ids[:, -SEQ:])[:, -1] / 0.8
            ids = torch.cat([ids, torch.multinomial(torch.softmax(logits, -1), 1)], 1)
    sample = "".join(chars[i] for i in ids[0].tolist())

    # ---- compile the TRAINED model and verify against PyTorch ----
    b = 8
    g_ids = torch.Generator().manual_seed(7)
    xs, ys = batch(val, b, g_ids)
    with torch.no_grad():
        ref = model(xs).numpy()
    graph = build_graph(batch=b, seq=SEQ)
    opt_graph, infos, ctime = run_pipeline(graph, partition=False)
    feed = xs.numpy().astype(np.float32)
    out_ir = execute(graph, feed=feed, repeats=10)["output"]
    out_opt = execute(opt_graph, feed=feed, repeats=10)["output"]
    metrics = compute_metrics(graph, opt_graph, infos, ctime, batch_size=b)

    def nll(logits):
        lp = logits - np.log(np.exp(logits - logits.max(-1, keepdims=True)).sum(-1, keepdims=True)) - logits.max(-1, keepdims=True)
        return float(-np.take_along_axis(lp, ys.numpy()[..., None], -1).mean())

    def top1(logits):
        return float((logits.argmax(-1) == ys.numpy()).mean())

    res = {
        "dataset": "Tiny Shakespeare (char-level)", "characters": len(text),
        "vocab": len(chars), "parameters": params, "steps": STEPS,
        "batch": BATCH, "seq": SEQ, "train_seconds": time.time() - t0,
        "train_loss": train_loss, "val_loss": val_loss,
        "val_perplexity": math.exp(val_loss),
        "export_vs_pytorch_max_diff": float(np.abs(out_ir - ref).max()),
        "compiled_vs_pytorch_max_diff": float(np.abs(out_opt - ref).max()),
        "same_top1_prediction_pct": float((out_opt.argmax(-1) == ref.argmax(-1)).mean() * 100),
        "pytorch": {"nll": nll(ref), "top1": top1(ref)},
        "compiled": {"nll": nll(out_opt), "top1": top1(out_opt)},
        "nodes_before": metrics["counts"]["nodes_original"],
        "nodes_after": metrics["counts"]["nodes_optimized"],
        "grr": metrics["graph_reduction_ratio"], "omr": metrics["operator_merge_ratio"],
        "acr": metrics["attention_canonicalization_rate"],
        "attention": metrics["attention_subgraphs"],
        "kernel_launch_reduction": metrics["kernel_launch_reduction"],
        "latency_ms": metrics["latency_ms"], "speedup_measured": metrics["speedup"],
        "speedup_modeled": metrics["modeled"]["speedup"],
        "gpu_memory_mb": [metrics["modeled"]["memory_original_mb"], metrics["modeled"]["memory_optimized_mb"]],
        "compile_ms": ctime * 1e3, "fused_types": infos["semantic_merging"]["fused_types"],
        "sample": sample, "history": hist,
    }
    out = ROOT / "docs" / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / "lm_results.json").write_text(json.dumps(res, indent=1, default=float))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s, tr, va = zip(*hist)
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.plot(s, tr, label="train loss (batch)")
    ax.plot(s, va, label="validation loss")
    ax.set_xlabel("training step"); ax.set_ylabel("cross-entropy (nats)")
    ax.set_title("Tiny Shakespeare character LM"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout()
    for d in (ROOT / "docs" / "diagrams", ROOT / "frontend" / "public" / "diagrams"):
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / "fig_lm_training.png", dpi=170)
    print(json.dumps({k: v for k, v in res.items() if k not in ("history", "sample")}, indent=1, default=float))
    print("--- sample ---\n" + sample)


if __name__ == "__main__":
    main()
