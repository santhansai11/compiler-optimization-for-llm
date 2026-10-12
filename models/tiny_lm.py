"""Tiny character-level Transformer language model (pre-LN, single head).

The PyTorch module is trained by ``training/train_lm.py``. ``build_graph``
exports the *trained* weights into the compiler's ComputationGraph IR so the
real model can be optimized and checked against PyTorch.
"""

import math
from pathlib import Path

import numpy as np

from ir.graph import ComputationGraph

WEIGHTS_PATH = Path(__file__).resolve().parent / "tiny_lm_weights.npz"


def make_torch_model(vocab, d_model=64, n_layers=2, max_seq=64):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln1 = nn.LayerNorm(d_model)
            self.wq = nn.Linear(d_model, d_model, bias=False)
            self.wk = nn.Linear(d_model, d_model, bias=False)
            self.wv = nn.Linear(d_model, d_model, bias=False)
            self.wo = nn.Linear(d_model, d_model, bias=False)
            self.ln2 = nn.LayerNorm(d_model)
            self.fc1 = nn.Linear(d_model, 4 * d_model)
            self.fc2 = nn.Linear(4 * d_model, d_model)

        def forward(self, x):
            h = self.ln1(x)
            q, k, v = self.wq(h), self.wk(h), self.wv(h)
            s = (q @ k.transpose(-2, -1)) / math.sqrt(d_model)
            n = x.shape[1]
            mask = torch.tril(torch.ones(n, n, device=x.device)) != 0
            s = s.masked_fill(~mask, float("-inf"))
            x = x + self.wo(torch.softmax(s, -1) @ v)
            h = self.ln2(x)
            return x + self.fc2(F.gelu(self.fc1(h), approximate="tanh"))

    class TinyLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok = nn.Embedding(vocab, d_model)
            self.pos = nn.Parameter(torch.randn(max_seq, d_model) * 0.02)
            self.blocks = nn.ModuleList([Block() for _ in range(n_layers)])
            self.lnf = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab, bias=False)

        def forward(self, ids):
            x = self.tok(ids) + self.pos[: ids.shape[1]]
            for b in self.blocks:
                x = b(x)
            return self.head(self.lnf(x))

    return TinyLM()


def export_weights(model, path=WEIGHTS_PATH, chars=""):
    sd = {k: v.detach().cpu().numpy().astype(np.float32)
          for k, v in model.state_dict().items()}
    sd["chars"] = np.array(list(chars))
    np.savez_compressed(path, **sd)


def build_graph(batch=4, seq=64, path=WEIGHTS_PATH):
    """Trained weights -> ComputationGraph (executable by utils.executor)."""
    w = np.load(path)
    vocab, d = w["tok.weight"].shape
    seq = min(int(seq), w["pos"].shape[0])
    n_layers = len({k.split(".")[1] for k in w.files if k.startswith("blocks.")})
    g = ComputationGraph("tiny_lm")
    g.meta["config"] = {"batch": batch, "seq": seq, "d_model": d, "llm": True,
                        "vocab_size": vocab, "input_shape": [seq]}
    lin = lambda name: np.ascontiguousarray(w[name].T)  # (out,in)->(in,out)

    g.add_operation("Input", "Input")
    g.add_operation("TokEmbed", "Gather", weight=w["tok.weight"])
    g.add_dependency("Input", "TokEmbed")
    g.add_operation("PosEmbed", "Constant", value=w["pos"][:seq], shape=(seq, d))
    g.add_operation("Embed_Add", "Add")
    g.add_dependency("TokEmbed", "Embed_Add")
    g.add_dependency("PosEmbed", "Embed_Add")
    mask = np.tril(np.ones((seq, seq), dtype=np.float32))
    node = "Embed_Add"
    for i in range(n_layers):
        p, b = f"B{i}", f"blocks.{i}"
        g.add_operation(f"{p}_LN1", "LayerNorm", weight=w[f"{b}.ln1.weight"],
                        bias=w[f"{b}.ln1.bias"], eps=1e-5)
        g.add_dependency(node, f"{p}_LN1")
        for r in "qkv":
            g.add_operation(f"{p}_{r.upper()}", "MatMul", weight=lin(f"{b}.w{r}.weight"))
            g.add_dependency(f"{p}_LN1", f"{p}_{r.upper()}")
        g.add_operation(f"{p}_QK", "MatMul", transpose_rhs=True)
        g.add_dependency(f"{p}_Q", f"{p}_QK")
        g.add_dependency(f"{p}_K", f"{p}_QK")
        g.add_operation(f"{p}_Scale", "Scale", factor=float(1 / math.sqrt(d)))
        g.add_dependency(f"{p}_QK", f"{p}_Scale")
        g.add_operation(f"{p}_Mask", "Constant", value=mask, shape=mask.shape)
        g.add_operation(f"{p}_Causal", "MaskedFill", fill_value=float("-inf"))
        g.add_dependency(f"{p}_Scale", f"{p}_Causal")
        g.add_dependency(f"{p}_Mask", f"{p}_Causal")
        g.add_operation(f"{p}_Softmax", "Softmax")
        g.add_dependency(f"{p}_Causal", f"{p}_Softmax")
        g.add_operation(f"{p}_AV", "MatMul")
        g.add_dependency(f"{p}_Softmax", f"{p}_AV")
        g.add_dependency(f"{p}_V", f"{p}_AV")
        g.add_operation(f"{p}_O", "MatMul", weight=lin(f"{b}.wo.weight"))
        g.add_dependency(f"{p}_AV", f"{p}_O")
        g.add_operation(f"{p}_Res1", "Add")
        g.add_dependency(f"{p}_O", f"{p}_Res1")
        g.add_dependency(node, f"{p}_Res1")
        g.add_operation(f"{p}_LN2", "LayerNorm", weight=w[f"{b}.ln2.weight"],
                        bias=w[f"{b}.ln2.bias"], eps=1e-5)
        g.add_dependency(f"{p}_Res1", f"{p}_LN2")
        g.add_operation(f"{p}_FC1", "MatMul", weight=lin(f"{b}.fc1.weight"))
        g.add_dependency(f"{p}_LN2", f"{p}_FC1")
        g.add_operation(f"{p}_B1", "Constant", value=w[f"{b}.fc1.bias"],
                        shape=w[f"{b}.fc1.bias"].shape)
        g.add_operation(f"{p}_FC1b", "Add")
        g.add_dependency(f"{p}_FC1", f"{p}_FC1b")
        g.add_dependency(f"{p}_B1", f"{p}_FC1b")
        g.add_operation(f"{p}_Act", "GELU")
        g.add_dependency(f"{p}_FC1b", f"{p}_Act")
        g.add_operation(f"{p}_FC2", "MatMul", weight=lin(f"{b}.fc2.weight"))
        g.add_dependency(f"{p}_Act", f"{p}_FC2")
        g.add_operation(f"{p}_B2", "Constant", value=w[f"{b}.fc2.bias"],
                        shape=w[f"{b}.fc2.bias"].shape)
        g.add_operation(f"{p}_FC2b", "Add")
        g.add_dependency(f"{p}_FC2", f"{p}_FC2b")
        g.add_dependency(f"{p}_B2", f"{p}_FC2b")
        g.add_operation(f"{p}_Res2", "Add")
        g.add_dependency(f"{p}_FC2b", f"{p}_Res2")
        g.add_dependency(f"{p}_Res1", f"{p}_Res2")
        node = f"{p}_Res2"
    g.add_operation("LN_final", "LayerNorm", weight=w["lnf.weight"],
                    bias=w["lnf.bias"], eps=1e-5)
    g.add_dependency(node, "LN_final")
    g.add_operation("Logits", "MatMul", weight=lin("head.weight"), is_output=True)
    g.add_dependency("LN_final", "Logits")
    return g
