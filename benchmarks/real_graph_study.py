"""Do the passes work on REAL framework-traced graphs (not hand-built ones)?

Traces genuine PyTorch models with ``torch.fx.make_fx`` (aten level) and
imports them into the IR:
  * our trained Tiny Shakespeare Transformer (custom nn.Module, real weights)
  * HuggingFace GPT-2 architectures (randomly initialised: the *graph* is the
    real HF code path; no weights are downloaded)
Then runs every pass, checks logits against PyTorch and reports where the
node reduction comes from. Writes docs/results/real_graph_study.json.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.import_hf import import_module  # noqa: E402
from models.tiny_lm import WEIGHTS_PATH, make_torch_model  # noqa: E402
from pipeline import run_pipeline  # noqa: E402
from utils.executor import compare_latency, execute  # noqa: E402
from utils.metrics import compute_metrics  # noqa: E402


def tiny_lm(seq):
    w = np.load(WEIGHTS_PATH)
    vocab, d = w["tok.weight"].shape
    model = make_torch_model(vocab, d, 2, w["pos"].shape[0])
    model.load_state_dict({k: torch.from_numpy(w[k]) for k in model.state_dict()})
    model.eval()
    model.d_model, model.vocab = d, vocab
    return model, "tiny_lm (trained)", vocab


def gpt2(layers, dim, heads, vocab=1000, positions=64):
    import transformers

    cfg = transformers.GPT2Config(n_layer=layers, n_embd=dim, n_head=heads,
                                  vocab_size=vocab, n_positions=positions,
                                  attn_pdrop=0.0, resid_pdrop=0.0, embd_pdrop=0.0)
    torch.manual_seed(0)
    model = transformers.GPT2LMHeadModel._from_config(
        cfg, attn_implementation="eager").eval()
    return model, f"HF GPT-2 ({layers} layers, d={dim}, random init)", vocab


def study(model, label, vocab, batch=2, seq=32):
    ids = torch.randint(0, vocab, (batch, seq), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        ref = (model(input_ids=ids, use_cache=False).logits if hasattr(model, "config")
               else model(ids)).numpy()
    graph, info = import_module(model, ids, label.split()[0], label, label)
    feed = ids.numpy().astype(np.float32)

    def logits(g):
        return execute(g, feed=feed, repeats=1, warmup=0)["output"]

    orig_out = logits(graph)
    opt, infos, ctime = run_pipeline(graph, partition=False)
    opt_out = logits(opt)
    n0 = graph.node_count()
    na = infos["normalize"]["nodes_after"]
    nb = infos["attention_canonicalization"]["nodes_after"]
    nc = infos["semantic_merging"]["nodes_after"]
    ni = infos["normalize"]
    m = compute_metrics(graph, opt, infos, ctime)
    t_a, t_b = compare_latency(graph, opt, repeats=10, warmup=2, feed=feed)
    att = infos["attention_canonicalization"]
    return {
        "model": label, "ir_ops_emitted": info["ops_emitted"],
        "fx_nodes_traced": info["torch.fx nodes"], "parameters": info["parameters"],
        "nodes_original": n0, "nodes_after_normalize": na,
        "nodes_after_attention": nb, "nodes_after_merge": nc,
        "removed_by_normalize": n0 - na, "removed_by_attention": na - nb,
        "removed_by_merge": nb - nc,
        "normalize_detail": {k: ni[k] for k in ("identities_removed", "constants_folded",
                                                "cse_merges", "dead_nodes_removed",
                                                "transposes_removed", "transposes_folded",
                                                "scales_merged")},
        "attention_detected": att["detected"], "attention_canonicalized": att["canonicalized"],
        "grr": (n0 - nc) / n0,
        "ir_vs_pytorch_max_diff": float(np.abs(orig_out - ref).max()),
        "compiled_vs_pytorch_max_diff": float(np.abs(opt_out - ref).max()),
        "same_top1_pct": float((opt_out.argmax(-1) == ref.argmax(-1)).mean() * 100),
        "latency_ms_original": t_a * 1e3, "latency_ms_optimized": t_b * 1e3,
        "speedup_cpu": t_a / t_b, "compile_ms": ctime * 1e3,
        "op_histogram_before": graph.op_type_histogram(),
        "op_histogram_after": opt.op_type_histogram(),
        "fused_types": infos["semantic_merging"]["fused_types"],
    }


def unsupported_architectures():
    """Real HF architectures the importer cannot translate yet (honest list)."""
    import transformers as T

    makers = {
        "LLaMA": lambda: T.LlamaForCausalLM._from_config(T.LlamaConfig(
            hidden_size=64, intermediate_size=176, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=4, vocab_size=500, max_position_embeddings=64), attn_implementation="eager"),
        "OPT": lambda: T.OPTForCausalLM._from_config(T.OPTConfig(
            hidden_size=64, ffn_dim=256, num_hidden_layers=2, num_attention_heads=4, vocab_size=500,
            max_position_embeddings=64, word_embed_proj_dim=64), attn_implementation="eager"),
        "GPT-NeoX (Pythia)": lambda: T.GPTNeoXForCausalLM._from_config(T.GPTNeoXConfig(
            hidden_size=64, intermediate_size=256, num_hidden_layers=2, num_attention_heads=4,
            vocab_size=500, max_position_embeddings=64), attn_implementation="eager"),
        "GPT-Neo": lambda: T.GPTNeoForCausalLM._from_config(T.GPTNeoConfig(
            hidden_size=64, num_layers=2, num_heads=4, vocab_size=500, max_position_embeddings=64,
            attention_types=[[["global"], 2]]), attn_implementation="eager"),
    }
    out = []
    for name, make in makers.items():
        try:
            torch.manual_seed(0)
            ids = torch.randint(0, 500, (2, 32))
            import_module(make().eval(), ids, name.split()[0], name, name)
            out.append({"model": name, "imported": True})
        except Exception as exc:
            msg = str(exc)
            out.append({"model": name, "imported": False,
                        "reason": msg[msg.find("(first failure"):][:140] or msg[:140],
                        "unsupported_ops": msg.split(": ", 1)[-1].split(" (first")[0][:200]})
    return out


def main():
    rows = []
    specs = [lambda: tiny_lm(32), lambda: gpt2(2, 64, 2), lambda: gpt2(4, 128, 4),
             lambda: gpt2(6, 192, 6)]
    for make in specs:
        model, label, vocab = make()
        try:
            t0 = time.time()
            r = study(model, label, vocab)
            rows.append(r)
            print(f"{label:46s} ops {r['nodes_original']:4d} -norm {r['removed_by_normalize']:3d} "
                  f"-attn {r['removed_by_attention']:3d} -merge {r['removed_by_merge']:3d} -> {r['nodes_after_merge']:4d} "
                  f"| attn {r['attention_canonicalized']}/{r['attention_detected']} | "
                  f"max|diff| vs torch {r['compiled_vs_pytorch_max_diff']:.1e} top1 {r['same_top1_pct']:.0f}% "
                  f"({time.time()-t0:.0f}s)", flush=True)
        except Exception as exc:  # report, don't hide
            print(f"{label}: FAILED {type(exc).__name__}: {str(exc)[:200]}")
            rows.append({"model": label, "error": f"{type(exc).__name__}: {exc}"})
    out = ROOT / "docs" / "results" / "real_graph_study.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        unsupported = unsupported_architectures()
    except Exception as exc:  # transformers missing
        unsupported = [{"error": str(exc)}]
    out.write_text(json.dumps({"rows": rows, "torch": torch.__version__,
                               "other_architectures": unsupported}, indent=1, default=float))
    print("wrote", out)


if __name__ == "__main__":
    main()
