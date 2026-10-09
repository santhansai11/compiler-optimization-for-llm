"""CLI entry point: GNN prediction, RL pass-order search, full compile."""

import sys

from models.transformer import create_demo_transformer_graph
from pipeline import run_pipeline
from search.gnn import load_or_train
from search.rl import search_pass_order
from utils.metrics import compute_metrics, format_report


def main():
    graph = create_demo_transformer_graph()

    print("=== GNN scorer ===")
    scorer = load_or_train()
    predicted = scorer.score_graph(graph)
    print(f"GNN predicted optimizability (speedup) for the input graph: "
          f"{predicted:.3f}x")

    print("\n=== RL pass-order search (tabular Q-learning) ===")
    recommended, rl_info = search_pass_order(graph, episodes=80)
    sequence = ", ".join(rl_info["best_sequence"]) or "(none)"
    print(f"recommended pass sequence : {sequence}")
    print(f"best reward               : {rl_info['best_reward']:.3f}")
    print(f"Q-table                   : {rl_info['q_states']} states over "
          f"{rl_info['episodes']} episodes")

    flags = recommended or {}

    print("\n=== Compiling with the recommended pass order ===")
    optimized, infos, compile_time = run_pipeline(
        graph,
        normalize=flags.get("normalize", True),
        canonicalize=flags.get("canonicalize", True),
        merge=flags.get("merge", True),
        schedule=flags.get("schedule", True),
        partition=flags.get("partition", False),
        neuro_symbolic=True,
        num_partitions=2,
    )

    ns_info = infos.get("neuro_symbolic_search", {})
    if "iterations" in ns_info:
        print(f"neuro-symbolic rewrites   : {ns_info['iterations']} applied "
              f"(GNN score {ns_info['initial_score']:.3f} -> "
              f"{ns_info['final_score']:.3f})")

    metrics = compute_metrics(graph, optimized, infos, compile_time)
    print()
    print(format_report(metrics))

    if "--train" in sys.argv:
        from training.train import run_training

        print("\n=== Train -> export -> compile -> verify (MNIST) ===")
        outcome = run_training()
        print(f"dataset                 : {outcome['dataset']['name']}")
        print(f"final test accuracy     : "
              f"{outcome['test_acc'] * 100:.2f}%")
        print(f"exported graph          : "
              f"{outcome['graph'].node_count()} ops -> compiled "
              f"{outcome['optimized'].node_count()} ops")
        print(f"compiled speedup        : "
              f"{outcome['metrics']['modeled']['speedup']:.2f}x (modeled), "
              f"{outcome['metrics']['speedup']:.2f}x (measured)")
        print(f"compiled vs trained model: "
              f"{outcome['pred_match_pct']:.1f}% identical predictions "
              f"(max |diff| {outcome['max_diff']:.1e})")


if __name__ == "__main__":
    main()
