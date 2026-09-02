"""CLI entry point: GNN prediction, RL pass-order search, full compile."""

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


if __name__ == "__main__":
    main()
