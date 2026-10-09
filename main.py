import torch

from models.transformer_torch import TransformerModel
from ir.fx_graph import (
    extract_graph,
    print_graph,
    count_compute_nodes,
)
from passes.normalize import normalize
from utils.visualize_graph import draw_graph


def main():

    torch.manual_seed(42)

    # -------------------------
    # LLM-style input
    # -------------------------

    batch_size = 4
    sequence_length = 32
    vocab_size = 1000
    d_model = 64

    token_ids = torch.randint(
        0,
        vocab_size,
        (
            batch_size,
            sequence_length,
        )
    )

    print("===== DATA =====")
    print("Token IDs :", token_ids.shape)
    print("dtype     :", token_ids.dtype)


    # -------------------------
    # Model
    # -------------------------

    model = TransformerModel(
        vocab_size=vocab_size,
        max_seq_len=sequence_length,
        d_model=d_model,
        d_ff=256,
        num_blocks=2,
    )

    print("\n===== MODEL =====")
    print(model)


    # -------------------------
    # Original graph
    # -------------------------

    original_graph = extract_graph(model)

    print_graph(
        original_graph,
        "ORIGINAL TRANSFORMER DAG"
    )

    original_nodes = count_compute_nodes(
        original_graph
    )

    print(
        "\nOriginal compute nodes:",
        original_nodes
    )

    draw_graph(
        original_graph,
        "Original Transformer DAG",
        "original_graph.png"
    )


    # -------------------------
    # Original execution
    # -------------------------

    with torch.no_grad():

        original_output = original_graph(
            token_ids
        )

    print(
        "\nOriginal output:",
        original_output.shape
    )


    # -------------------------
    # Normalize
    # -------------------------

    optimized_graph = extract_graph(model)

    normalize(
        optimized_graph
    )

    print_graph(
        optimized_graph,
        "NORMALIZED TRANSFORMER DAG"
    )

    optimized_nodes = count_compute_nodes(
        optimized_graph
    )

    print(
        "\nNormalized compute nodes:",
        optimized_nodes
    )

    draw_graph(
        optimized_graph,
        "Normalized Transformer DAG",
        "normalized_graph.png"
    )


    # -------------------------
    # Optimized execution
    # -------------------------

    with torch.no_grad():

        optimized_output = optimized_graph(
            token_ids
        )

    print(
        "\nNormalized output:",
        optimized_output.shape
    )


    # -------------------------
    # Correctness
    # -------------------------

    torch.testing.assert_close(
        original_output,
        optimized_output,
        rtol=1e-5,
        atol=1e-6,
    )


    # -------------------------
    # Metrics
    # -------------------------

    reduction = (
        (original_nodes - optimized_nodes)
        / original_nodes
        * 100
    )

    print("\n===== RESULTS =====")

    print(
        "Original nodes    :",
        original_nodes
    )

    print(
        "Normalized nodes  :",
        optimized_nodes
    )

    print(
        "Graph reduction   :",
        f"{reduction:.2f}%"
    )

    print(
        "Output preserved  : YES"
    )

    print(
        "\nSaved:"
    )

    print(
        "original_graph.png"
    )

    print(
        "normalized_graph.png"
    )


if __name__ == "__main__":
    main()