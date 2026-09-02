"""End-to-end training experiment.

Loads the dataset → trains an MLP (NumPy, manual backprop + Adam) →
exports the trained weights into the computation-graph IR → compiles
that graph with the optimizer passes → verifies the compiled graph
predicts identically → measures metrics. Artifacts are cached on disk
so repeat runs (and the UI) are fast.
"""

import os
import time

import numpy as np

from training.dataset import load_dataset
from training.model import MLP, evaluate, train
from training.export_graph import export_mlp_graph
from pipeline import run_pipeline

ARTIFACTS = os.path.join(os.path.dirname(__file__), "artifacts")
SIZES = (784, 256, 64, 10)


def _load_or_train(epochs=6, batch_size=64, lr=2e-3, n_train=12000,
                   n_test=2000, force=False):
    os.makedirs(ARTIFACTS, exist_ok=True)
    params_path = os.path.join(ARTIFACTS, "mnist_mlp_params.npz")
    history_path = os.path.join(ARTIFACTS, "training_history.npz")

    data = load_dataset("mnist", n_train=n_train, n_test=n_test)
    model = MLP(SIZES, seed=0)

    if force or not os.path.exists(params_path):
        started = time.perf_counter()
        history = train(model, data, epochs=epochs,
                        batch_size=batch_size, lr=lr)
        train_time_s = time.perf_counter() - started
        np.savez(params_path, sizes=np.array(model.sizes), **{
            f"W{i}": W for i, (W, _) in enumerate(model.params)
        }, **{
            f"b{i}": b for i, (_, b) in enumerate(model.params)
        })
        np.savez(history_path, **{key: np.array(value)
                                  for key, value in history.items()})
    else:
        stored = np.load(params_path)
        model.load(stored)
        stored_h = np.load(history_path)
        history = {key: list(stored_h[key]) for key in stored_h.files}
        train_time_s = None

    test_loss, test_acc = evaluate(model, data["x_test"], data["y_test"])
    y_pred = model.predict(data["x_test"])
    confusion = np.zeros((10, 10), dtype=int)
    np.add.at(confusion, (data["y_test"], y_pred), 1)
    return data, model, history, test_loss, test_acc, confusion, train_time_s


def run_training(epochs=6, batch_size=64, lr=2e-3, n_train=12000,
                 n_test=2000, force=False):
    """Full experiment; returns everything the UI/CLI/figures need."""
    (data, model, history, test_loss, test_acc, confusion,
     train_time_s) = _load_or_train(epochs, batch_size, lr, n_train,
                                    n_test, force)

    graph = export_mlp_graph(model.params, act="relu",
                             name="trained_mnist_mlp")
    optimized, infos, compile_time = run_pipeline(graph, partition=False)
    trained_metrics = _metrics_for(graph, optimized, infos, compile_time)

    from utils.executor import execute

    feed = data["x_test"][:512]
    original_out = execute(graph, feed=feed)["output"]
    compiled_out = execute(optimized, feed=feed)["output"]
    pred_match_pct = float(np.mean(
        np.argmax(original_out, axis=1) == np.argmax(compiled_out, axis=1)
    )) * 100.0
    max_diff = float(np.max(np.abs(original_out - compiled_out)))

    return {
        "dataset": data,
        "history": history,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "train_time_s": train_time_s,
        "confusion": confusion,
        "graph": graph,
        "optimized": optimized,
        "infos": infos,
        "metrics": trained_metrics,
        "pred_match_pct": pred_match_pct,
        "max_diff": max_diff,
        "compile_time_s": compile_time,
    }


def _metrics_for(original, optimized, infos, compile_time):
    from utils.metrics import compute_metrics

    return compute_metrics(original, optimized, infos, compile_time,
                           batch_size=256)


if __name__ == "__main__":
    outcome = run_training(force=False)
    print(f"dataset            : {outcome['dataset']['name']}")
    print(f"final test accuracy: {outcome['test_acc'] * 100:.2f}%")
    print(f"training time      : "
          f"{(outcome['train_time_s'] or 0):.1f} s")
    print(f"compiled graph     : "
          f"{outcome['graph'].node_count()} -> "
          f"{outcome['optimized'].node_count()} ops, "
          f"speedup {outcome['metrics']['modeled']['speedup']:.2f}x "
          f"(modeled), {outcome['metrics']['speedup']:.2f}x (measured)")
    print(f"compiled predictions match trained model: "
          f"{outcome['pred_match_pct']:.2f}% "
          f"(max |diff| {outcome['max_diff']:.2e})")
