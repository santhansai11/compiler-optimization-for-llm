"""Train and report the PyTorch graph cost model used by learned search."""

import argparse
import json

from search.gnn_torch import train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graphs", type=int, default=96,
                        help="synthetic DAG workloads used for supervision")
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=None,
                        help="optional model checkpoint path")
    args = parser.parse_args()
    kwargs = {"corpus_size": args.graphs, "epochs": args.epochs,
              "seed": args.seed}
    if args.output:
        kwargs["path"] = args.output
    scorer = train(**kwargs)
    print(json.dumps(scorer.training_info, indent=2))


if __name__ == "__main__":
    main()
