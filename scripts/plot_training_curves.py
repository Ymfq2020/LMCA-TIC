"""Plot convergence curves from train_history.jsonl."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from _bootstrap import ensure_src_on_path

ensure_src_on_path()

from lmca_tic.utils.io import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = read_jsonl(args.history)
    epochs = [row["epoch"] for row in rows]
    plt.figure(figsize=(10, 6))
    for metric in ("valid_mrr", "valid_hits@1", "valid_hits@3", "valid_hits@10"):
        plt.plot(epochs, [row.get(metric, 0.0) for row in rows], label=metric)
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.legend()
    plt.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=200)


if __name__ == "__main__":
    main()
