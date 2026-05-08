"""Figure 3-5: continuous-history-chain heatmap (MRR / Hits@1 / Hits@10).

Reads JSON metric files produced by ``history-chain`` runs. Each model
contributes one CSV ``summary`` (history_length, MRR, Hits@1, Hits@10).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_summary(path: Path) -> dict[int, dict[str, float]]:
    rows: dict[int, dict[str, float]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            length = int(float(row["history_length"]))
            rows[length] = {
                "MRR": float(row.get("MRR", 0.0)),
                "Hits@1": float(row.get("Hits@1", 0.0)),
                "Hits@10": float(row.get("Hits@10", 0.0)),
            }
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--lengths", nargs="+", type=int, default=[1, 3, 5, 7, 10])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if len(args.summary) != len(args.labels):
        raise SystemExit("--summary and --labels lengths must match")

    summaries = [load_summary(Path(path)) for path in args.summary]
    metrics = ["MRR", "Hits@1", "Hits@10"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 5))
    for ax, metric in zip(axes, metrics):
        matrix = np.zeros((len(args.labels), len(args.lengths)))
        for i, summary in enumerate(summaries):
            for j, length in enumerate(args.lengths):
                matrix[i, j] = summary.get(length, {}).get(metric, 0.0)
        image = ax.imshow(matrix, aspect="auto", cmap="YlGnBu")
        ax.set_xticks(range(len(args.lengths)))
        ax.set_xticklabels([f"{length}段" for length in args.lengths])
        ax.set_yticks(range(len(args.labels)))
        ax.set_yticklabels(args.labels)
        ax.set_title(metric)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(
                    j,
                    i,
                    f"{matrix[i, j]:.1f}",
                    ha="center",
                    va="center",
                    color="black" if matrix[i, j] < matrix.max() * 0.7 else "white",
                    fontsize=9,
                )
        plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)


if __name__ == "__main__":
    main()
