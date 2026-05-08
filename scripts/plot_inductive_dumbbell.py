"""Figure 3-3: dumbbell + scatter for inductive 1-Shot vs 10-Shot.

Inputs are JSON metric files produced by ``eval-subset --history-shot``
runs: one file per (model, shot) pair. Pass ``--metrics`` as
``model_label:shot:path`` triples and the script will assemble the
dumbbell connecting 1-Shot and 10-Shot per model on the left axis, and
plot the (1-Shot/10-Shot retention vs absolute drop) scatter on the
right axis.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def parse_triplet(value: str) -> tuple[str, int, str]:
    parts = value.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"--metrics entry must be model:shot:path, got '{value}'"
        )
    return parts[0], int(parts[1]), parts[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics",
        nargs="+",
        type=parse_triplet,
        required=True,
        help="Repeated 'model_label:shot:path' triples.",
    )
    parser.add_argument("--metric-key", default="MRR")
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", default="Inductive 1-Shot vs 10-Shot")
    args = parser.parse_args()

    bucket: dict[str, dict[int, float]] = defaultdict(dict)
    for label, shot, path in args.metrics:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        bucket[label][int(shot)] = float(payload[args.metric_key])

    models = list(bucket.keys())
    fig, (left_ax, right_ax) = plt.subplots(1, 2, figsize=(13, 5))

    y_positions = list(range(len(models)))
    for y, model in zip(y_positions, models):
        shots = bucket[model]
        if 1 in shots and 10 in shots:
            left_ax.plot([shots[1], shots[10]], [y, y], color="steelblue", linewidth=2.0)
            left_ax.scatter([shots[1]], [y], color="orange", s=70, label="1-Shot" if y == 0 else None, zorder=3)
            left_ax.scatter([shots[10]], [y], color="navy", s=70, label="10-Shot" if y == 0 else None, zorder=3)
            drop = shots[10] - shots[1]
            left_ax.annotate(f"Δ={drop:.2f}", xy=((shots[1] + shots[10]) / 2, y + 0.15), ha="center", fontsize=9)
    left_ax.set_yticks(y_positions)
    left_ax.set_yticklabels(models)
    left_ax.set_xlabel(args.metric_key)
    left_ax.set_title("1-Shot ↔ 10-Shot dumbbell")
    left_ax.grid(axis="x", alpha=0.3)
    left_ax.legend(loc="lower right")

    for model in models:
        shots = bucket[model]
        if 1 in shots and 10 in shots and shots[10] > 0:
            retention = 100.0 * shots[1] / shots[10]
            absolute_drop = shots[10] - shots[1]
            right_ax.scatter([retention], [absolute_drop], s=120)
            right_ax.annotate(model, xy=(retention, absolute_drop), xytext=(5, 5), textcoords="offset points", fontsize=9)
    right_ax.set_xlabel("1-Shot / 10-Shot retention rate (%)")
    right_ax.set_ylabel(f"10-Shot − 1-Shot {args.metric_key} drop")
    right_ax.set_title("Retention rate vs absolute drop")
    right_ax.grid(alpha=0.3)

    fig.suptitle(args.title)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)


if __name__ == "__main__":
    main()
