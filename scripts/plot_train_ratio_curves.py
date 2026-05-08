"""Figure 3-4: training-data ratio learning curves with retention scatter.

Reads ``--summary`` CSV files produced by ``run-suite --seeds`` for each
model under different training ratios. Each row should provide an
``experiment`` column tagged with the ratio (e.g. ``..._train20p``) and
mean/std for ``MRR_mean`` / ``MRR_std`` etc.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


RATIO_RE = re.compile(r"_train(\d+)p")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", nargs="+", required=True, help="Per-model summary CSV files (one per model)")
    parser.add_argument("--labels", nargs="+", required=True, help="Display label per --summary entry")
    parser.add_argument("--metric", default="MRR")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if len(args.summary) != len(args.labels):
        raise SystemExit("--summary and --labels lengths must match")

    fig, (curve_ax, scatter_ax) = plt.subplots(1, 2, figsize=(13, 5))

    retention_rows = []
    for path, label in zip(args.summary, args.labels):
        ratios: list[float] = []
        means: list[float] = []
        stds: list[float] = []
        with Path(path).open("r", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                experiment = row.get("experiment", "")
                match = RATIO_RE.search(experiment)
                if not match:
                    continue
                ratio = float(match.group(1))
                ratios.append(ratio)
                means.append(float(row[f"{args.metric}_mean"]))
                stds.append(float(row[f"{args.metric}_std"]))
        if not ratios:
            continue
        order = sorted(range(len(ratios)), key=lambda i: ratios[i])
        ratios = [ratios[i] for i in order]
        means = [means[i] for i in order]
        stds = [stds[i] for i in order]
        line, = curve_ax.plot(ratios, means, marker="o", label=label)
        curve_ax.fill_between(
            ratios,
            [m - s for m, s in zip(means, stds)],
            [m + s for m, s in zip(means, stds)],
            color=line.get_color(),
            alpha=0.15,
        )
        if 100.0 in ratios and 20.0 in ratios:
            full = means[ratios.index(100.0)]
            low = means[ratios.index(20.0)]
            retention_rows.append((label, 100.0 * low / full if full > 0 else 0.0, full, low))

    curve_ax.set_xlabel("Training ratio (%)")
    curve_ax.set_ylabel(args.metric)
    curve_ax.set_title("Learning curves with std band")
    curve_ax.grid(alpha=0.3)
    curve_ax.legend()

    if retention_rows:
        retention_rows.sort(key=lambda item: item[1], reverse=True)
        labels = [item[0] for item in retention_rows]
        retentions = [item[1] for item in retention_rows]
        bars = scatter_ax.barh(labels, retentions, color="steelblue")
        for bar, value in zip(bars, retentions):
            scatter_ax.text(
                value + 0.5,
                bar.get_y() + bar.get_height() / 2.0,
                f"{value:.1f}%",
                va="center",
            )
        scatter_ax.set_xlabel("20% / 100% retention rate (%)")
        scatter_ax.set_title("Low-resource retention")
        scatter_ax.grid(axis="x", alpha=0.3)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)


if __name__ == "__main__":
    main()
