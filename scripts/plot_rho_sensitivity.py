"""Figure 3-7: ρ sweep curves for MRR / Hits@1 / Hits@10.

Reads the rho-sensitivity summary CSV produced by ``run_rho_sensitivity``
and plots the three metrics against ρ.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt


RHO_RE = re.compile(r"_rho([0-9p]+)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, help="rho_sensitivity summary CSV")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rhos: list[float] = []
    metrics: dict[str, list[float]] = {"MRR": [], "Hits@1": [], "Hits@10": []}
    with Path(args.summary).open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            match = RHO_RE.search(row.get("experiment", ""))
            if not match:
                continue
            rho = float(match.group(1).replace("p", "."))
            rhos.append(rho)
            for key in metrics:
                metrics[key].append(float(row.get(f"{key}_mean", 0.0)))

    if not rhos:
        raise SystemExit("No rho rows found; check the summary CSV column names.")
    order = sorted(range(len(rhos)), key=lambda i: rhos[i])
    rhos = [rhos[i] for i in order]
    for key in metrics:
        metrics[key] = [metrics[key][i] for i in order]

    fig, ax = plt.subplots(figsize=(9, 5))
    for key, values in metrics.items():
        ax.plot(rhos, values, marker="o", label=key)
    ax.set_xlabel("Summary weight coefficient ρ")
    ax.set_ylabel("Metric (%)")
    ax.set_title("ρ sensitivity")
    ax.grid(alpha=0.3)
    ax.legend()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)


if __name__ == "__main__":
    main()
