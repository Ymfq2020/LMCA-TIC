"""Figure 3-6: compute load vs MRR scatter for micro ablation variants.

The compute load is taken from `step_time_sec * peak_memory` reported in
each variant's `train_history.jsonl` and normalised against the heaviest
variant. MRR is read from the variant's `test_metrics.json`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_runtime(history_path: Path) -> tuple[float, float]:
    if not history_path.exists():
        return 0.0, 0.0
    rows = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        return 0.0, 0.0
    avg_step = sum(float(row.get("step_time_sec", 0.0)) for row in rows) / len(rows)
    peak_mem = max(float(row.get("peak_memory", 0.0)) for row in rows)
    return avg_step, peak_mem


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        nargs="+",
        required=True,
        help="Repeated 'label:metrics_path:history_path' triples.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--metric-key", default="MRR")
    args = parser.parse_args()

    items: list[dict[str, object]] = []
    for entry in args.variant:
        parts = entry.split(":")
        if len(parts) != 3:
            raise SystemExit(f"--variant expects label:metrics_path:history_path, got {entry}")
        label, metrics_path, history_path = parts
        metrics = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
        avg_step, peak_mem = load_runtime(Path(history_path))
        items.append(
            {
                "label": label,
                "metric": float(metrics.get(args.metric_key, 0.0)),
                "compute": avg_step * (peak_mem or 1.0),
            }
        )

    if not items:
        raise SystemExit("No variants supplied")
    max_compute = max(item["compute"] for item in items) or 1.0
    for item in items:
        item["compute_norm"] = 100.0 * item["compute"] / max_compute

    fig, ax = plt.subplots(figsize=(9, 5))
    xs = [item["compute_norm"] for item in items]
    ys = [item["metric"] for item in items]
    ax.scatter(xs, ys, s=140)
    for item in items:
        ax.annotate(
            item["label"],
            xy=(item["compute_norm"], item["metric"]),
            xytext=(8, 6),
            textcoords="offset points",
        )
    ax.set_xlabel("Compute load (relative, %)")
    ax.set_ylabel(args.metric_key)
    ax.set_title("Compute–accuracy trade-off")
    ax.grid(alpha=0.3)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)


if __name__ == "__main__":
    main()
