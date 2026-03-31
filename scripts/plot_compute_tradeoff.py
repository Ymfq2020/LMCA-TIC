"""Plot compute-load vs accuracy trade-off."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True, nargs="+")
    parser.add_argument("--labels", required=True, nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = []
    for label, metric_path in zip(args.labels, args.metrics):
        row = _read_first_row(metric_path)
        row["label"] = label
        row["compute_load"] = float(row.get("step_time_sec", 1.0)) * max(float(row.get("peak_memory", 1.0)), 1.0)
        rows.append(row)
    max_compute = max(row["compute_load"] for row in rows) or 1.0
    for row in rows:
        row["compute_load_norm"] = row["compute_load"] / max_compute

    plt.figure(figsize=(8, 5))
    plt.scatter([row["compute_load_norm"] for row in rows], [float(row["MRR"]) for row in rows])
    for row in rows:
        plt.text(row["compute_load_norm"], float(row["MRR"]), row["label"])
    plt.xlabel("Normalized Compute Load")
    plt.ylabel("MRR")
    plt.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=200)


def _read_first_row(path: str) -> dict[str, object]:
    if path.endswith(".json"):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return payload[0]
        return payload
    with Path(path).open("r", encoding="utf-8") as handle:
        return next(csv.DictReader(handle))


if __name__ == "__main__":
    main()
