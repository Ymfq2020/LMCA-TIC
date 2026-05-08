"""Aggregate Chapter 3 result tables into one CSV per paper table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev

METRIC_KEYS = ["MRR", "Hits@1", "Hits@3", "Hits@10"]


def load_summary(path: Path) -> list[dict[str, object]]:
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def collect_per_seed(rows: list[dict[str, object]]) -> dict[str, dict[str, list[float]]]:
    bucket: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        experiment = str(row.get("experiment"))
        bucket.setdefault(experiment, {key: [] for key in METRIC_KEYS})
        for key in METRIC_KEYS:
            value = row.get(key)
            if value is None or value == "":
                continue
            bucket[experiment][key].append(float(value))
    return bucket


def summarize(rows: dict[str, list[float]]) -> dict[str, float]:
    summary: dict[str, float] = {}
    for key, values in rows.items():
        summary[f"{key}_mean"] = mean(values) if values else 0.0
        summary[f"{key}_std"] = stdev(values) if len(values) > 1 else 0.0
    return summary


def aggregate_main_table(per_seed_path: Path, reference_path: Path | None, output_path: Path) -> None:
    bucket = collect_per_seed(load_summary(per_seed_path))
    rows: list[dict[str, object]] = []
    if reference_path and reference_path.exists():
        for row in load_summary(reference_path):
            rows.append({**row, "source": "reference"})
    for experiment, metrics in bucket.items():
        summary = summarize(metrics)
        rows.append({"experiment": experiment, "source": "lmca_tic", **summary})
    write_csv(output_path, rows)


def aggregate_simple(summary_path: Path, output_path: Path) -> None:
    rows = load_summary(summary_path)
    write_csv(output_path, rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-per-seed")
    parser.add_argument("--main-reference")
    parser.add_argument("--ablation-summary")
    parser.add_argument("--negative-summary")
    parser.add_argument("--micro-summary")
    parser.add_argument("--rho-summary")
    parser.add_argument("--backbone-summary")
    parser.add_argument("--window-summary")
    parser.add_argument("--train-ratio-summary")
    parser.add_argument("--history-chain-summary")
    parser.add_argument("--noise-summary")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.main_per_seed:
        aggregate_main_table(
            Path(args.main_per_seed),
            Path(args.main_reference) if args.main_reference else None,
            output_dir / "table_3_3_main.csv",
        )
    for arg_name, table_name in [
        ("ablation_summary", "table_3_9_ablation.csv"),
        ("negative_summary", "table_3_13_negative.csv"),
        ("micro_summary", "table_3_12_micro.csv"),
        ("rho_summary", "table_3_14_rho.csv"),
        ("backbone_summary", "table_3_10_backbone.csv"),
        ("window_summary", "table_3_8_window.csv"),
        ("train_ratio_summary", "table_3_6_train_ratio.csv"),
        ("history_chain_summary", "table_3_7_history_chain.csv"),
        ("noise_summary", "table_3_11_noise.csv"),
    ]:
        path_value = getattr(args, arg_name)
        if path_value:
            aggregate_simple(Path(path_value), output_dir / table_name)


if __name__ == "__main__":
    main()
