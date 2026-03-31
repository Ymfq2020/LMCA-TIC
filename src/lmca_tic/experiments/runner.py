"""Experiment orchestration and result aggregation."""

from __future__ import annotations

import csv
from pathlib import Path

from lmca_tic.config.loader import load_experiment_config
from lmca_tic.data.preprocess import LocalTKGPreprocessor
from lmca_tic.training.trainer import LMCATICTrainer
from lmca_tic.utils.io import read_json, write_json


def run_experiment_suite(config_paths: list[str]) -> dict[str, object]:
    results: list[dict[str, object]] = []
    for config_path in config_paths:
        config = load_experiment_config(config_path)
        LocalTKGPreprocessor(config).run()
        trainer = LMCATICTrainer(config=config, smoke_mode=config.metadata.get("smoke_mode", False))
        metrics = trainer.train()
        results.append({"name": config.name, **metrics, "output_dir": config.output_dir})
    if results:
        output_dir = Path(results[0]["output_dir"]).parent
        _write_csv(output_dir / "suite_metrics.csv", results)
    return {"runs": results}


def merge_reference_baselines(reference_path: str, model_metrics_path: str, output_path: str) -> None:
    reference = _read_csv(reference_path)
    metrics = [read_json(model_metrics_path)]
    merged = reference + metrics
    _write_csv(output_path, merged)
    write_json(Path(output_path).with_suffix(".json"), merged)


def _read_csv(path: str | Path) -> list[dict[str, object]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: str | Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
