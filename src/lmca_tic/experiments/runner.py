"""Experiment orchestration and result aggregation."""

from __future__ import annotations

import csv
import math
import shutil
from copy import deepcopy
from pathlib import Path
from statistics import mean, stdev
from typing import Iterable

from lmca_tic.config.loader import dump_experiment_config, load_experiment_config
from lmca_tic.config.schemas import ExperimentConfig
from lmca_tic.data.dataset import LocalProcessedDataset
from lmca_tic.data.preprocess import LocalTKGPreprocessor
from lmca_tic.evaluation.filtered import FilteredEvaluator
from lmca_tic.training.trainer import LMCATICTrainer
from lmca_tic.utils.io import ensure_dir, read_json, read_jsonl, write_json, write_jsonl


DEFAULT_SEEDS = [42, 123, 456, 789, 1024]
METRIC_KEYS = ["MRR", "Hits@1", "Hits@3", "Hits@10"]

SUITES: dict[str, list[str]] = {
    "main": [
        "configs/experiments/full_icews14.yaml",
        "configs/experiments/full_icews05_15.yaml",
    ],
    "ablation": [
        "configs/experiments/full_icews14.yaml",
        "configs/experiments/ablation_wo_llm_icews14.yaml",
        "configs/experiments/ablation_wo_tcn_icews14.yaml",
        "configs/experiments/ablation_wo_tgn_icews14.yaml",
        "configs/experiments/ablation_wo_gate_icews14.yaml",
    ],
    "negative": [
        "configs/experiments/negative_random_icews14.yaml",
        "configs/experiments/negative_contrastive_icews14.yaml",
        "configs/experiments/negative_ontology_icews14.yaml",
    ],
    "micro": [
        "configs/experiments/micro_v1_gs_icews14.yaml",
        "configs/experiments/micro_v2_ni_icews14.yaml",
        "configs/experiments/micro_v3_sl_icews14.yaml",
        "configs/experiments/micro_v4_gs_ni_sl_icews14.yaml",
        "configs/experiments/micro_ours_ni_sl_icews14.yaml",
    ],
}


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


def run_seeded_suite(
    config_paths: Iterable[str | Path],
    seeds: Iterable[int] = DEFAULT_SEEDS,
    output_root: str | Path = "outputs/experiments",
    smoke: bool = False,
) -> dict[str, object]:
    """Run each config for all seeds and aggregate mean/std metrics."""

    output_root_path = ensure_dir(output_root)
    all_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for config_path in config_paths:
        base_config = load_experiment_config(config_path)
        run_rows = _run_config_for_seeds(base_config, seeds, output_root_path, smoke)
        all_rows.extend(run_rows)
        summary = summarize_metric_rows(run_rows, group_name=base_config.name, dataset=base_config.dataset_name)
        summaries.append(summary)
        _write_csv(output_root_path / base_config.name / "per_seed_metrics.csv", run_rows)
        write_json(output_root_path / base_config.name / "summary.json", summary)

    _write_csv(output_root_path / "per_seed_metrics.csv", all_rows)
    _write_csv(output_root_path / "summary.csv", summaries)
    write_json(output_root_path / "summary.json", summaries)
    return {"runs": all_rows, "summary": summaries}


def run_window_sensitivity(
    base_config_path: str | Path = "configs/experiments/full_icews14.yaml",
    windows: Iterable[int] = (3, 7, 14, 21, 30),
    seeds: Iterable[int] = DEFAULT_SEEDS,
    output_root: str | Path = "outputs/experiments/window_sensitivity",
    smoke: bool = False,
) -> dict[str, object]:
    """Run local history window width sensitivity experiments."""

    base_config = load_experiment_config(base_config_path)
    configs = []
    for window in windows:
        config = deepcopy(base_config)
        config.name = f"{base_config.name}_w{int(window)}d"
        config.model.tgn_time_window_days = int(window)
        configs.append(config)
    return run_seeded_config_objects(configs, seeds=seeds, output_root=output_root, smoke=smoke)


def run_train_ratio_sensitivity(
    base_config_path: str | Path = "configs/experiments/full_icews14.yaml",
    ratios: Iterable[float] = (0.2, 0.4, 0.6, 0.8, 1.0),
    seeds: Iterable[int] = DEFAULT_SEEDS,
    output_root: str | Path = "outputs/experiments/train_ratio",
    raw_output_root: str | Path = "data/derived/train_ratio",
    smoke: bool = False,
) -> dict[str, object]:
    """Run training-data scale sensitivity experiments.

    Generated raw split directories keep valid/test unchanged and truncate
    train.txt by chronological prefix, leaving the original dataset untouched.
    """

    base_config = load_experiment_config(base_config_path)
    configs = []
    for ratio in ratios:
        ratio_value = float(ratio)
        suffix = f"train{int(round(ratio_value * 100))}p"
        raw_dir = materialize_train_ratio_split(
            raw_dir=base_config.raw_dir,
            output_dir=Path(raw_output_root) / suffix,
            ratio=ratio_value,
        )
        config = deepcopy(base_config)
        config.name = f"{base_config.name}_{suffix}"
        config.raw_dir = str(raw_dir)
        configs.append(config)
    return run_seeded_config_objects(configs, seeds=seeds, output_root=output_root, smoke=smoke)


def evaluate_history_chain_sensitivity(
    config_path: str | Path,
    checkpoint_dir: str | Path | None = None,
    lengths: Iterable[int] = (1, 3, 5, 7, 10),
    output_root: str | Path = "outputs/experiments/history_chain",
    split: str = "test",
    checkpoint_name: str = "best.pt",
) -> dict[str, object]:
    """Evaluate a trained model after truncating visible local histories.

    This implements the continuous-history-chain sensitivity experiment on top
    of the processed artifacts. Each variant is written to a derived processed
    directory and the original processed data remains untouched.
    """

    base_config = load_experiment_config(config_path)
    if checkpoint_dir is not None:
        base_config.checkpoint_dir = str(checkpoint_dir)
    output_root_path = ensure_dir(output_root)
    rows: list[dict[str, object]] = []
    for length in lengths:
        variant_dir = output_root_path / "processed" / f"history_{int(length)}"
        materialize_processed_history_variant(base_config.processed_dir, variant_dir, max_history=int(length))
        config = deepcopy(base_config)
        config.processed_dir = str(variant_dir)
        config.output_dir = str(output_root_path / f"history_{int(length)}")
        trainer = LMCATICTrainer(config=config, smoke_mode=config.metadata.get("smoke_mode", False))
        metrics = trainer.evaluate(split=split, checkpoint_name=checkpoint_name)
        rows.append({"history_length": int(length), **_select_metrics(metrics), "output_dir": config.output_dir})
    _write_csv(output_root_path / "summary.csv", rows)
    write_json(output_root_path / "summary.json", rows)
    return {"runs": rows}


def evaluate_noise_sensitivity(
    config_path: str | Path,
    checkpoint_dir: str | Path | None = None,
    rates: Iterable[float] = (0.0, 0.1, 0.2, 0.3),
    output_root: str | Path = "outputs/experiments/noise_sensitivity",
    split: str = "test",
    checkpoint_name: str = "best.pt",
    seed: int = 42,
) -> dict[str, object]:
    """Evaluate structural-noise robustness with processed-neighbor variants.

    The current processed format stores sampled neighbor ids rather than full
    relation-bearing history edges. Noise is therefore injected by replacing a
    rate-controlled share of visible neighbor ids with random entities.
    """

    base_config = load_experiment_config(config_path)
    if checkpoint_dir is not None:
        base_config.checkpoint_dir = str(checkpoint_dir)
    output_root_path = ensure_dir(output_root)
    rows: list[dict[str, object]] = []
    for rate in rates:
        rate_value = float(rate)
        suffix = f"noise_{int(round(rate_value * 100))}p"
        variant_dir = output_root_path / "processed" / suffix
        materialize_processed_noise_variant(
            base_config.processed_dir,
            variant_dir,
            noise_rate=rate_value,
            seed=seed,
        )
        config = deepcopy(base_config)
        config.processed_dir = str(variant_dir)
        config.output_dir = str(output_root_path / suffix)
        trainer = LMCATICTrainer(config=config, smoke_mode=config.metadata.get("smoke_mode", False))
        metrics = trainer.evaluate(split=split, checkpoint_name=checkpoint_name)
        rows.append({"noise_rate": rate_value, **_select_metrics(metrics), "output_dir": config.output_dir})
    _write_csv(output_root_path / "summary.csv", rows)
    write_json(output_root_path / "summary.json", rows)
    return {"runs": rows}


def run_seeded_config_objects(
    configs: Iterable[ExperimentConfig],
    seeds: Iterable[int] = DEFAULT_SEEDS,
    output_root: str | Path = "outputs/experiments",
    smoke: bool = False,
) -> dict[str, object]:
    output_root_path = ensure_dir(output_root)
    all_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for base_config in configs:
        run_rows = _run_config_for_seeds(base_config, seeds, output_root_path, smoke)
        all_rows.extend(run_rows)
        summary = summarize_metric_rows(run_rows, group_name=base_config.name, dataset=base_config.dataset_name)
        summaries.append(summary)
        _write_csv(output_root_path / base_config.name / "per_seed_metrics.csv", run_rows)
        write_json(output_root_path / base_config.name / "summary.json", summary)
    _write_csv(output_root_path / "per_seed_metrics.csv", all_rows)
    _write_csv(output_root_path / "summary.csv", summaries)
    write_json(output_root_path / "summary.json", summaries)
    return {"runs": all_rows, "summary": summaries}


def evaluate_prediction_subset(
    processed_dir: str | Path,
    predictions_path: str | Path,
    output_path: str | Path,
    split: str = "test",
    subset: str = "all",
    history_shot: int | None = None,
) -> dict[str, object]:
    """Recompute filtered metrics on an existing prediction subset.

    ``history_shot`` uses processed local neighbor counts as the available
    history proxy. It keeps samples where both subject and object visible
    histories are no larger than the requested shot value.
    """

    processed_dir_path = Path(processed_dir)
    dataset = LocalProcessedDataset(processed_dir_path, split)
    predictions = read_jsonl(predictions_path)
    if len(dataset.samples) != len(predictions):
        raise ValueError(
            f"Prediction/sample length mismatch: {len(predictions)} predictions vs {len(dataset.samples)} samples."
        )
    selected_predictions = []
    for sample, prediction in zip(dataset.samples, predictions):
        if subset == "inductive" and not sample.quadruple.is_inductive:
            continue
        if subset == "transductive" and sample.quadruple.is_inductive:
            continue
        if subset not in {"all", "inductive", "transductive"}:
            raise ValueError(f"Unsupported subset: {subset}")
        if history_shot is not None:
            subject_history = len(sample.subject_neighbors)
            object_history = len(sample.object_neighbors)
            if max(subject_history, object_history) > history_shot:
                continue
        selected_predictions.append(prediction)

    filtered_targets = read_json(processed_dir_path / "filtered_targets.json")
    metrics = FilteredEvaluator(filtered_targets).evaluate(selected_predictions).to_dict()
    payload = {
        "split": split,
        "subset": subset,
        "history_shot": history_shot,
        "num_samples": len(selected_predictions),
        **metrics,
    }
    write_json(output_path, payload)
    return payload


def aggregate_with_reference(
    reference_path: str | Path,
    summary_path: str | Path,
    output_path: str | Path,
) -> list[dict[str, object]]:
    """Merge external baseline references with generated experiment summaries."""

    reference_rows = _read_table(reference_path)
    summary_rows = _read_table(summary_path)
    merged = reference_rows + summary_rows
    _write_csv(output_path, merged)
    write_json(Path(output_path).with_suffix(".json"), merged)
    return merged


def materialize_train_ratio_split(raw_dir: str | Path, output_dir: str | Path, ratio: float) -> Path:
    if ratio <= 0.0 or ratio > 1.0:
        raise ValueError(f"ratio must be in (0, 1], got {ratio}")
    source = Path(raw_dir)
    target = ensure_dir(output_dir)
    train_lines = _read_lines(source / "train.txt")
    keep = max(1, math.ceil(len(train_lines) * ratio))
    (target / "train.txt").write_text("".join(train_lines[:keep]), encoding="utf-8")
    for split in ("valid", "test"):
        shutil.copyfile(source / f"{split}.txt", target / f"{split}.txt")
    return target


def materialize_processed_history_variant(
    processed_dir: str | Path,
    output_dir: str | Path,
    max_history: int,
) -> Path:
    if max_history < 0:
        raise ValueError(f"max_history must be non-negative, got {max_history}")
    source = Path(processed_dir)
    target = _copy_processed_metadata(source, output_dir)
    for split in ("train", "valid", "test"):
        rows = read_jsonl(source / f"{split}.jsonl")
        for row in rows:
            _truncate_sample_history(row, max_history=max_history)
        write_jsonl(target / f"{split}.jsonl", rows)
    return target


def materialize_processed_noise_variant(
    processed_dir: str | Path,
    output_dir: str | Path,
    noise_rate: float,
    seed: int = 42,
) -> Path:
    if noise_rate < 0.0 or noise_rate > 1.0:
        raise ValueError(f"noise_rate must be in [0, 1], got {noise_rate}")
    import random

    source = Path(processed_dir)
    target = _copy_processed_metadata(source, output_dir)
    entities = list(read_json(source / "entities.json").keys())
    rng = random.Random(seed)
    for split in ("train", "valid", "test"):
        rows = read_jsonl(source / f"{split}.jsonl")
        for row in rows:
            _inject_neighbor_noise(row, entities=entities, noise_rate=noise_rate, rng=rng)
        write_jsonl(target / f"{split}.jsonl", rows)
    return target


def prepare_seeded_config(base_config: ExperimentConfig, seed: int, output_root: str | Path) -> ExperimentConfig:
    """Return an isolated config for one seed without mutating the base config."""

    output_root_path = Path(output_root)
    config = deepcopy(base_config)
    config.seed = int(seed)
    run_name = f"{base_config.name}_seed{seed}"
    config.output_dir = str(output_root_path / base_config.name / f"seed_{seed}" / "outputs")
    config.log_dir = str(output_root_path / base_config.name / f"seed_{seed}" / "logs")
    config.checkpoint_dir = str(output_root_path / base_config.name / f"seed_{seed}" / "checkpoints")
    config.processed_dir = str(output_root_path / "_processed" / run_name)
    config.metadata = {**config.metadata, "experiment_run_name": run_name}
    return config


def summarize_metric_rows(
    rows: list[dict[str, object]],
    group_name: str,
    dataset: str | None = None,
) -> dict[str, object]:
    summary: dict[str, object] = {"experiment": group_name, "num_runs": len(rows)}
    if dataset is not None:
        summary["dataset"] = dataset
    for metric in METRIC_KEYS:
        values = [float(row[metric]) for row in rows if metric in row]
        summary[f"{metric}_mean"] = mean(values) if values else 0.0
        summary[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
    return summary


def dump_prepared_config(config: ExperimentConfig, output_path: str | Path) -> None:
    ensure_dir(Path(output_path).parent)
    write_json(output_path, dump_experiment_config(config))


def _run_config_for_seeds(
    base_config: ExperimentConfig,
    seeds: Iterable[int],
    output_root: Path,
    smoke: bool,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for seed in seeds:
        config = prepare_seeded_config(base_config, seed=seed, output_root=output_root)
        LocalTKGPreprocessor(config).run()
        trainer = LMCATICTrainer(config=config, smoke_mode=smoke or config.metadata.get("smoke_mode", False))
        metrics = trainer.train()
        rows.append(
            {
                "experiment": base_config.name,
                "dataset": base_config.dataset_name,
                "seed": int(seed),
                **_select_metrics(metrics),
                "output_dir": config.output_dir,
                "checkpoint_dir": config.checkpoint_dir,
            }
        )
    return rows


def _select_metrics(metrics: dict[str, float]) -> dict[str, float]:
    return {key: float(metrics.get(key, 0.0)) for key in METRIC_KEYS}


def _read_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return handle.readlines()


def _read_table(path: str | Path) -> list[dict[str, object]]:
    path_obj = Path(path)
    if path_obj.suffix.lower() == ".json":
        payload = read_json(path_obj)
        if isinstance(payload, list):
            return [dict(row) for row in payload]
        return [dict(payload)]
    return _read_csv(path_obj)


def _copy_processed_metadata(source: Path, output_dir: str | Path) -> Path:
    target = ensure_dir(output_dir)
    for filename in (
        "entities.json",
        "relations.json",
        "filtered_targets.json",
        "relation_frequency.json",
        "manifest.json",
    ):
        src = source / filename
        if src.exists():
            shutil.copyfile(src, target / filename)
    return target


def _truncate_sample_history(row: dict[str, object], max_history: int) -> None:
    for key in ("subject_neighbors", "object_neighbors"):
        row[key] = list(row.get(key, []))[-max_history:] if max_history else []
    extra = dict(row.get("extra", {}))
    for key in ("subject_neighbor_deltas", "object_neighbor_deltas"):
        extra[key] = list(extra.get(key, []))[-max_history:] if max_history else []
    row["extra"] = extra


def _inject_neighbor_noise(row: dict[str, object], entities: list[str], noise_rate: float, rng) -> None:
    if not entities or noise_rate <= 0.0:
        return
    for key in ("subject_neighbors", "object_neighbors"):
        neighbors = list(row.get(key, []))
        for index, neighbor in enumerate(neighbors):
            if rng.random() < noise_rate:
                replacement = rng.choice(entities)
                if len(entities) > 1:
                    while replacement == neighbor:
                        replacement = rng.choice(entities)
                neighbors[index] = replacement
        row[key] = neighbors


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
    path_obj = Path(path)
    ensure_dir(path_obj.parent)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path_obj.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
