"""Configuration loading from YAML/JSON files."""

from __future__ import annotations

import ast
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency
    yaml = None

from .schemas import ExperimentConfig, ModelConfig, NegativeSamplerConfig


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    data = _load_mapping(path)
    model = ModelConfig(**data.pop("model", {}))
    negative_sampling = NegativeSamplerConfig(**data.pop("negative_sampling", {}))
    return ExperimentConfig(model=model, negative_sampling=negative_sampling, **data)


def dump_experiment_config(config: ExperimentConfig) -> dict[str, Any]:
    return {
        "name": config.name,
        "dataset_name": config.dataset_name,
        "raw_dir": config.raw_dir,
        "processed_dir": config.processed_dir,
        "bie_path": config.bie_path,
        "bie_ordered_keys": config.bie_ordered_keys,
        "ontology_keys": config.ontology_keys,
        "delimiter": config.delimiter,
        "seed": config.seed,
        "num_epochs": config.num_epochs,
        "micro_batch_size": config.micro_batch_size,
        "eval_batch_size": config.eval_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "num_workers": config.num_workers,
        "learning_rate": config.learning_rate,
        "warmup_ratio": config.warmup_ratio,
        "early_stopping_patience": config.early_stopping_patience,
        "amp_enabled": config.amp_enabled,
        "enable_data_parallel": config.enable_data_parallel,
        "candidate_chunk_size": config.candidate_chunk_size,
        "log_every_n_steps": config.log_every_n_steps,
        "output_dir": config.output_dir,
        "log_dir": config.log_dir,
        "checkpoint_dir": config.checkpoint_dir,
        "run_train": config.run_train,
        "run_eval": config.run_eval,
        "baseline_reference_path": config.baseline_reference_path,
        "model": asdict(config.model),
        "negative_sampling": asdict(config.negative_sampling),
        "metadata": config.metadata,
    }


def _load_mapping(path: str | Path) -> dict[str, Any]:
    payload = Path(path).read_text(encoding="utf-8")
    suffix = Path(path).suffix.lower()
    if suffix in {".yaml", ".yml"}:
        if yaml is None:
            return _mini_yaml_load(payload)
        return yaml.safe_load(payload)
    return json.loads(payload)


def _mini_yaml_load(payload: str) -> dict[str, Any]:
    lines = []
    for raw_line in payload.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        lines.append(raw_line.rstrip("\n"))

    def parse_mapping(index: int, indent: int) -> tuple[dict[str, Any], int]:
        mapping: dict[str, Any] = {}
        while index < len(lines):
            line = lines[index]
            curr_indent = len(line) - len(line.lstrip(" "))
            if curr_indent < indent:
                break
            if curr_indent != indent:
                raise ValueError(f"Invalid indentation in config line: {line}")
            stripped = line.strip()
            key, _, rest = stripped.partition(":")
            if not _:
                raise ValueError(f"Invalid YAML mapping line: {line}")
            if rest.strip():
                mapping[key] = _parse_scalar(rest.strip())
                index += 1
                continue

            if index + 1 >= len(lines):
                mapping[key] = {}
                index += 1
                continue
            next_line = lines[index + 1]
            next_indent = len(next_line) - len(next_line.lstrip(" "))
            if next_indent <= curr_indent:
                mapping[key] = {}
                index += 1
                continue
            if next_line.strip().startswith("- "):
                value, index = parse_list(index + 1, next_indent)
            else:
                value, index = parse_mapping(index + 1, next_indent)
            mapping[key] = value
        return mapping, index

    def parse_list(index: int, indent: int) -> tuple[list[Any], int]:
        items: list[Any] = []
        while index < len(lines):
            line = lines[index]
            curr_indent = len(line) - len(line.lstrip(" "))
            if curr_indent < indent:
                break
            if curr_indent != indent:
                raise ValueError(f"Invalid list indentation in config line: {line}")
            stripped = line.strip()
            if not stripped.startswith("- "):
                break
            content = stripped[2:].strip()
            if content:
                items.append(_parse_scalar(content))
                index += 1
            else:
                value, index = parse_mapping(index + 1, indent + 2)
                items.append(value)
        return items, index

    parsed, final_index = parse_mapping(0, 0)
    if final_index != len(lines):
        raise ValueError("Mini YAML parser did not consume the entire file.")
    return parsed


def _parse_scalar(value: str) -> Any:
    lower = value.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if lower in {"null", "none"}:
        return None
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part.strip()) for part in inner.split(",")]
    if value.startswith(("'", '"')) and value.endswith(("'", '"')):
        return ast.literal_eval(value)
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value
