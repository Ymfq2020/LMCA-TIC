"""Typed configuration objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class NegativeSamplerConfig:
    mode: str = "ontology_weighted"
    k_recall: int = 256
    n_neg: int = 64
    tau: float = 0.7
    alpha: float = 0.5
    faiss_enabled: bool = False


@dataclass
class ModelConfig:
    llm_name: str = "Qwen/Qwen3-8B"
    smoke_llm_name: str = "distilbert-base-uncased"
    embedding_dim: int = 200
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    tcn_kernel_size: int = 2
    tcn_dilations: list[int] = field(default_factory=lambda: [1, 2, 4, 8])
    tgn_neighbor_size: int = 10
    tgn_time_window_days: int = 30
    tgn_memory_dim: int = 200
    tgn_time_decay_init: float = 0.1
    fusion_hidden_dim: int = 128
    use_llm: bool = True
    use_tcn: bool = True
    use_tgn: bool = True
    use_gate: bool = True
    use_gs: bool = False
    use_ni: bool = True
    use_sl: bool = True
    use_4bit: bool = True


@dataclass
class ExperimentConfig:
    name: str
    dataset_name: str
    raw_dir: str
    processed_dir: str
    bie_path: str | None = None
    bie_ordered_keys: list[str] = field(default_factory=list)
    ontology_keys: list[str] = field(default_factory=lambda: ["entity_type"])
    delimiter: str = "\t"
    seed: int = 42
    num_epochs: int = 100
    micro_batch_size: int = 64
    eval_batch_size: int = 8
    gradient_accumulation_steps: int = 16
    num_workers: int = 0
    learning_rate: float = 1e-3
    warmup_ratio: float = 0.1
    early_stopping_patience: int = 10
    amp_enabled: bool = True
    enable_data_parallel: bool = False
    candidate_chunk_size: int = 64
    output_dir: str = "outputs/default"
    log_dir: str = "logs/default"
    checkpoint_dir: str = "checkpoints/default"
    run_train: bool = True
    run_eval: bool = True
    baseline_reference_path: str | None = None
    model: ModelConfig = field(default_factory=ModelConfig)
    negative_sampling: NegativeSamplerConfig = field(default_factory=NegativeSamplerConfig)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)
