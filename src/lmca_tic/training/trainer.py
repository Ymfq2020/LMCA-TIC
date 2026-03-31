"""Training orchestration for LMCA-TIC."""

from __future__ import annotations

import math
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

try:
    import torch
    from torch import nn
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    nn = SimpleNamespace(Module=object, DataParallel=object)
    AdamW = None
    DataLoader = None

from lmca_tic.config.loader import dump_experiment_config
from lmca_tic.config.schemas import ExperimentConfig
from lmca_tic.data.dataset import LocalProcessedDataset
from lmca_tic.evaluation.filtered import FilteredEvaluator
from lmca_tic.kgist.miner import KGISTSummaryMiner, NegativeErrorScorer
from lmca_tic.models.model import LMCATICModel
from lmca_tic.training.negative_sampling import HardNegativeSampler
from lmca_tic.training.scheduler import build_warmup_scheduler
from lmca_tic.utils.deps import require_dependency
from lmca_tic.utils.io import ensure_dir, read_json, write_json, write_jsonl
from lmca_tic.utils.logging import build_logger, capture_manifest, write_manifest
from lmca_tic.utils.seed import set_global_seed


def _identity_collate(batch):
    return batch


class LMCATICTrainer:
    def __init__(self, config: ExperimentConfig, smoke_mode: bool = False) -> None:
        require_dependency(torch, "torch")
        require_dependency(DataLoader, "torch")
        self.config = config
        self.smoke_mode = smoke_mode
        set_global_seed(config.seed)
        self.output_dir = ensure_dir(config.output_dir)
        self.checkpoint_dir = ensure_dir(config.checkpoint_dir)
        self.logger = build_logger(config.log_dir)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.use_amp = bool(config.amp_enabled and torch.cuda.is_available())
        self.entities = read_json(Path(config.processed_dir) / "entities.json")
        self.relation_manifest = read_json(Path(config.processed_dir) / "relations.json")
        self.filtered_targets = read_json(Path(config.processed_dir) / "filtered_targets.json")
        self.entity_to_idx = {entity_id: idx for idx, entity_id in enumerate(self.entities)}
        self.entity_prompts = {entity_id: self._entity_prompt(entity_id) for entity_id in self.entities}
        self.entity_types = {entity_id: self._entity_types(entity_id) for entity_id in self.entities}
        self.relation_to_idx = {
            relation: idx
            for idx, relation in enumerate(
                self.relation_manifest["relations"] + self.relation_manifest["inverse_relations"]
            )
        }
        self.train_dataset = LocalProcessedDataset(config.processed_dir, "train")
        self.valid_dataset = LocalProcessedDataset(config.processed_dir, "valid")
        self.test_dataset = LocalProcessedDataset(config.processed_dir, "test")
        self.entity_neighbor_cache, self.entity_delta_cache = self._build_entity_context_cache(self.train_dataset.samples)
        artifact = KGISTSummaryMiner(min_support=1).mine(self.train_dataset.samples)
        self.negative_scorer = NegativeErrorScorer(artifact)
        self.negative_sampler = HardNegativeSampler(config.negative_sampling, scorer=self.negative_scorer)
        self.model = LMCATICModel(
            num_entities=len(self.entity_to_idx),
            num_relations=len(self.relation_to_idx),
            config=config.model,
            smoke_mode=smoke_mode,
        ).to(self.device)
        self.enable_data_parallel = bool(
            config.enable_data_parallel
            and torch.cuda.is_available()
            and torch.cuda.device_count() > 1
            and not config.model.use_4bit
            and not smoke_mode
        )
        if self.enable_data_parallel:
            self.model = nn.DataParallel(self.model)
        self.optimizer = AdamW(self._model_module().parameters(), lr=config.learning_rate)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.train_loader = self._build_dataloader(self.train_dataset, batch_size=config.micro_batch_size, shuffle=True)
        self.valid_loader = self._build_dataloader(self.valid_dataset, batch_size=config.eval_batch_size, shuffle=False)
        self.test_loader = self._build_dataloader(self.test_dataset, batch_size=config.eval_batch_size, shuffle=False)
        steps_per_epoch = max(math.ceil(len(self.train_loader) / max(config.gradient_accumulation_steps, 1)), 1)
        self.total_optimization_steps = steps_per_epoch * config.num_epochs
        self.scheduler = build_warmup_scheduler(
            self.optimizer,
            total_steps=self.total_optimization_steps,
            warmup_ratio=config.warmup_ratio,
        )
        write_json(
            self.output_dir / "graph_summary.json",
            {
                "rules": [
                    {
                        "rule_id": rule.rule_id,
                        "support": rule.support,
                        "coverage": rule.coverage,
                        "rule_gain": rule.rule_gain,
                    }
                    for rule in artifact.rules
                ],
                "coverage": artifact.coverage,
                "type_constraints": artifact.type_constraints,
                "rule_gain": artifact.rule_gain,
                "negative_error_weight": artifact.negative_error_weight,
            },
        )

    def train(self) -> dict[str, float]:
        manifest = capture_manifest(
            extra={
                "config": dump_experiment_config(self.config),
                "smoke_mode": self.smoke_mode,
                "device": str(self.device),
                "data_parallel": self.enable_data_parallel,
            }
        )
        write_manifest(self.output_dir / "run_manifest.json", manifest)
        best_mrr = -1.0
        patience = 0
        history: list[dict[str, float]] = []
        optimizer_steps = 0

        for epoch in range(1, self.config.num_epochs + 1):
            start = time.perf_counter()
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            epoch_loss = 0.0
            num_batches = 0
            for batch_index, samples in enumerate(self.train_loader, start=1):
                batch = self._build_batch(samples)
                batch = self._move_batch_to_device(batch)
                with self._autocast():
                    outputs = self.model(batch)
                    loss = self._compute_loss(outputs, batch)
                    scaled_loss = loss / max(self.config.gradient_accumulation_steps, 1)
                if self.use_amp:
                    self.scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                should_step = (
                    batch_index % max(self.config.gradient_accumulation_steps, 1) == 0
                    or batch_index == len(self.train_loader)
                )
                if should_step:
                    if self.use_amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    if self.scheduler is not None:
                        self.scheduler.step()
                    optimizer_steps += 1

                epoch_loss += float(loss.detach().item())
                num_batches += 1

            valid_metrics = self.evaluate(split="valid")
            step_time = time.perf_counter() - start
            record = {
                "epoch": float(epoch),
                "loss": epoch_loss / max(num_batches, 1),
                "valid_mrr": valid_metrics["MRR"],
                "valid_hits@1": valid_metrics["Hits@1"],
                "valid_hits@3": valid_metrics["Hits@3"],
                "valid_hits@10": valid_metrics["Hits@10"],
                "step_time_sec": step_time,
                "samples_per_sec": len(self.train_dataset) / max(step_time, 1e-6),
                "peak_memory": float(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0.0,
                "optimizer_steps": float(optimizer_steps),
                "lr": float(self.optimizer.param_groups[0]["lr"]),
            }
            history.append(record)
            self.logger.info(
                "epoch=%s loss=%.6f valid_mrr=%.4f lr=%.6g",
                epoch,
                record["loss"],
                record["valid_mrr"],
                record["lr"],
            )
            if record["valid_mrr"] > best_mrr:
                best_mrr = record["valid_mrr"]
                patience = 0
                self._save_checkpoint("best.pt")
            else:
                patience += 1
                if patience >= self.config.early_stopping_patience:
                    self.logger.info("early stopping triggered at epoch %s", epoch)
                    break

        write_jsonl(self.output_dir / "train_history.jsonl", history)
        return self.evaluate(split="test", checkpoint_name="best.pt")

    def evaluate(self, split: str = "test", checkpoint_name: str | None = None) -> dict[str, float]:
        if checkpoint_name:
            self._load_checkpoint(checkpoint_name)
        dataset = {
            "valid": self.valid_dataset,
            "test": self.test_dataset,
            "train": self.train_dataset,
        }[split]
        evaluator = FilteredEvaluator(self.filtered_targets)
        predictions: list[dict[str, object]] = []
        for sample in dataset.samples:
            predictions.append(self._predict_sample(sample))
        metrics = evaluator.evaluate(predictions).to_dict()
        write_json(self.output_dir / f"{split}_metrics.json", metrics)
        write_jsonl(self.output_dir / f"{split}_predictions.jsonl", predictions)
        return metrics

    def _build_dataloader(self, dataset, batch_size: int, shuffle: bool):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=_identity_collate,
        )

    def _predict_sample(self, sample) -> dict[str, object]:
        scores: dict[str, float] = {}
        entity_ids = list(self.entities.keys())
        for start in range(0, len(entity_ids), self.config.candidate_chunk_size):
            chunk_entities = entity_ids[start : start + self.config.candidate_chunk_size]
            batch = self._build_batch([sample], forced_negative_candidates=chunk_entities)
            batch = self._move_batch_to_device(batch)
            with torch.no_grad():
                with self._autocast():
                    outputs = self.model(batch)
            chunk_scores = outputs["negative_scores"][0, : len(chunk_entities)].detach().cpu().tolist()
            for entity_id, score in zip(chunk_entities, chunk_scores):
                scores[entity_id] = float(score)
        return {
            "subject": sample.quadruple.subject,
            "relation": sample.quadruple.relation,
            "timestamp": sample.quadruple.timestamp,
            "gold": sample.quadruple.object,
            "scores": scores,
        }

    def _build_batch(self, samples, forced_negative_candidates: list[str] | None = None) -> dict[str, object]:
        relation_histories = torch.tensor([sample.relation_history for sample in samples], dtype=torch.float32)
        subject_ids = torch.tensor([self.entity_to_idx[sample.quadruple.subject] for sample in samples], dtype=torch.long)
        object_ids = torch.tensor([self.entity_to_idx[sample.quadruple.object] for sample in samples], dtype=torch.long)
        relation_ids = torch.tensor([self.relation_to_idx[sample.quadruple.relation] for sample in samples], dtype=torch.long)
        subject_neighbor_ids, subject_neighbor_deltas = self._pad_neighbors(
            [sample.subject_neighbors for sample in samples],
            [sample.extra.get("subject_neighbor_deltas", []) for sample in samples],
        )
        object_neighbor_ids, object_neighbor_deltas = self._pad_neighbors(
            [sample.object_neighbors for sample in samples],
            [sample.extra.get("object_neighbor_deltas", []) for sample in samples],
        )
        negative_candidates = []
        if forced_negative_candidates is None:
            candidate_scores_batch = self._current_candidate_scores_batch(samples)
        else:
            candidate_scores_batch = None
        for sample_index, sample in enumerate(samples):
            if forced_negative_candidates is not None:
                negative_candidates.append(list(forced_negative_candidates))
                continue
            negative_candidates.append(
                self.negative_sampler.sample(
                    positive_object=sample.quadruple.object,
                    relation=sample.quadruple.relation,
                    subject_types=sample.subject_types,
                    candidate_scores=candidate_scores_batch[sample_index],
                    candidate_types=self.entity_types,
                )
            )
        negative_payload = self._build_negative_payload(negative_candidates)
        batch = {
            "subject_prompts": [sample.subject_prompt for sample in samples],
            "positive_object_prompts": [sample.object_prompt for sample in samples],
            "relation_histories": relation_histories,
            "subject_ids": subject_ids,
            "positive_object_ids": object_ids,
            "negative_object_ids": negative_payload["negative_object_ids"],
            "negative_object_prompts_flat": negative_payload["negative_object_prompts_flat"],
            "negative_object_ids_flat": negative_payload["negative_object_ids_flat"],
            "negative_neighbor_ids_flat": negative_payload["negative_neighbor_ids_flat"],
            "negative_neighbor_deltas_flat": negative_payload["negative_neighbor_deltas_flat"],
            "negative_mask": negative_payload["negative_mask"],
            "relation_ids": relation_ids,
            "subject_neighbor_ids": subject_neighbor_ids,
            "subject_neighbor_deltas": subject_neighbor_deltas,
            "object_neighbor_ids": object_neighbor_ids,
            "object_neighbor_deltas": object_neighbor_deltas,
        }
        return self._tokenize_prompt_fields(batch)

    def _tokenize_prompt_fields(self, batch: dict[str, object]) -> dict[str, object]:
        encoder = self._model_module().text_encoder
        batch["subject_prompts"] = encoder.tokenize_texts(batch["subject_prompts"], device=None)
        batch["positive_object_prompts"] = encoder.tokenize_texts(batch["positive_object_prompts"], device=None)
        if batch["negative_object_prompts_flat"]:
            batch["negative_object_prompts_flat"] = encoder.tokenize_texts(batch["negative_object_prompts_flat"], device=None)
        return batch

    def _build_negative_payload(self, negative_candidates: list[list[str]]) -> dict[str, object]:
        max_neg = max((len(candidates) for candidates in negative_candidates), default=0)
        if max_neg == 0:
            return {
                "negative_object_ids": torch.zeros((len(negative_candidates), 0), dtype=torch.long),
                "negative_object_ids_flat": torch.zeros((0,), dtype=torch.long),
                "negative_object_prompts_flat": [],
                "negative_neighbor_ids_flat": torch.zeros((0, 0), dtype=torch.long),
                "negative_neighbor_deltas_flat": torch.zeros((0, 0), dtype=torch.float32),
                "negative_mask": torch.zeros((len(negative_candidates), 0), dtype=torch.bool),
            }
        padded_ids: list[list[int]] = []
        padded_mask: list[list[bool]] = []
        flat_ids: list[int] = []
        flat_prompts: list[str] = []
        neighbor_lists: list[list[str]] = []
        delta_lists: list[list[float]] = []
        for candidates in negative_candidates:
            row_ids: list[int] = []
            row_mask: list[bool] = []
            fill_candidate = candidates[0] if candidates else next(iter(self.entities))
            padded_candidates = list(candidates) + [fill_candidate] * (max_neg - len(candidates))
            for idx, candidate in enumerate(padded_candidates):
                row_ids.append(self.entity_to_idx[candidate])
                row_mask.append(idx < len(candidates))
                flat_ids.append(self.entity_to_idx[candidate])
                flat_prompts.append(self.entity_prompts[candidate])
                neighbor_lists.append(self.entity_neighbor_cache.get(candidate, []))
                delta_lists.append(self.entity_delta_cache.get(candidate, []))
            padded_ids.append(row_ids)
            padded_mask.append(row_mask)
        negative_neighbor_ids, negative_neighbor_deltas = self._pad_neighbors(neighbor_lists, delta_lists)
        return {
            "negative_object_ids": torch.tensor(padded_ids, dtype=torch.long),
            "negative_object_ids_flat": torch.tensor(flat_ids, dtype=torch.long),
            "negative_object_prompts_flat": flat_prompts,
            "negative_neighbor_ids_flat": negative_neighbor_ids,
            "negative_neighbor_deltas_flat": negative_neighbor_deltas,
            "negative_mask": torch.tensor(padded_mask, dtype=torch.bool),
        }

    def _current_candidate_scores_batch(self, samples) -> list[dict[str, float]]:
        model = self._model_module()
        was_training = model.training
        model.eval()
        with torch.no_grad():
            subject_prompts = model.text_encoder.tokenize_texts([sample.subject_prompt for sample in samples], device=self.device)
            relation_histories = torch.tensor([sample.relation_history for sample in samples], dtype=torch.float32, device=self.device)
            subject_ids = torch.tensor([self.entity_to_idx[sample.quadruple.subject] for sample in samples], dtype=torch.long, device=self.device)
            relation_ids = torch.tensor([self.relation_to_idx[sample.quadruple.relation] for sample in samples], dtype=torch.long, device=self.device)
            subject_neighbor_ids, subject_neighbor_deltas = self._pad_neighbors(
                [sample.subject_neighbors for sample in samples],
                [sample.extra.get("subject_neighbor_deltas", []) for sample in samples],
            )
            subject_neighbor_ids = subject_neighbor_ids.to(self.device)
            subject_neighbor_deltas = subject_neighbor_deltas.to(self.device)
            with self._autocast():
                subject_embed, _ = model.encode_entities(
                    prompts=subject_prompts,
                    relation_histories=relation_histories,
                    entity_ids=subject_ids,
                    neighbor_ids=subject_neighbor_ids,
                    neighbor_deltas=subject_neighbor_deltas,
                )
                candidate_score_rows = [dict() for _ in samples]
                entity_ids_all = list(self.entities.keys())
                for start in range(0, len(entity_ids_all), self.config.candidate_chunk_size):
                    entity_chunk = entity_ids_all[start : start + self.config.candidate_chunk_size]
                    chunk_size = len(entity_chunk)
                    flat_prompts = []
                    flat_entity_ids = []
                    flat_relation_histories = []
                    neighbor_lists = []
                    delta_lists = []
                    for sample in samples:
                        for entity_id in entity_chunk:
                            flat_prompts.append(self.entity_prompts[entity_id])
                            flat_entity_ids.append(self.entity_to_idx[entity_id])
                            flat_relation_histories.append(sample.relation_history)
                            neighbor_lists.append(self.entity_neighbor_cache.get(entity_id, []))
                            delta_lists.append(self.entity_delta_cache.get(entity_id, []))
                    tokenized_prompts = model.text_encoder.tokenize_texts(flat_prompts, device=self.device)
                    entity_ids_tensor = torch.tensor(flat_entity_ids, dtype=torch.long, device=self.device)
                    relation_histories_tensor = torch.tensor(flat_relation_histories, dtype=torch.float32, device=self.device)
                    candidate_neighbor_ids, candidate_neighbor_deltas = self._pad_neighbors(neighbor_lists, delta_lists)
                    candidate_neighbor_ids = candidate_neighbor_ids.to(self.device)
                    candidate_neighbor_deltas = candidate_neighbor_deltas.to(self.device)
                    candidate_embed, _ = model.encode_entities(
                        prompts=tokenized_prompts,
                        relation_histories=relation_histories_tensor,
                        entity_ids=entity_ids_tensor,
                        neighbor_ids=candidate_neighbor_ids,
                        neighbor_deltas=candidate_neighbor_deltas,
                    )
                    repeated_subject = subject_embed.repeat_interleave(chunk_size, dim=0)
                    repeated_relations = relation_ids.repeat_interleave(chunk_size, dim=0)
                    scores = model.scorer(repeated_subject, repeated_relations, candidate_embed)
                    score_matrix = scores.reshape(len(samples), chunk_size).detach().cpu()
                    for row_idx, _sample in enumerate(samples):
                        for col_idx, entity_id in enumerate(entity_chunk):
                            candidate_score_rows[row_idx][entity_id] = float(score_matrix[row_idx, col_idx].item())
        if was_training:
            model.train()
        return candidate_score_rows

    def _compute_loss(self, outputs: dict[str, object], batch: dict[str, object]):
        positive_scores = outputs["positive_scores"]
        negative_scores = outputs["negative_scores"]
        positive_labels = torch.ones_like(positive_scores)
        loss = self.loss_fn(positive_scores, positive_labels)
        if negative_scores is not None and batch["negative_mask"].any():
            masked_negative_scores = negative_scores[batch["negative_mask"]]
            negative_labels = torch.zeros_like(masked_negative_scores)
            loss = loss + self.loss_fn(masked_negative_scores, negative_labels)
        return loss

    def _move_batch_to_device(self, payload):
        if isinstance(payload, torch.Tensor):
            return payload.to(self.device, non_blocking=torch.cuda.is_available())
        if isinstance(payload, dict):
            return {key: self._move_batch_to_device(value) for key, value in payload.items()}
        if isinstance(payload, list):
            return [self._move_batch_to_device(value) for value in payload]
        return payload

    def _pad_neighbors(self, neighbor_lists, delta_lists):
        max_neighbors = max((len(items) for items in neighbor_lists), default=0)
        if max_neighbors == 0:
            empty_ids = torch.zeros((len(neighbor_lists), 0), dtype=torch.long)
            empty_deltas = torch.zeros((len(neighbor_lists), 0), dtype=torch.float32)
            return empty_ids, empty_deltas
        padded_ids = []
        padded_deltas = []
        for neighbors, deltas in zip(neighbor_lists, delta_lists):
            ids = [self.entity_to_idx.get(neighbor, 0) for neighbor in neighbors]
            ids += [0] * (max_neighbors - len(ids))
            padded_ids.append(ids)
            padded_deltas.append(list(deltas) + [0.0] * (max_neighbors - len(deltas)))
        return (
            torch.tensor(padded_ids, dtype=torch.long),
            torch.tensor(padded_deltas, dtype=torch.float32),
        )

    def _build_entity_context_cache(self, samples):
        latest_neighbors: dict[str, tuple[int, list[str], list[float]]] = {}
        for sample in samples:
            timestamp = sample.quadruple.timestamp
            for entity_id, neighbors, deltas in (
                (sample.quadruple.subject, sample.subject_neighbors, sample.extra.get("subject_neighbor_deltas", [])),
                (sample.quadruple.object, sample.object_neighbors, sample.extra.get("object_neighbor_deltas", [])),
            ):
                previous = latest_neighbors.get(entity_id)
                if previous is None or timestamp >= previous[0]:
                    latest_neighbors[entity_id] = (timestamp, list(neighbors), list(deltas))
        neighbor_cache = defaultdict(list)
        delta_cache = defaultdict(list)
        for entity_id, (_, neighbors, deltas) in latest_neighbors.items():
            neighbor_cache[entity_id] = neighbors
            delta_cache[entity_id] = deltas
        return neighbor_cache, delta_cache

    def _entity_prompt(self, entity_id: str) -> str:
        record = self.entities[entity_id]
        name = record.get("entity_name", entity_id)
        attrs = record.get("attributes", {})
        if not attrs:
            return f"实体名称：{name}。背景知识：未提供额外背景属性。请提取其核心属性以支持时序知识图谱推理。"
        fragments = "；".join(f"{k}: {v}" for k, v in attrs.items())
        return f"实体名称：{name}。背景知识：{fragments}。请提取其核心属性以支持时序知识图谱推理。"

    def _entity_types(self, entity_id: str) -> tuple[str, ...]:
        record = self.entities[entity_id]
        attrs = record.get("attributes", {})
        values = [
            f"{key}={attrs[key]}"
            for key in self.config.ontology_keys
            if key in attrs and attrs[key]
        ]
        return tuple(values) if values else ("entity_type=UNKNOWN",)

    def _save_checkpoint(self, name: str) -> None:
        torch.save(self._model_module().state_dict(), self.checkpoint_dir / name)

    def _load_checkpoint(self, name: str) -> None:
        path = self.checkpoint_dir / name
        if path.exists():
            self._model_module().load_state_dict(torch.load(path, map_location=self.device))

    def _model_module(self):
        return self.model.module if self.enable_data_parallel else self.model

    def _autocast(self):
        if not self.use_amp:
            return nullcontext()
        return torch.cuda.amp.autocast(dtype=torch.float16)
