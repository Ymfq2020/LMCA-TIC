"""Training orchestration for LMCA-TIC."""

from __future__ import annotations

import math
import subprocess
import sys
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

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None

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


def _build_progress(iterable, total: int, desc: str):
    if tqdm is None:
        return iterable
    return tqdm(
        iterable,
        total=total,
        desc=desc,
        leave=False,
        dynamic_ncols=True,
        disable=not sys.stderr.isatty(),
    )


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
        self.entity_popularity = self._build_entity_popularity(self.train_dataset.samples) if hasattr(self, "train_dataset") else {}
        self.relation_to_idx = {
            relation: idx
            for idx, relation in enumerate(
                self.relation_manifest["relations"] + self.relation_manifest["inverse_relations"]
            )
        }
        self.train_dataset = LocalProcessedDataset(config.processed_dir, "train")
        self.valid_dataset = LocalProcessedDataset(config.processed_dir, "valid")
        self.test_dataset = LocalProcessedDataset(config.processed_dir, "test")
        self.entity_popularity = self._build_entity_popularity(self.train_dataset.samples)
        self.entities_by_type = self._build_entities_by_type()
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
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
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
        self._log_runtime_diagnostics()
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
            epoch_batches = len(self.train_loader)
            progress = _build_progress(
                enumerate(self.train_loader, start=1),
                total=epoch_batches,
                desc=f"train epoch {epoch}",
            )
            for batch_index, samples in progress:
                batch_start = time.perf_counter()
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
                elapsed = max(time.perf_counter() - start, 1e-6)
                batches_per_sec = num_batches / elapsed
                remaining_batches = max(epoch_batches - batch_index, 0)
                epoch_eta_sec = remaining_batches / max(batches_per_sec, 1e-6)
                batch_step_sec = time.perf_counter() - batch_start
                if tqdm is not None and hasattr(progress, "set_postfix"):
                    progress.set_postfix(
                        loss=f"{epoch_loss / max(num_batches, 1):.4f}",
                        lr=f"{self.optimizer.param_groups[0]['lr']:.2e}",
                        opt_steps=optimizer_steps,
                        step_s=f"{batch_step_sec:.2f}",
                        bps=f"{batches_per_sec:.2f}",
                        eta=f"{epoch_eta_sec / 60:.1f}m",
                    )
                if (
                    optimizer_steps > 0
                    and optimizer_steps % max(self.config.log_every_n_steps, 1) == 0
                    and should_step
                ):
                    gpu_stats = self._query_gpu_stats()
                    self.logger.info(
                        "epoch=%s/%s batch=%s/%s opt_step=%s loss=%.6f lr=%.6g batch_step_sec=%.3f batches_per_sec=%.3f epoch_eta_sec=%.1f gpu_util=%s gpu_mem_mb=%s/%s",
                        epoch,
                        self.config.num_epochs,
                        batch_index,
                        epoch_batches,
                        optimizer_steps,
                        epoch_loss / max(num_batches, 1),
                        self.optimizer.param_groups[0]["lr"],
                        batch_step_sec,
                        batches_per_sec,
                        epoch_eta_sec,
                        gpu_stats["utilization_gpu"],
                        gpu_stats["memory_used_mb"],
                        gpu_stats["memory_total_mb"],
                    )

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
                "epoch=%s loss=%.6f valid_mrr=%.4f lr=%.6g epoch_sec=%.2f peak_mem_mb=%.1f",
                epoch,
                record["loss"],
                record["valid_mrr"],
                record["lr"],
                step_time,
                record["peak_memory"] / (1024 ** 2) if torch.cuda.is_available() else 0.0,
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
        progress = _build_progress(dataset.samples, total=len(dataset.samples), desc=f"eval {split}")
        for sample in progress:
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
            candidate_pools = [self._candidate_pool_for_sample(sample) for sample in samples]
            candidate_scores_batch = self._current_candidate_scores_batch(samples, candidate_pools)
        else:
            candidate_scores_batch = None
            candidate_pools = None
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

    def _current_candidate_scores_batch(self, samples, candidate_pools: list[list[str]]) -> list[dict[str, float]]:
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
                for row_idx, sample in enumerate(samples):
                    pool = candidate_pools[row_idx]
                    if not pool:
                        continue
                    for start in range(0, len(pool), self.config.candidate_chunk_size):
                        entity_chunk = pool[start : start + self.config.candidate_chunk_size]
                        if not entity_chunk:
                            continue
                        tokenized_prompts = model.text_encoder.tokenize_texts(
                            [self.entity_prompts[entity_id] for entity_id in entity_chunk],
                            device=self.device,
                        )
                        entity_ids_tensor = torch.tensor(
                            [self.entity_to_idx[entity_id] for entity_id in entity_chunk],
                            dtype=torch.long,
                            device=self.device,
                        )
                        relation_histories_tensor = torch.tensor(
                            [sample.relation_history for _ in entity_chunk],
                            dtype=torch.float32,
                            device=self.device,
                        )
                        candidate_neighbor_ids, candidate_neighbor_deltas = self._pad_neighbors(
                            [self.entity_neighbor_cache.get(entity_id, []) for entity_id in entity_chunk],
                            [self.entity_delta_cache.get(entity_id, []) for entity_id in entity_chunk],
                        )
                        candidate_neighbor_ids = candidate_neighbor_ids.to(self.device)
                        candidate_neighbor_deltas = candidate_neighbor_deltas.to(self.device)
                        candidate_embed, _ = model.encode_entities(
                            prompts=tokenized_prompts,
                            relation_histories=relation_histories_tensor,
                            entity_ids=entity_ids_tensor,
                            neighbor_ids=candidate_neighbor_ids,
                            neighbor_deltas=candidate_neighbor_deltas,
                        )
                        repeated_subject = subject_embed[row_idx : row_idx + 1].repeat(len(entity_chunk), 1)
                        repeated_relations = relation_ids[row_idx : row_idx + 1].repeat(len(entity_chunk))
                        scores = model.scorer(repeated_subject, repeated_relations, candidate_embed).detach().cpu()
                        for entity_id, score in zip(entity_chunk, scores):
                            candidate_score_rows[row_idx][entity_id] = float(score.item())
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

    def _build_entity_popularity(self, samples):
        popularity = defaultdict(int)
        for sample in samples:
            popularity[sample.quadruple.subject] += 1
            popularity[sample.quadruple.object] += 1
        return popularity

    def _build_entities_by_type(self):
        entities_by_type = defaultdict(list)
        for entity_id, entity_types in self.entity_types.items():
            for entity_type in entity_types:
                entities_by_type[entity_type].append(entity_id)
        for entity_type in entities_by_type:
            entities_by_type[entity_type].sort(key=lambda entity_id: self.entity_popularity.get(entity_id, 0), reverse=True)
        return entities_by_type

    def _candidate_pool_for_sample(self, sample):
        allowed_tail_types = self.negative_scorer.allowed_tail_types(sample.quadruple.relation)
        candidate_entities: list[str] = []
        if allowed_tail_types:
            seen = set()
            for entity_type in allowed_tail_types:
                for entity_id in self.entities_by_type.get(entity_type, []):
                    if entity_id not in seen:
                        candidate_entities.append(entity_id)
                        seen.add(entity_id)
        else:
            candidate_entities = sorted(
                self.entities.keys(),
                key=lambda entity_id: self.entity_popularity.get(entity_id, 0),
                reverse=True,
            )
        pool_size = min(
            len(candidate_entities),
            max(
                self.config.negative_sampling.k_recall * 4,
                self.config.negative_sampling.n_neg * 16,
                128,
            ),
        )
        pool = [
            entity_id
            for entity_id in candidate_entities[:pool_size]
            if entity_id != sample.quadruple.object
        ]
        if not pool:
            fallback = [
                entity_id
                for entity_id in sorted(
                    self.entities.keys(),
                    key=lambda entity_id: self.entity_popularity.get(entity_id, 0),
                    reverse=True,
                )
                if entity_id != sample.quadruple.object
            ]
            pool = fallback[: max(self.config.negative_sampling.k_recall, self.config.negative_sampling.n_neg, 16)]
        return pool

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
        return torch.amp.autocast("cuda", dtype=torch.float16)

    def _log_runtime_diagnostics(self) -> None:
        self.logger.info(
            "runtime device=%s cuda_available=%s cuda_device_count=%s amp=%s data_parallel=%s",
            self.device,
            torch.cuda.is_available(),
            torch.cuda.device_count() if torch.cuda.is_available() else 0,
            self.use_amp,
            self.enable_data_parallel,
        )
        self.logger.info(
            "dataset sizes train=%s valid=%s test=%s micro_batch=%s grad_accum=%s eval_batch=%s candidate_chunk=%s",
            len(self.train_dataset),
            len(self.valid_dataset),
            len(self.test_dataset),
            self.config.micro_batch_size,
            self.config.gradient_accumulation_steps,
            self.config.eval_batch_size,
            self.config.candidate_chunk_size,
        )
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(self.device)
            self.logger.info(
                "gpu name=%s total_memory_gb=%.2f capability=%s.%s",
                props.name,
                props.total_memory / (1024 ** 3),
                props.major,
                props.minor,
            )
            gpu_stats = self._query_gpu_stats()
            self.logger.info(
                "gpu initial util=%s mem_mb=%s/%s",
                gpu_stats["utilization_gpu"],
                gpu_stats["memory_used_mb"],
                gpu_stats["memory_total_mb"],
            )

    def _query_gpu_stats(self) -> dict[str, str]:
        default = {
            "utilization_gpu": "NA",
            "memory_used_mb": "NA",
            "memory_total_mb": "NA",
        }
        if not torch.cuda.is_available():
            return default
        try:
            output = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            lines = [line.strip() for line in output.splitlines() if line.strip()]
            device_index = self.device.index or 0
            if device_index >= len(lines):
                return default
            util, mem_used, mem_total = [part.strip() for part in lines[device_index].split(",")]
            return {
                "utilization_gpu": util,
                "memory_used_mb": mem_used,
                "memory_total_mb": mem_total,
            }
        except Exception:
            return default
