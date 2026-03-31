"""Local ICEWS preprocessing for LMCA-TIC."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from lmca_tic.config.schemas import ExperimentConfig
from lmca_tic.data.bie import BIEPromptBuilder, empty_bie_record, load_bie_records
from lmca_tic.data.types import ProcessedSample, TemporalQuadruple
from lmca_tic.utils.io import ensure_dir, write_json, write_jsonl


class LocalTKGPreprocessor:
    """Preprocess local train/valid/test splits into reusable artifacts."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.raw_dir = Path(config.raw_dir)
        self.processed_dir = ensure_dir(config.processed_dir)
        self.prompt_builder = BIEPromptBuilder(config.bie_ordered_keys)
        self.bie_records = (
            load_bie_records(config.bie_path)
            if config.bie_path
            else {}
        )

    def run(self) -> dict[str, object]:
        splits = {
            split: self._load_split(split)
            for split in ("train", "valid", "test")
        }
        train_entities = {q.subject for q in splits["train"]} | {q.object for q in splits["train"]}
        all_entities = sorted(
            {
                entity
                for values in splits.values()
                for q in values
                for entity in (q.subject, q.object)
            }
        )
        relations = sorted({q.relation for values in splits.values() for q in values})
        inverse_relations = [f"{relation}__inverse" for relation in relations]

        filtered_targets = build_filtered_targets(splits.values())
        split_payloads: dict[str, list[dict[str, object]]] = {}
        inductive_stats: dict[str, int] = defaultdict(int)

        for split, quadruples in splits.items():
            processed_samples = [
                self._build_sample(
                    q,
                    history=splits["train"] if split != "train" else quadruples,
                    train_entities=train_entities,
                )
                for q in quadruples
            ]
            split_payloads[split] = [sample.to_dict() for sample in processed_samples]
            inductive_stats[f"{split}_inductive"] = sum(sample.quadruple.is_inductive for sample in processed_samples)
            inductive_stats[f"{split}_transductive"] = len(processed_samples) - inductive_stats[f"{split}_inductive"]
            write_jsonl(self.processed_dir / f"{split}.jsonl", split_payloads[split])

        entity_payload = {
            entity_id: asdict(self.bie_records.get(entity_id, empty_bie_record(entity_id)))
            for entity_id in all_entities
        }
        relation_frequency = compute_relation_frequency(splits["train"])
        manifest = {
            "dataset_name": self.config.dataset_name,
            "num_entities": len(all_entities),
            "num_relations": len(relations),
            "inverse_relations": inverse_relations,
            "inductive_stats": dict(inductive_stats),
        }
        write_json(self.processed_dir / "entities.json", entity_payload)
        write_json(self.processed_dir / "relations.json", {"relations": relations, "inverse_relations": inverse_relations})
        write_json(self.processed_dir / "filtered_targets.json", filtered_targets)
        write_json(self.processed_dir / "relation_frequency.json", relation_frequency)
        write_json(self.processed_dir / "manifest.json", manifest)
        return manifest

    def _load_split(self, split: str) -> list[TemporalQuadruple]:
        path = self.raw_dir / f"{split}.txt"
        quadruples: list[TemporalQuadruple] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                subject, relation, obj, timestamp = _split_line(line, self.config.delimiter)
                quadruples.append(
                    TemporalQuadruple(
                        subject=subject,
                        relation=relation,
                        object=obj,
                        timestamp=int(timestamp),
                        split=split,
                    )
                )
        return sorted(quadruples, key=lambda item: item.timestamp)

    def _build_sample(
        self,
        quadruple: TemporalQuadruple,
        history: list[TemporalQuadruple],
        train_entities: set[str],
    ) -> ProcessedSample:
        subject_record = self.bie_records.get(quadruple.subject, empty_bie_record(quadruple.subject))
        object_record = self.bie_records.get(quadruple.object, empty_bie_record(quadruple.object))
        relation_history = relation_history_vector(history, quadruple.relation, quadruple.timestamp)
        subject_neighbors, object_neighbors = sample_temporal_neighbors(
            history=history,
            subject=quadruple.subject,
            obj=quadruple.object,
            timestamp=quadruple.timestamp,
            window_days=self.config.model.tgn_time_window_days,
            max_neighbors=self.config.model.tgn_neighbor_size,
        )
        is_inductive = quadruple.subject not in train_entities or quadruple.object not in train_entities
        subject_types = extract_entity_types(subject_record, self.config.ontology_keys)
        object_types = extract_entity_types(object_record, self.config.ontology_keys)
        processed_quadruple = TemporalQuadruple(
            subject=quadruple.subject,
            relation=quadruple.relation,
            object=quadruple.object,
            timestamp=quadruple.timestamp,
            split=quadruple.split,
            is_inductive=is_inductive,
        )
        return ProcessedSample(
            quadruple=processed_quadruple,
            subject_prompt=self.prompt_builder.build_prompt(subject_record),
            object_prompt=self.prompt_builder.build_prompt(object_record),
            relation_history=relation_history,
            subject_neighbors=subject_neighbors,
            object_neighbors=object_neighbors,
            subject_types=subject_types,
            object_types=object_types,
            extra={
                "subject_neighbor_deltas": neighbor_time_deltas(
                    history,
                    quadruple.subject,
                    quadruple.timestamp,
                    self.config.model.tgn_time_window_days,
                    self.config.model.tgn_neighbor_size,
                ),
                "object_neighbor_deltas": neighbor_time_deltas(
                    history,
                    quadruple.object,
                    quadruple.timestamp,
                    self.config.model.tgn_time_window_days,
                    self.config.model.tgn_neighbor_size,
                ),
            },
        )


def build_filtered_targets(split_groups: Iterable[list[TemporalQuadruple]]) -> dict[str, list[str]]:
    targets: dict[str, set[str]] = defaultdict(set)
    for group in split_groups:
        for q in group:
            key = f"{q.subject}\t{q.relation}\t{q.timestamp}"
            targets[key].add(q.object)
            inverse_key = f"{q.object}\t{q.relation}__inverse\t{q.timestamp}"
            targets[inverse_key].add(q.subject)
    return {key: sorted(value) for key, value in targets.items()}


def compute_relation_frequency(quadruples: list[TemporalQuadruple]) -> dict[str, list[int]]:
    by_relation: dict[str, Counter[int]] = defaultdict(Counter)
    for quadruple in quadruples:
        by_relation[quadruple.relation][quadruple.timestamp] += 1
    payload: dict[str, list[int]] = {}
    for relation, counter in by_relation.items():
        timeline = [counter[timestamp] for timestamp in sorted(counter)]
        payload[relation] = timeline
    return payload


def relation_history_vector(
    history: list[TemporalQuadruple],
    relation: str,
    timestamp: int,
    window_size: int = 16,
) -> list[float]:
    relevant = [
        q.timestamp
        for q in history
        if q.relation == relation and q.timestamp <= timestamp
    ]
    relevant = sorted(relevant)[-window_size:]
    if not relevant:
        return [0.0] * window_size
    counts = Counter(relevant)
    ordered = [float(counts[t]) for t in relevant]
    if len(ordered) < window_size:
        ordered = [0.0] * (window_size - len(ordered)) + ordered
    return ordered


def sample_temporal_neighbors(
    history: list[TemporalQuadruple],
    subject: str,
    obj: str,
    timestamp: int,
    window_days: int,
    max_neighbors: int,
) -> tuple[list[str], list[str]]:
    lower_bound = timestamp - window_days
    subject_neighbors: list[str] = []
    object_neighbors: list[str] = []
    for quadruple in history:
        if quadruple.timestamp > timestamp:
            continue
        if quadruple.timestamp < lower_bound:
            continue
        if quadruple.subject == subject:
            subject_neighbors.append(quadruple.object)
        elif quadruple.object == subject:
            subject_neighbors.append(quadruple.subject)
        if quadruple.subject == obj:
            object_neighbors.append(quadruple.object)
        elif quadruple.object == obj:
            object_neighbors.append(quadruple.subject)
    return subject_neighbors[-max_neighbors:], object_neighbors[-max_neighbors:]


def neighbor_time_deltas(
    history: list[TemporalQuadruple],
    entity: str,
    timestamp: int,
    window_days: int,
    max_neighbors: int,
) -> list[float]:
    lower_bound = timestamp - window_days
    deltas: list[float] = []
    for quadruple in history:
        if quadruple.timestamp > timestamp:
            continue
        if quadruple.timestamp < lower_bound:
            continue
        if quadruple.subject == entity or quadruple.object == entity:
            deltas.append(float(timestamp - quadruple.timestamp))
    return deltas[-max_neighbors:]


def extract_entity_types(record, ontology_keys: list[str]) -> tuple[str, ...]:
    collected: list[str] = []
    for key in ontology_keys:
        value = record.attributes.get(key)
        if value:
            collected.append(f"{key}={value}")
    if not collected:
        collected.append("entity_type=UNKNOWN")
    return tuple(collected)


def _split_line(line: str, delimiter: str) -> tuple[str, str, str, str]:
    if delimiter == " ":
        parts = line.split()
    else:
        parts = line.split(delimiter)
    if len(parts) != 4:
        raise ValueError(f"Expected 4 columns, got {len(parts)} in line: {line}")
    return parts[0], parts[1], parts[2], parts[3]
