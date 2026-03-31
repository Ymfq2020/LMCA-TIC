"""Canonical data interfaces used across preprocessing, training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class BIERecord:
    entity_id: str
    entity_name: str
    attributes: dict[str, str]


@dataclass(frozen=True)
class TemporalQuadruple:
    subject: str
    relation: str
    object: str
    timestamp: int
    split: str
    is_inductive: bool = False


@dataclass
class ProcessedSample:
    quadruple: TemporalQuadruple
    subject_prompt: str
    object_prompt: str
    relation_history: list[float]
    subject_neighbors: list[str]
    object_neighbors: list[str]
    subject_types: tuple[str, ...]
    object_types: tuple[str, ...]
    negative_candidates: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "quadruple": {
                "subject": self.quadruple.subject,
                "relation": self.quadruple.relation,
                "object": self.quadruple.object,
                "timestamp": self.quadruple.timestamp,
                "split": self.quadruple.split,
                "is_inductive": self.quadruple.is_inductive,
            },
            "subject_prompt": self.subject_prompt,
            "object_prompt": self.object_prompt,
            "relation_history": self.relation_history,
            "subject_neighbors": self.subject_neighbors,
            "object_neighbors": self.object_neighbors,
            "subject_types": list(self.subject_types),
            "object_types": list(self.object_types),
            "negative_candidates": self.negative_candidates,
            "extra": self.extra,
        }
