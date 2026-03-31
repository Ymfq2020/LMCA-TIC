"""KGIST summary artifacts derived from the provided notebook prototype."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SummaryRule:
    source_types: tuple[str, ...]
    relation: str
    target_types: tuple[str, ...]
    support: int
    coverage: float
    rule_gain: float

    @property
    def rule_id(self) -> str:
        return "|".join(
            [
                ",".join(self.source_types),
                self.relation,
                ",".join(self.target_types),
            ]
        )


@dataclass
class GraphSummaryArtifact:
    rules: list[SummaryRule] = field(default_factory=list)
    coverage: dict[str, float] = field(default_factory=dict)
    type_constraints: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    rule_gain: dict[str, float] = field(default_factory=dict)
    negative_error_weight: dict[str, float] = field(default_factory=dict)
