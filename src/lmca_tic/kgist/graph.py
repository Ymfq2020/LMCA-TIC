"""A typed temporal graph representation used by the KGIST miner."""

from __future__ import annotations

from collections import defaultdict

from lmca_tic.data.types import ProcessedSample


class TypedTemporalGraph:
    """Compact typed graph for mining relation ontology summaries."""

    def __init__(self, samples: list[ProcessedSample]) -> None:
        self.samples = samples
        self.entity_to_types: dict[str, tuple[str, ...]] = {}
        self.relation_edges: dict[str, list[ProcessedSample]] = defaultdict(list)
        for sample in samples:
            quad = sample.quadruple
            self.entity_to_types[quad.subject] = sample.subject_types
            self.entity_to_types[quad.object] = sample.object_types
            self.relation_edges[quad.relation].append(sample)

    @property
    def num_edges(self) -> int:
        return len(self.samples)

    @property
    def num_entities(self) -> int:
        return len(self.entity_to_types)

    def subject_types(self, entity_id: str) -> tuple[str, ...]:
        return self.entity_to_types.get(entity_id, ("entity_type=UNKNOWN",))

    def object_types(self, entity_id: str) -> tuple[str, ...]:
        return self.subject_types(entity_id)
