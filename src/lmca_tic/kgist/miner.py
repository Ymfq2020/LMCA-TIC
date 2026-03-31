"""KGIST summary miner and negative-error scorer.

The implementation is an engineering rewrite of the user-provided
`kgist.ipynb`. The notebook contains a broader MDL rule search prototype with
prototype-only code paths and syntax defects; this module extracts the
reproducible subset needed for ontology summaries and hard negative generation.
"""

from __future__ import annotations

from collections import Counter, defaultdict

from lmca_tic.data.types import ProcessedSample
from lmca_tic.kgist.evaluator import NegativeErrorEvaluator
from lmca_tic.kgist.graph import TypedTemporalGraph
from lmca_tic.kgist.types import GraphSummaryArtifact, SummaryRule


class KGISTSummaryMiner:
    def __init__(self, min_support: int = 1) -> None:
        self.min_support = min_support

    def mine(self, samples: list[ProcessedSample]) -> GraphSummaryArtifact:
        graph = TypedTemporalGraph(samples)
        evaluator = NegativeErrorEvaluator(
            num_entities=graph.num_entities,
            num_relations=len(graph.relation_edges),
            num_edges=graph.num_edges,
        )
        rule_counter: Counter[tuple[tuple[str, ...], str, tuple[str, ...]]] = Counter()
        relation_counter: Counter[str] = Counter()
        source_constraints: dict[str, set[str]] = defaultdict(set)
        target_constraints: dict[str, set[str]] = defaultdict(set)

        for sample in samples:
            quad = sample.quadruple
            relation_counter[quad.relation] += 1
            source_constraints[quad.relation].update(sample.subject_types)
            target_constraints[quad.relation].update(sample.object_types)
            rule_counter[(sample.subject_types, quad.relation, sample.object_types)] += 1

        rules: list[SummaryRule] = []
        rule_gain: dict[str, float] = {}
        negative_error_weight: dict[str, float] = {}
        coverage: dict[str, float] = {}

        total_edges = max(graph.num_edges, 1)
        for relation, count in relation_counter.items():
            coverage[relation] = count / total_edges

        for (source_types, relation, target_types), support in sorted(rule_counter.items()):
            if support < self.min_support:
                continue
            gain = evaluator.rule_gain(modeled_edges=support)
            rule = SummaryRule(
                source_types=source_types,
                relation=relation,
                target_types=target_types,
                support=support,
                coverage=support / total_edges,
                rule_gain=gain,
            )
            rules.append(rule)
            rule_gain[rule.rule_id] = gain
            negative_error_weight[_negative_error_key(source_types, relation, target_types)] = gain
            for source_type in source_types:
                for target_type in target_types:
                    singleton_key = _negative_error_key((source_type,), relation, (target_type,))
                    negative_error_weight[singleton_key] = negative_error_weight.get(singleton_key, 0.0) + gain

        type_constraints = {
            relation: {
                "src": sorted(source_constraints[relation]),
                "dst": sorted(target_constraints[relation]),
            }
            for relation in relation_counter
        }

        return GraphSummaryArtifact(
            rules=rules,
            coverage=coverage,
            type_constraints=type_constraints,
            rule_gain=rule_gain,
            negative_error_weight=negative_error_weight,
        )


class NegativeErrorScorer:
    def __init__(self, artifact: GraphSummaryArtifact) -> None:
        self.artifact = artifact

    def score(
        self,
        subject_types: tuple[str, ...],
        relation: str,
        candidate_types: tuple[str, ...],
    ) -> float:
        total = 0.0
        for source_type in subject_types:
            for candidate_type in candidate_types:
                key = _negative_error_key((source_type,), relation, (candidate_type,))
                total += self.artifact.negative_error_weight.get(key, 0.0)
        return total

    def allowed_tail_types(self, relation: str) -> set[str]:
        return set(self.artifact.type_constraints.get(relation, {}).get("dst", []))


def _negative_error_key(
    source_types: tuple[str, ...],
    relation: str,
    target_types: tuple[str, ...],
) -> str:
    return f"{','.join(source_types)}|{relation}|{','.join(target_types)}"
