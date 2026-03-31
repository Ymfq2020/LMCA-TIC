"""Notebook-inspired rule gain scoring.

This module borrows the negative-error perspective from the provided
`kgist.ipynb` prototype, but narrows the scope to the typed relation summaries
required by LMCA-TIC negative sampling.
"""

from __future__ import annotations

from math import lgamma, log2


def log_binomial(n: int, k: int) -> float:
    if k < 0 or k > n:
        return 0.0
    if k == 0 or k == n:
        return 0.0
    return (lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)) / log2(2.0)


class NegativeErrorEvaluator:
    """Compute rule-level negative error reduction."""

    def __init__(self, num_entities: int, num_relations: int, num_edges: int) -> None:
        self.num_entities = max(num_entities, 1)
        self.num_relations = max(num_relations, 1)
        self.num_edges = max(num_edges, 1)
        self.null_error = self._edge_error(self.num_edges, 0)

    def rule_gain(self, modeled_edges: int) -> float:
        modeled_edges = max(0, min(modeled_edges, self.num_edges))
        return self.null_error - self._edge_error(self.num_edges, modeled_edges)

    def _edge_error(self, observed_edges: int, modeled_edges: int) -> float:
        unexplained = max(observed_edges - modeled_edges, 0)
        possible = max((self.num_entities ** 2) * self.num_relations - modeled_edges, unexplained)
        return log_binomial(possible, unexplained)
