"""Hard negative sampling for LMCA-TIC."""

from __future__ import annotations

import math
import random

from lmca_tic.config.schemas import NegativeSamplerConfig
from lmca_tic.kgist.miner import NegativeErrorScorer


class HardNegativeSampler:
    def __init__(self, config: NegativeSamplerConfig, scorer: NegativeErrorScorer | None = None) -> None:
        self.config = config
        self.scorer = scorer

    def sample(
        self,
        positive_object: str,
        relation: str,
        subject_types: tuple[str, ...],
        candidate_scores: dict[str, float],
        candidate_types: dict[str, tuple[str, ...]],
    ) -> list[str]:
        recalled = self._recall_candidates(candidate_scores)
        filtered = [
            entity
            for entity in recalled
            if entity != positive_object and self._type_compatible(relation, subject_types, entity, candidate_types)
        ]
        if not filtered:
            return []
        if self.config.mode == "random_uniform":
            random.shuffle(filtered)
            return filtered[: self.config.n_neg]
        weighted = self._weighted_scores(filtered, relation, subject_types, candidate_scores, candidate_types)
        if self.config.mode == "contrastive_equal":
            weighted.sort(key=lambda item: item[1], reverse=True)
            return [entity for entity, _ in weighted[: self.config.n_neg]]
        return self._hybrid_sample(weighted)

    def _recall_candidates(self, candidate_scores: dict[str, float]) -> list[str]:
        return [
            entity
            for entity, _ in sorted(candidate_scores.items(), key=lambda item: item[1], reverse=True)[: self.config.k_recall]
        ]

    def _type_compatible(
        self,
        relation: str,
        subject_types: tuple[str, ...],
        candidate: str,
        candidate_types: dict[str, tuple[str, ...]],
    ) -> bool:
        if self.scorer is None:
            return True
        allowed = self.scorer.allowed_tail_types(relation)
        if not allowed:
            return True
        return any(candidate_type in allowed for candidate_type in candidate_types.get(candidate, ()))

    def _weighted_scores(
        self,
        filtered: list[str],
        relation: str,
        subject_types: tuple[str, ...],
        candidate_scores: dict[str, float],
        candidate_types: dict[str, tuple[str, ...]],
    ) -> list[tuple[str, float]]:
        weighted: list[tuple[str, float]] = []
        for entity in filtered:
            base_score = self._finite_or_default(candidate_scores.get(entity, 0.0))
            ontology_bonus = 0.0
            if self.scorer is not None:
                ontology_bonus = self._finite_or_default(
                    self.scorer.score(subject_types, relation, candidate_types.get(entity, ()))
                )
            weighted.append((entity, self._finite_or_default(base_score + ontology_bonus)))
        return weighted

    def _hybrid_sample(self, weighted: list[tuple[str, float]]) -> list[str]:
        weighted.sort(key=lambda item: item[1], reverse=True)
        hard_count = min(len(weighted), max(1, int(round(self.config.alpha * self.config.n_neg))))
        deterministic = [entity for entity, _ in weighted[:hard_count]]
        residual = weighted[hard_count:]
        probabilistic_count = max(self.config.n_neg - len(deterministic), 0)
        if probabilistic_count == 0 or not residual:
            return deterministic[: self.config.n_neg]

        logits = [self._finite_or_default(score) / max(self.config.tau, 1e-6) for _, score in residual]
        max_logit = max(logits)
        probs = [self._finite_or_default(math.exp(logit - max_logit)) for logit in logits]
        norm = sum(probs)
        if not math.isfinite(norm) or norm <= 0.0:
            probs = [1.0] * len(residual)
            norm = float(len(residual))
        probs = [value / norm for value in probs]
        sampled: list[str] = []
        pool = list(range(len(residual)))
        while pool and len(sampled) < probabilistic_count:
            pool_weights = [self._finite_or_default(probs[i]) for i in pool]
            weight_total = sum(pool_weights)
            if not math.isfinite(weight_total) or weight_total <= 0.0:
                idx = random.choice(pool)
            else:
                idx = random.choices(pool, weights=pool_weights, k=1)[0]
            sampled.append(residual[idx][0])
            pool.remove(idx)
        return deterministic + sampled

    @staticmethod
    def _finite_or_default(value: float, default: float = 0.0) -> float:
        return value if math.isfinite(value) else default
