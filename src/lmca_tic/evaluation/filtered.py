"""Filtered ranking metrics for temporal KG completion."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean


@dataclass
class RankingMetrics:
    mrr: float
    hits_at_1: float
    hits_at_3: float
    hits_at_10: float
    auc_pr: float | None = None

    def to_dict(self) -> dict[str, float]:
        payload = {
            "MRR": self.mrr,
            "Hits@1": self.hits_at_1,
            "Hits@3": self.hits_at_3,
            "Hits@10": self.hits_at_10,
        }
        if self.auc_pr is not None:
            payload["AUC-PR"] = self.auc_pr
        return payload


class FilteredEvaluator:
    def __init__(self, filtered_targets: dict[str, list[str]]) -> None:
        self.filtered_targets = filtered_targets

    def evaluate(
        self,
        predictions: list[dict[str, object]],
    ) -> RankingMetrics:
        reciprocal_ranks: list[float] = []
        hits1: list[float] = []
        hits3: list[float] = []
        hits10: list[float] = []
        pr_values: list[float] = []

        for prediction in predictions:
            rank = self._filtered_rank(
                subject=prediction["subject"],
                relation=prediction["relation"],
                timestamp=int(prediction["timestamp"]),
                gold=prediction["gold"],
                scores=prediction["scores"],
            )
            reciprocal_ranks.append(1.0 / rank)
            hits1.append(1.0 if rank <= 1 else 0.0)
            hits3.append(1.0 if rank <= 3 else 0.0)
            hits10.append(1.0 if rank <= 10 else 0.0)
            pr_values.append(1.0 / (rank + 1.0))

        return RankingMetrics(
            mrr=mean(reciprocal_ranks) if reciprocal_ranks else 0.0,
            hits_at_1=mean(hits1) if hits1 else 0.0,
            hits_at_3=mean(hits3) if hits3 else 0.0,
            hits_at_10=mean(hits10) if hits10 else 0.0,
            auc_pr=mean(pr_values) if pr_values else 0.0,
        )

    def _filtered_rank(
        self,
        subject: str,
        relation: str,
        timestamp: int,
        gold: str,
        scores: dict[str, float],
    ) -> int:
        key = f"{subject}\t{relation}\t{timestamp}"
        filtered = set(self.filtered_targets.get(key, []))
        gold_score = scores[gold]
        better = 0
        for candidate, score in scores.items():
            if candidate == gold:
                continue
            if candidate in filtered:
                continue
            if score > gold_score:
                better += 1
        return better + 1
