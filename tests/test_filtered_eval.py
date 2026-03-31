from lmca_tic.evaluation.filtered import FilteredEvaluator


def test_filtered_evaluator_ignores_known_true_corruptions():
    evaluator = FilteredEvaluator({"s\tr\t1": ["gold", "other_true"]})
    metrics = evaluator.evaluate(
        [
            {
                "subject": "s",
                "relation": "r",
                "timestamp": 1,
                "gold": "gold",
                "scores": {"gold": 0.5, "other_true": 0.9, "neg": 0.4},
            }
        ]
    )
    assert metrics.mrr == 1.0
