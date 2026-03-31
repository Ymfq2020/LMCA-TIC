from lmca_tic.data.types import ProcessedSample, TemporalQuadruple
from lmca_tic.kgist.miner import KGISTSummaryMiner, NegativeErrorScorer


def _sample(subject: str, relation: str, obj: str, stype: str, otype: str) -> ProcessedSample:
    return ProcessedSample(
        quadruple=TemporalQuadruple(subject, relation, obj, 1, "train"),
        subject_prompt="s",
        object_prompt="o",
        relation_history=[1.0, 0.0],
        subject_neighbors=[],
        object_neighbors=[],
        subject_types=(stype,),
        object_types=(otype,),
    )


def test_kgist_miner_produces_positive_rule_gain():
    samples = [
        _sample("A", "ally", "B", "entity_type=Country", "entity_type=Country"),
        _sample("C", "ally", "D", "entity_type=Country", "entity_type=Country"),
    ]
    artifact = KGISTSummaryMiner().mine(samples)
    assert artifact.rules
    assert max(rule.rule_gain for rule in artifact.rules) >= 0.0
    scorer = NegativeErrorScorer(artifact)
    assert scorer.score(("entity_type=Country",), "ally", ("entity_type=Country",)) >= 0.0
