from lmca_tic.config.schemas import NegativeSamplerConfig
from lmca_tic.kgist.types import GraphSummaryArtifact
from lmca_tic.kgist.miner import NegativeErrorScorer
from lmca_tic.training.negative_sampling import HardNegativeSampler


def build_sampler(mode: str, rho: float = 0.3) -> HardNegativeSampler:
    artifact = GraphSummaryArtifact(
        type_constraints={"r": {"src": ["entity_type=Country"], "dst": ["entity_type=Org"]}},
        negative_error_weight={"entity_type=Country|r|entity_type=Org": 2.0},
    )
    scorer = NegativeErrorScorer(artifact)
    return HardNegativeSampler(
        NegativeSamplerConfig(mode=mode, k_recall=4, n_neg=2, tau=0.5, alpha=0.2, rho=rho),
        scorer=scorer,
    )


def test_negative_sampler_supports_all_modes():
    candidate_scores = {"o1": 0.9, "o2": 0.8, "o3": 0.2, "o4": 0.1}
    candidate_types = {
        "o1": ("entity_type=Org",),
        "o2": ("entity_type=Org",),
        "o3": ("entity_type=Person",),
        "o4": ("entity_type=Org",),
    }
    for mode in ("random_uniform", "contrastive_equal", "ontology_weighted"):
        negatives = build_sampler(mode).sample(
            positive_object="gold",
            relation="r",
            subject_types=("entity_type=Country",),
            candidate_scores=candidate_scores,
            candidate_types=candidate_types,
        )
        assert len(negatives) <= 2
        assert all(candidate != "gold" for candidate in negatives)


def test_negative_sampler_tolerates_non_finite_weights():
    sampler = build_sampler("ontology_weighted")
    negatives = sampler._hybrid_sample(
        [
            ("o1", float("inf")),
            ("o2", float("nan")),
            ("o4", float("-inf")),
        ]
    )
    assert len(negatives) <= 2
    assert all(candidate in {"o1", "o2", "o4"} for candidate in negatives)


def test_rho_zero_skips_ontology_bonus():
    sampler = build_sampler("ontology_weighted", rho=0.0)
    candidate_scores = {"o1": 0.9, "o2": 0.8, "o3": 0.7}
    candidate_types = {
        "o1": ("entity_type=Org",),
        "o2": ("entity_type=Org",),
        "o3": ("entity_type=Org",),
    }
    weighted = sampler._weighted_scores(
        ["o1", "o2", "o3"],
        relation="r",
        subject_types=("entity_type=Country",),
        candidate_scores=candidate_scores,
        candidate_types=candidate_types,
    )
    # With rho=0 the weighted score equals the base candidate score.
    assert {entity: round(score, 6) for entity, score in weighted} == {
        "o1": 0.9,
        "o2": 0.8,
        "o3": 0.7,
    }


def test_rho_positive_adds_ontology_bonus():
    sampler = build_sampler("ontology_weighted", rho=0.5)
    candidate_scores = {"o1": 0.9}
    candidate_types = {"o1": ("entity_type=Org",)}
    weighted = sampler._weighted_scores(
        ["o1"],
        relation="r",
        subject_types=("entity_type=Country",),
        candidate_scores=candidate_scores,
        candidate_types=candidate_types,
    )
    # ontology score returned by NegativeErrorScorer is 2.0; weighted = 0.9 + 0.5 * 2.0
    assert round(weighted[0][1], 6) == 1.9
