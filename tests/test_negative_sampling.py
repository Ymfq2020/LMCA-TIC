from lmca_tic.config.schemas import NegativeSamplerConfig
from lmca_tic.kgist.types import GraphSummaryArtifact
from lmca_tic.kgist.miner import NegativeErrorScorer
from lmca_tic.training.negative_sampling import HardNegativeSampler


def build_sampler(mode: str) -> HardNegativeSampler:
    artifact = GraphSummaryArtifact(
        type_constraints={"r": {"src": ["entity_type=Country"], "dst": ["entity_type=Org"]}},
        negative_error_weight={"entity_type=Country|r|entity_type=Org": 2.0},
    )
    scorer = NegativeErrorScorer(artifact)
    return HardNegativeSampler(
        NegativeSamplerConfig(mode=mode, k_recall=4, n_neg=2, tau=0.7, alpha=0.5),
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
