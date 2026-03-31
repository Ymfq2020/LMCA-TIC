from lmca_tic.config.loader import load_experiment_config


def test_load_smoke_config():
    config = load_experiment_config("configs/experiments/smoke_icews14.yaml")
    assert config.name == "smoke_icews14"
    assert config.model.embedding_dim == 64
    assert config.negative_sampling.mode == "ontology_weighted"
