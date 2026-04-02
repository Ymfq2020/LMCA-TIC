from lmca_tic.config.loader import load_experiment_config


def test_load_smoke_config():
    config = load_experiment_config("configs/experiments/smoke_icews14.yaml")
    assert config.name == "smoke_icews14"
    assert config.model.embedding_dim == 64
    assert config.negative_sampling.mode == "ontology_weighted"


def test_load_a10_record_config():
    config = load_experiment_config("configs/experiments/icews14_record_qwen25_05b_a10.yaml")
    assert config.name == "icews14_record_qwen25_05b_a10"
    assert config.num_epochs == 1
    assert config.micro_batch_size == 2
    assert config.model.llm_name == "models/Qwen2.5-0.5B-Instruct"
    assert config.negative_sampling.k_recall == 8


def test_load_a10_demo_plus_config():
    config = load_experiment_config("configs/experiments/icews14_demo_plus_qwen25_05b_a10.yaml")
    assert config.name == "icews14_demo_plus_qwen25_05b_a10"
    assert config.num_epochs == 2
    assert config.gradient_accumulation_steps == 2
    assert config.model.use_4bit is True
    assert config.negative_sampling.k_recall == 16
