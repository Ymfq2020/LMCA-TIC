from pathlib import Path

from lmca_tic.config.schemas import ExperimentConfig
from lmca_tic.data.preprocess import LocalTKGPreprocessor, neighbor_time_deltas, sample_temporal_neighbors
from lmca_tic.utils.io import read_json, read_jsonl


def test_preprocess_builds_inductive_subset(tmp_path):
    root = Path("data/smoke/icews14")
    config = ExperimentConfig(
        name="smoke",
        dataset_name="ICEWS14-SMOKE",
        raw_dir=str(root),
        processed_dir=str(tmp_path / "processed"),
        bie_path="data/smoke/bie/entity_metadata.jsonl",
        bie_ordered_keys=["entity_type", "country", "sector"],
        ontology_keys=["entity_type", "country"],
        output_dir=str(tmp_path / "outputs"),
        log_dir=str(tmp_path / "logs"),
        checkpoint_dir=str(tmp_path / "ckpt"),
    )
    manifest = LocalTKGPreprocessor(config).run()
    assert manifest["inductive_stats"]["test_inductive"] >= 1
    test_rows = read_jsonl(tmp_path / "processed" / "test.jsonl")
    assert any(row["quadruple"]["is_inductive"] for row in test_rows)
    filtered = read_json(tmp_path / "processed" / "filtered_targets.json")
    assert "USA\tmeet\t1" in filtered


def test_temporal_neighbor_sampling_blocks_future_leakage():
    from lmca_tic.data.types import TemporalQuadruple

    history = [
        TemporalQuadruple("A", "r", "B", 1, "train"),
        TemporalQuadruple("A", "r", "C", 3, "train"),
        TemporalQuadruple("A", "r", "D", 5, "train"),
    ]
    neighbors, _ = sample_temporal_neighbors(history, "A", "B", timestamp=3, window_days=5, max_neighbors=10)
    assert "D" not in neighbors
    deltas = neighbor_time_deltas(history, "A", timestamp=3, window_days=5, max_neighbors=10)
    assert all(delta >= 0 for delta in deltas)
