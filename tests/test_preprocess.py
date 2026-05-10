from pathlib import Path

from lmca_tic.config.schemas import ExperimentConfig
from lmca_tic.data.preprocess import (
    LocalTKGPreprocessor,
    build_entity_history_index,
    entity_temporal_context_from_index,
    neighbor_time_deltas,
    relation_history_vector,
    sample_temporal_neighbors,
)
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


def test_temporal_neighbor_sampling_uses_strictly_past_history():
    from lmca_tic.data.types import TemporalQuadruple

    history = [
        TemporalQuadruple("A", "r", "B", 1, "train"),
        TemporalQuadruple("A", "r", "C", 3, "train"),
        TemporalQuadruple("A", "r", "D", 5, "train"),
    ]
    (
        subject_neighbors,
        subject_relations,
        subject_deltas,
        _,
        _,
        _,
    ) = sample_temporal_neighbors(history, "A", "B", timestamp=3, window_days=5, max_neighbors=10)
    assert subject_neighbors == ["B"]
    assert subject_relations == ["r"]
    assert subject_deltas == [2.0]
    assert "D" not in subject_neighbors
    assert "C" not in subject_neighbors
    assert len(subject_neighbors) == len(subject_relations) == len(subject_deltas)
    assert all(delta >= 0 for delta in subject_deltas)
    deltas = neighbor_time_deltas(history, "A", timestamp=3, window_days=5, max_neighbors=10)
    assert all(delta >= 0 for delta in deltas)
    assert deltas == [2.0]
    assert relation_history_vector(history, "r", timestamp=3, window_size=4) == [0.0, 0.0, 0.0, 1.0]


def test_entity_history_index_recomputes_context_for_each_query_timestamp():
    from lmca_tic.data.types import TemporalQuadruple

    history = [
        TemporalQuadruple("A", "r", "B", 1, "train"),
        TemporalQuadruple("A", "r", "C", 4, "train"),
        TemporalQuadruple("A", "r", "D", 5, "train"),
    ]
    history_index = build_entity_history_index(history)
    neighbors, relations, deltas = entity_temporal_context_from_index(
        history_index,
        entity="A",
        timestamp=5,
        window_days=2,
        max_neighbors=10,
    )
    assert neighbors == ["C"]
    assert relations == ["r"]
    assert deltas == [1.0]
    neighbors, relations, deltas = entity_temporal_context_from_index(
        history_index,
        entity="A",
        timestamp=6,
        window_days=2,
        max_neighbors=10,
    )
    assert neighbors == ["C", "D"]
    assert relations == ["r", "r"]
    assert deltas == [2.0, 1.0]
