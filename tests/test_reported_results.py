from pathlib import Path

from lmca_tic.experiments.reported_results import (
    CHAPTER3_REPORTED_TABLES,
    audit_chapter3_assets,
    export_chapter3_reported_results,
)
from lmca_tic.utils.io import read_json, read_jsonl


def test_export_chapter3_reported_results_writes_provenance_files(tmp_path):
    output_dir = tmp_path / "paper_reported_results" / "chapter3"
    manifest = export_chapter3_reported_results(output_dir)

    assert manifest["chapter"] == 3
    assert manifest["provenance"] == "paper_reported"
    assert (output_dir / "manifest.json").exists()
    assert (output_dir / "reported_results.jsonl").exists()

    rows = read_jsonl(output_dir / "reported_results.jsonl")
    assert rows
    assert all(row["provenance"] == "paper_reported" for row in rows)
    assert all(row["reported_in_manuscript"] is True for row in rows)

    table_payload = read_json(output_dir / "table_3_3_main.json")
    assert table_payload["paper_table_label"] == "tab:lmcatic_main_results"
    assert table_payload["rows"][0]["provenance"] == "paper_reported"


def test_audit_chapter3_assets_distinguishes_reported_and_runtime_assets(tmp_path):
    reported_root = tmp_path / "paper_reported_results" / "chapter3"
    export_chapter3_reported_results(reported_root)

    experiment_root = tmp_path / "outputs" / "experiments"
    (experiment_root / "main").mkdir(parents=True)
    (experiment_root / "main" / "summary.csv").write_text("experiment,MRR_mean\nx,1.0\n", encoding="utf-8")

    audit = audit_chapter3_assets(experiment_root=experiment_root, reported_root=reported_root)
    assert audit["reported_ok"] is True
    assert audit["runtime_ok"] is False

    reported_assets = {item["name"]: item["exists"] for item in audit["reported_assets"]}
    runtime_assets = {item["name"]: item["exists"] for item in audit["runtime_assets"]}

    assert reported_assets["manifest"] is True
    assert reported_assets["table_3_3_main"] is True
    assert runtime_assets["main_summary"] is True
    assert runtime_assets["main_per_seed"] is False


def test_reported_table_registry_covers_all_expected_tables():
    assert len(CHAPTER3_REPORTED_TABLES) >= 13
    assert "table_3_3_main" in CHAPTER3_REPORTED_TABLES
    assert "table_3_10_backbone" in CHAPTER3_REPORTED_TABLES
    assert "table_3_15_negative_open" in CHAPTER3_REPORTED_TABLES
