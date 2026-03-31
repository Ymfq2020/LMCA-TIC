from pathlib import Path

from lmca_tic.data.bie_builder import OfflineBIEBuilder


def test_offline_bie_builder_generates_expected_fields(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "train.txt").write_text(
        "United States\tmeet\tUnited Nations\t1\n"
        "Ministry of Defense\tattack\tArmed Group\t2\n"
        "President John Smith\tvisit\tFrance\t3\n",
        encoding="utf-8",
    )
    (raw_dir / "valid.txt").write_text("", encoding="utf-8")
    (raw_dir / "test.txt").write_text("", encoding="utf-8")
    output_path = tmp_path / "bie" / "entity_metadata.jsonl"

    records = OfflineBIEBuilder().build_from_dir(raw_dir=raw_dir, output_path=output_path)

    assert output_path.exists()
    assert records["United States"]["entity_type"] == "Country"
    assert records["United Nations"]["entity_type"] in {"InternationalOrganization", "Organization"}
    assert records["Ministry of Defense"]["entity_type"] == "GovernmentOrganization"
    assert "description" in records["President John Smith"]
    assert "top_relations" in records["President John Smith"]
