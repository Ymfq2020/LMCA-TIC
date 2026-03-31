from lmca_tic.data.bie import BIEPromptBuilder, load_bie_records


def test_bie_prompt_builder_keeps_order(tmp_path):
    path = tmp_path / "bie.jsonl"
    path.write_text(
        '{"entity_id":"e1","entity_name":"Entity 1","entity_type":"Country","country":"US","sector":"Gov"}\n',
        encoding="utf-8",
    )
    records = load_bie_records(path)
    prompt = BIEPromptBuilder(["entity_type", "country", "sector"]).build_prompt(records["e1"])
    assert "entity_type: Country" in prompt
    assert "country: US" in prompt
    assert "sector: Gov" in prompt
