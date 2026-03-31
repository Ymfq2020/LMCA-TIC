"""Background information enhancement utilities.

The prompt template follows the role of Eq. (3-1) in Chapter 3.2.2 of the
provided PDF while keeping the metadata schema configurable.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

from .types import BIERecord


class BIEPromptBuilder:
    """Convert structured entity metadata into LLM prompts."""

    def __init__(self, ordered_keys: list[str], max_attributes: int = 8) -> None:
        self.ordered_keys = ordered_keys[:max_attributes]
        self.max_attributes = max_attributes

    def build_prompt(self, record: BIERecord) -> str:
        fragments: list[str] = []
        for key in self.ordered_keys:
            value = record.attributes.get(key)
            if value:
                fragments.append(f"{key}: {value}")
        if not fragments:
            fragments.append("未提供额外背景属性")
        background = "；".join(fragments[: self.max_attributes])
        return (
            f"实体名称：{record.entity_name}。"
            f"背景知识：{background}。"
            "请提取其核心属性以支持时序知识图谱推理。"
        )


def load_bie_records(path: str | Path) -> dict[str, BIERecord]:
    source = Path(path)
    if source.suffix.lower() == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return _records_from_iterable(payload)
        if isinstance(payload, dict):
            rows = []
            for entity_id, entry in payload.items():
                entry = dict(entry)
                entry["entity_id"] = entity_id
                rows.append(entry)
            return _records_from_iterable(rows)
    if source.suffix.lower() == ".jsonl":
        rows = [
            json.loads(line)
            for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return _records_from_iterable(rows)
    if source.suffix.lower() == ".csv":
        with source.open("r", encoding="utf-8") as handle:
            return _records_from_iterable(csv.DictReader(handle))
    raise ValueError(f"Unsupported BIE file format: {source.suffix}")


def empty_bie_record(entity_id: str) -> BIERecord:
    return BIERecord(entity_id=entity_id, entity_name=entity_id, attributes={})


def _records_from_iterable(rows: Iterable[dict[str, object]]) -> dict[str, BIERecord]:
    records: dict[str, BIERecord] = {}
    for row in rows:
        entity_id = str(row["entity_id"])
        entity_name = str(row.get("entity_name", entity_id))
        attributes = {
            str(key): str(value)
            for key, value in row.items()
            if key not in {"entity_id", "entity_name"} and value is not None and str(value) != ""
        }
        records[entity_id] = BIERecord(entity_id=entity_id, entity_name=entity_name, attributes=attributes)
    return records
