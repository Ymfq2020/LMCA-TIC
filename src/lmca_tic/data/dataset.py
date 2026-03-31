"""Dataset loading for processed LMCA-TIC samples."""

from __future__ import annotations

from pathlib import Path

from lmca_tic.data.types import ProcessedSample, TemporalQuadruple
from lmca_tic.utils.io import read_jsonl


class LocalProcessedDataset:
    def __init__(self, processed_dir: str | Path, split: str) -> None:
        self.processed_dir = Path(processed_dir)
        rows = read_jsonl(self.processed_dir / f"{split}.jsonl")
        self.samples = [self._deserialize(row) for row in rows]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> ProcessedSample:
        return self.samples[index]

    @staticmethod
    def _deserialize(row: dict[str, object]) -> ProcessedSample:
        quadruple = TemporalQuadruple(**row["quadruple"])
        return ProcessedSample(
            quadruple=quadruple,
            subject_prompt=row["subject_prompt"],
            object_prompt=row["object_prompt"],
            relation_history=[float(v) for v in row["relation_history"]],
            subject_neighbors=list(row["subject_neighbors"]),
            object_neighbors=list(row["object_neighbors"]),
            subject_types=tuple(row["subject_types"]),
            object_types=tuple(row["object_types"]),
            negative_candidates=list(row.get("negative_candidates", [])),
            extra=dict(row.get("extra", {})),
        )
