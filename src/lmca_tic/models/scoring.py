"""Bilinear link scoring."""

from __future__ import annotations

from types import SimpleNamespace

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    nn = SimpleNamespace(Module=object)

from lmca_tic.utils.deps import require_dependency


_BaseModule = nn.Module if hasattr(nn, "Module") else object


class BilinearScorer(_BaseModule):
    def __init__(self, num_relations: int, embedding_dim: int) -> None:
        require_dependency(torch, "torch")
        super().__init__()
        self.relation_embedding = nn.Embedding(num_relations, embedding_dim)

    def forward(self, subject_embed, relation_ids, object_embed):
        relation = self.relation_embedding(relation_ids)
        return (subject_embed * relation * object_embed).sum(dim=-1)
