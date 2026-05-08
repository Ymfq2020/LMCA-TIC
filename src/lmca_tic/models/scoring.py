"""Bilinear link scoring per Eq. (3-19)."""

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
    """z_j(s, r, t) = (W_s e_s_final(t) + r_t) · (W_o e_o_final(t))."""

    def __init__(self, embedding_dim: int) -> None:
        require_dependency(torch, "torch")
        super().__init__()
        self.subject_proj = nn.Linear(embedding_dim, embedding_dim)
        self.object_proj = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, subject_embed, time_aware_relation, object_embed):
        # Broadcast over candidate axes if needed.
        subject_part = self.subject_proj(subject_embed) + time_aware_relation
        object_part = self.object_proj(object_embed)
        return (subject_part * object_part).sum(dim=-1)
