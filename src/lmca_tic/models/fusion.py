"""Adaptive gating for Eq. (3-9) and Eq. (3-10)."""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Sequence

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    nn = SimpleNamespace(Module=object)

from lmca_tic.config.schemas import ModelConfig
from lmca_tic.utils.deps import require_dependency


_BaseModule = nn.Module if hasattr(nn, "Module") else object


def normalize_modal_weights(values: Sequence[float]) -> list[float]:
    exp_values = [math.exp(v) for v in values]
    total = sum(exp_values) or 1.0
    return [value / total for value in exp_values]


class AdaptiveFusion(_BaseModule):
    def __init__(self, config: ModelConfig) -> None:
        require_dependency(torch, "torch")
        super().__init__()
        self.use_gate = config.use_gate
        self.gate = nn.Sequential(
            nn.Linear(config.embedding_dim * 3, config.fusion_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.fusion_hidden_dim, config.fusion_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.fusion_hidden_dim, 3),
        )

    def forward(self, text_embed, time_embed, struct_embed):
        if not self.use_gate:
            return (text_embed + time_embed + struct_embed) / 3.0, None
        concat = torch.cat([text_embed, time_embed, struct_embed], dim=-1)
        weights = torch.softmax(self.gate(concat), dim=-1)
        fused = (
            weights[:, 0:1] * text_embed
            + weights[:, 1:2] * time_embed
            + weights[:, 2:3] * struct_embed
        )
        return fused, weights
