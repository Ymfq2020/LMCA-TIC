"""Adaptive gating for Eq. (3-16), (3-17), (3-18)."""

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
    """Sigmoid binary gate over (text, struct) per Eq. (3-17)/(3-18)."""

    def __init__(self, config: ModelConfig) -> None:
        require_dependency(torch, "torch")
        super().__init__()
        self.use_gate = config.use_gate
        self.gate = nn.Sequential(
            nn.Linear(config.embedding_dim * 2, config.fusion_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.fusion_hidden_dim, 1),
        )

    def forward(self, text_embed, struct_embed):
        if not self.use_gate:
            fused = 0.5 * text_embed + 0.5 * struct_embed
            weights = torch.full(
                (text_embed.size(0), 1),
                0.5,
                dtype=text_embed.dtype,
                device=text_embed.device,
            )
            return fused, weights
        # Eq. (3-16): z_x(t) = [e_x_text || e_x_struct(t)]
        concat = torch.cat([text_embed, struct_embed], dim=-1)
        # Eq. (3-17): g_x(t) = σ(w_g · z_x(t) + b_g)
        gate_weight = torch.sigmoid(self.gate(concat))
        # Eq. (3-18): e_final = g · e_text + (1 - g) · e_struct
        fused = gate_weight * text_embed + (1.0 - gate_weight) * struct_embed
        return fused, gate_weight
