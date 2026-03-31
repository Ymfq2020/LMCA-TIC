"""Temporal convolution encoder for Eq. (3-4) and Eq. (3-5)."""

from __future__ import annotations

from types import SimpleNamespace

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    nn = SimpleNamespace(Module=object)

from lmca_tic.config.schemas import ModelConfig
from lmca_tic.utils.deps import require_dependency


_BaseModule = nn.Module if hasattr(nn, "Module") else object


class CausalConvBlock(_BaseModule):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float = 0.2) -> None:
        require_dependency(torch, "torch")
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequence):
        padded = torch.nn.functional.pad(sequence, (self.left_padding, 0))
        hidden = self.conv(padded)
        hidden = hidden.transpose(1, 2)
        hidden = self.norm(hidden)
        hidden = torch.relu(hidden)
        hidden = self.dropout(hidden)
        return hidden.transpose(1, 2)


class RelationTCNEncoder(_BaseModule):
    def __init__(self, config: ModelConfig) -> None:
        require_dependency(torch, "torch")
        super().__init__()
        self.input_proj = nn.Conv1d(1, config.embedding_dim, kernel_size=1)
        self.blocks = nn.ModuleList(
            [
                CausalConvBlock(
                    channels=config.embedding_dim,
                    kernel_size=config.tcn_kernel_size,
                    dilation=dilation,
                )
                for dilation in config.tcn_dilations
            ]
        )
        self.output_proj = nn.Linear(config.embedding_dim, config.embedding_dim)

    def forward(self, histories):
        if histories.ndim == 2:
            histories = histories.unsqueeze(1)
        hidden = self.input_proj(histories.float())
        for block in self.blocks:
            residual = hidden
            hidden = block(hidden)
            hidden = hidden + residual
        pooled = hidden.mean(dim=-1)
        return self.output_proj(pooled)
