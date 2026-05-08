"""Sinusoidal time encoding (Eq. 3-7) and time-aware relation (Eq. 3-8)."""

from __future__ import annotations

import math
from types import SimpleNamespace

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    nn = SimpleNamespace(Module=object)

from lmca_tic.utils.deps import require_dependency


_BaseModule = nn.Module if hasattr(nn, "Module") else object


class SinusoidalTimeEncoder(_BaseModule):
    """τ(t) following Eq. (3-7).

    For an even index 2j and odd index 2j+1 the values are
    sin(t / 10000^{2j / d_τ}) and cos(t / 10000^{2j / d_τ}) respectively.
    """

    def __init__(self, dim: int) -> None:
        require_dependency(torch, "torch")
        super().__init__()
        if dim <= 0:
            raise ValueError("time_encoding_dim must be positive")
        self.dim = int(dim)
        # Pre-compute the inverse frequencies once.
        even_indices = torch.arange(0, self.dim, 2, dtype=torch.float32)
        inv_freq = torch.exp(-math.log(10000.0) * even_indices / float(self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, timestamps):
        if not torch.is_tensor(timestamps):
            timestamps = torch.tensor(timestamps, dtype=torch.float32)
        original_shape = timestamps.shape
        flat = timestamps.reshape(-1).to(self.inv_freq.dtype)
        scaled = flat.unsqueeze(-1) * self.inv_freq.unsqueeze(0)
        encoded = torch.zeros(flat.shape[0], self.dim, dtype=self.inv_freq.dtype, device=flat.device)
        encoded[:, 0::2] = torch.sin(scaled)
        encoded[:, 1::2] = torch.cos(scaled)
        return encoded.reshape(*original_shape, self.dim)


class TimeAwareRelation(_BaseModule):
    """Eq. (3-8): r_t = W_r e_r + W_τ τ(t)."""

    def __init__(
        self,
        num_relations: int,
        embedding_dim: int,
        time_encoding_dim: int,
        relation_embedding: "nn.Embedding | None" = None,
    ) -> None:
        require_dependency(torch, "torch")
        super().__init__()
        if relation_embedding is None:
            relation_embedding = nn.Embedding(num_relations, embedding_dim)
        self.relation_embedding = relation_embedding
        self.relation_proj = nn.Linear(embedding_dim, embedding_dim)
        self.time_proj = nn.Linear(time_encoding_dim, embedding_dim)

    def forward(self, relation_ids, time_features):
        relation = self.relation_embedding(relation_ids)
        return self.relation_proj(relation) + self.time_proj(time_features)
