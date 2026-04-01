"""Temporal graph encoder for Eq. (3-6) to Eq. (3-8)."""

from __future__ import annotations

import math
from types import SimpleNamespace

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    nn = SimpleNamespace(Module=object)

try:
    from torch_geometric.utils import softmax as pyg_softmax
except ImportError:  # pragma: no cover - optional dependency
    pyg_softmax = None

from lmca_tic.config.schemas import ModelConfig
from lmca_tic.utils.deps import require_dependency


_BaseModule = nn.Module if hasattr(nn, "Module") else object


class TemporalGraphEncoder(_BaseModule):
    def __init__(self, num_entities: int, config: ModelConfig) -> None:
        require_dependency(torch, "torch")
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(num_entities, config.embedding_dim)
        self.memory = nn.Embedding(num_entities, config.tgn_memory_dim)
        self.query_proj = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.key_proj = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.value_proj = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.gru = nn.GRUCell(config.embedding_dim, config.tgn_memory_dim)
        self.out_proj = nn.Linear(config.embedding_dim + config.tgn_memory_dim, config.embedding_dim)
        self.time_decay = nn.Parameter(torch.tensor(float(config.tgn_time_decay_init)))

    def forward(
        self,
        entity_ids,
        neighbor_ids,
        neighbor_deltas,
    ):
        center = self.embedding(entity_ids)
        base_memory = self.memory(entity_ids)
        if not self.config.use_tgn:
            return center
        if neighbor_ids.numel() == 0:
            updated = self.gru(center, base_memory)
            return self.out_proj(torch.cat([center, updated], dim=-1))

        neighbor_emb = self.embedding(neighbor_ids)
        query = self.query_proj(center).unsqueeze(1)
        key = self.key_proj(neighbor_emb)
        value = self.value_proj(neighbor_emb)

        raw_attn = (query * key).sum(dim=-1) / math.sqrt(center.size(-1))
        raw_attn = raw_attn * torch.exp(-torch.abs(self.time_decay) * neighbor_deltas)
        if pyg_softmax is not None:
            batch_index = torch.arange(
                raw_attn.size(0),
                device=raw_attn.device,
            ).repeat_interleave(raw_attn.size(1))
            attn = pyg_softmax(raw_attn.reshape(-1), batch_index).reshape_as(raw_attn)
        else:
            attn = torch.softmax(raw_attn, dim=-1)

        aggregated = (attn.unsqueeze(-1) * value).sum(dim=1)
        if not self.config.use_ni:
            aggregated = torch.zeros_like(aggregated)
        memory_input = aggregated
        if self.config.use_sl:
            updated_memory = self.gru(memory_input, base_memory)
        else:
            updated_memory = base_memory
        if self.config.use_gs:
            global_signal = neighbor_emb.mean(dim=1)
            aggregated = aggregated + global_signal
        return self.out_proj(torch.cat([aggregated, updated_memory], dim=-1))
