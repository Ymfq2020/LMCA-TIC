"""Temporal graph encoder following Eq. (3-9), (3-10), (3-13), (3-15)."""

from __future__ import annotations

import math
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


class TemporalGraphEncoder(_BaseModule):
    """Time-aware TGN with attention-based neighbor aggregation."""

    def __init__(
        self,
        num_entities: int,
        config: ModelConfig,
        relation_embedding: "nn.Embedding | None" = None,
        time_encoder: "nn.Module | None" = None,
    ) -> None:
        require_dependency(torch, "torch")
        super().__init__()
        self.config = config
        self.embedding_dim = int(config.embedding_dim)
        self.time_encoding_dim = int(config.time_encoding_dim)
        self.entity_embedding = nn.Embedding(num_entities, self.embedding_dim)
        self.memory = nn.Embedding(num_entities, config.tgn_memory_dim)
        if relation_embedding is None:
            raise ValueError("TemporalGraphEncoder requires a shared relation embedding")
        self.relation_embedding = relation_embedding
        self.time_encoder = time_encoder
        # Eq. (3-9): q_x(t) = W_q [e_x_text || r_t] -> R^{embedding_dim}
        self.query_proj = nn.Linear(self.embedding_dim * 2, self.embedding_dim)
        # Eq. (3-10): c_xi(t) = [e_ui || e_ri || τ(Δti)] -> R^{2d + d_τ}
        context_dim = self.embedding_dim * 2 + self.time_encoding_dim
        self.key_proj = nn.Linear(context_dim, self.embedding_dim)
        self.value_proj = nn.Linear(context_dim, self.embedding_dim)
        # Eq. (3-15): GRU memory update
        self.gru = nn.GRUCell(self.embedding_dim, config.tgn_memory_dim)
        self.out_proj = nn.Linear(self.embedding_dim + config.tgn_memory_dim, self.embedding_dim)
        # Eq. (3-13): attention with learnable -γΔt decay term
        self.time_decay = nn.Parameter(torch.tensor(float(config.tgn_time_decay_init)))

    def forward(
        self,
        entity_ids,
        neighbor_ids,
        neighbor_relation_ids,
        neighbor_deltas,
        query_text,
        r_t,
    ):
        center = self.entity_embedding(entity_ids)
        base_memory = self.memory(entity_ids)

        if neighbor_ids.numel() == 0 or neighbor_ids.size(1) == 0:
            zero_msg = torch.zeros_like(center)
            updated = self.gru(zero_msg, base_memory) if self.config.use_sl else base_memory
            return self.out_proj(torch.cat([center, updated], dim=-1))

        # Eq. (3-10): neighbor context features
        neighbor_emb = self.entity_embedding(neighbor_ids)
        neighbor_rel_emb = self.relation_embedding(neighbor_relation_ids)
        if self.time_encoder is not None and self.config.use_temporal:
            neighbor_time_emb = self.time_encoder(neighbor_deltas.float())
        else:
            neighbor_time_emb = torch.zeros(
                (*neighbor_deltas.shape, self.time_encoding_dim),
                dtype=center.dtype,
                device=center.device,
            )
        context = torch.cat([neighbor_emb, neighbor_rel_emb, neighbor_time_emb], dim=-1)

        # Eq. (3-9): query from [e_text || r_t]
        query = self.query_proj(torch.cat([query_text, r_t], dim=-1)).unsqueeze(1)
        key = self.key_proj(context)
        value = self.value_proj(context)

        # Eq. (3-13): additive time decay term
        scaled_dot = (query * key).sum(dim=-1) / math.sqrt(self.embedding_dim)
        attention_logits = scaled_dot - torch.abs(self.time_decay) * neighbor_deltas.float()
        attention = torch.softmax(attention_logits, dim=-1)

        aggregated = (attention.unsqueeze(-1) * value).sum(dim=1)
        if not self.config.use_ni:
            aggregated = torch.zeros_like(aggregated)

        # Optional GS branch (kept for micro ablation V1/V4)
        if self.config.use_gs:
            global_signal = neighbor_emb.mean(dim=1)
            aggregated = aggregated + global_signal

        # Eq. (3-15): GRU memory update; SL switch keeps the legacy self-loop branch.
        if self.config.use_sl:
            updated_memory = self.gru(aggregated, base_memory)
        else:
            updated_memory = base_memory

        return self.out_proj(torch.cat([aggregated, updated_memory], dim=-1))
