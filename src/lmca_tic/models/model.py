"""Full LMCA-TIC model."""

from __future__ import annotations

from types import SimpleNamespace

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    nn = SimpleNamespace(Module=object)

from lmca_tic.config.schemas import ModelConfig
from lmca_tic.models.fusion import AdaptiveFusion
from lmca_tic.models.scoring import BilinearScorer
from lmca_tic.models.tcn import RelationTCNEncoder
from lmca_tic.models.temporal_graph import TemporalGraphEncoder
from lmca_tic.models.text_encoder import LLMTextEncoder
from lmca_tic.utils.deps import require_dependency


_BaseModule = nn.Module if hasattr(nn, "Module") else object


class LMCATICModel(_BaseModule):
    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        config: ModelConfig,
        smoke_mode: bool = False,
    ) -> None:
        require_dependency(torch, "torch")
        super().__init__()
        self.config = config
        self.text_encoder = LLMTextEncoder(config, smoke_mode=smoke_mode)
        self.time_encoder = RelationTCNEncoder(config)
        self.graph_encoder = TemporalGraphEncoder(num_entities=num_entities, config=config)
        self.fusion = AdaptiveFusion(config)
        self.scorer = BilinearScorer(num_relations=num_relations, embedding_dim=config.embedding_dim)

    def encode_entities(
        self,
        prompts,
        relation_histories,
        entity_ids,
        neighbor_ids,
        neighbor_deltas,
    ):
        device = entity_ids.device
        batch_size = int(entity_ids.shape[0])
        text_embed = self.text_encoder(prompts) if self.config.use_llm else torch.zeros(
            (batch_size, self.config.embedding_dim),
            dtype=torch.float32,
            device=device,
        )
        time_embed = self.time_encoder(relation_histories) if self.config.use_tcn else torch.zeros_like(text_embed, device=device)
        struct_embed = self.graph_encoder(entity_ids, neighbor_ids, neighbor_deltas) if self.config.use_tgn else torch.zeros_like(text_embed, device=device)
        fused, weights = self.fusion(text_embed, time_embed, struct_embed)
        return fused, weights

    def forward(self, batch: dict[str, object]) -> dict[str, object]:
        subject_embed, gate_weights = self.encode_entities(
            prompts=batch["subject_prompts"],
            relation_histories=batch["relation_histories"],
            entity_ids=batch["subject_ids"],
            neighbor_ids=batch["subject_neighbor_ids"],
            neighbor_deltas=batch["subject_neighbor_deltas"],
        )
        positive_embed, _ = self.encode_entities(
            prompts=batch["positive_object_prompts"],
            relation_histories=batch["relation_histories"],
            entity_ids=batch["positive_object_ids"],
            neighbor_ids=batch["object_neighbor_ids"],
            neighbor_deltas=batch["object_neighbor_deltas"],
        )
        positive_scores = self.scorer(subject_embed, batch["relation_ids"], positive_embed)

        negative_scores = None
        if batch["negative_object_ids"].numel() > 0:
            flat_negative_ids = batch["negative_object_ids_flat"]
            flat_negative_prompts = batch["negative_object_prompts_flat"]
            repeated_histories = batch["relation_histories"].repeat_interleave(batch["negative_object_ids"].size(1), dim=0)
            negative_embed, _ = self.encode_entities(
                prompts=flat_negative_prompts,
                relation_histories=repeated_histories,
                entity_ids=flat_negative_ids,
                neighbor_ids=batch["negative_neighbor_ids_flat"],
                neighbor_deltas=batch["negative_neighbor_deltas_flat"],
            )
            repeated_subject = subject_embed.repeat_interleave(batch["negative_object_ids"].size(1), dim=0)
            repeated_relations = batch["relation_ids"].repeat_interleave(batch["negative_object_ids"].size(1), dim=0)
            negative_scores = self.scorer(repeated_subject, repeated_relations, negative_embed).reshape(batch["negative_object_ids"].shape)
            negative_scores = negative_scores.masked_fill(~batch["negative_mask"], 0.0)

        return {
            "positive_scores": positive_scores,
            "negative_scores": negative_scores,
            "gate_weights": gate_weights,
        }
