"""Full LMCA-TIC model wiring text, time, structure and bilinear scoring."""

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
from lmca_tic.models.temporal_graph import TemporalGraphEncoder
from lmca_tic.models.text_encoder import LLMTextEncoder
from lmca_tic.models.time_encoder import SinusoidalTimeEncoder, TimeAwareRelation
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
        self.embedding_dim = int(config.embedding_dim)
        self.text_encoder = LLMTextEncoder(config, smoke_mode=smoke_mode)
        # Shared sinusoidal τ(·) encoder used for both query timestamp and
        # neighbor Δt features (Eq. 3-7, Eq. 3-10).
        self.time_encoder = SinusoidalTimeEncoder(config.time_encoding_dim)
        # Shared relation embedding so that r_t (Eq. 3-8), neighbor relation
        # features (Eq. 3-10) and the bilinear scorer all use one e_r table.
        self._shared_relation_embedding = nn.Embedding(num_relations, self.embedding_dim)
        self.relation_module = TimeAwareRelation(
            num_relations=num_relations,
            embedding_dim=self.embedding_dim,
            time_encoding_dim=config.time_encoding_dim,
            relation_embedding=self._shared_relation_embedding,
        )
        self.graph_encoder = TemporalGraphEncoder(
            num_entities=num_entities,
            config=config,
            relation_embedding=self._shared_relation_embedding,
            time_encoder=self.time_encoder,
        )
        self.fusion = AdaptiveFusion(config)
        self.scorer = BilinearScorer(embedding_dim=self.embedding_dim)
        # When the LLM branch is disabled we still need a stable text-side
        # representation for the gate input and the scorer; fall back to a
        # pure entity embedding so that downstream shapes line up.
        self.text_fallback_embedding = nn.Embedding(num_entities, self.embedding_dim)

    # ------------------------------------------------------------------
    # Time-aware relation helpers.
    # ------------------------------------------------------------------
    def relation_time_features(self, query_timestamps):
        if not torch.is_tensor(query_timestamps):
            query_timestamps = torch.tensor(
                query_timestamps,
                dtype=torch.float32,
                device=self._shared_relation_embedding.weight.device,
            )
        if self.config.use_temporal:
            return self.time_encoder(query_timestamps.float())
        return torch.zeros(
            (*query_timestamps.shape, self.config.time_encoding_dim),
            dtype=self._shared_relation_embedding.weight.dtype,
            device=query_timestamps.device,
        )

    def compute_r_t(self, relation_ids, query_timestamps):
        time_features = self.relation_time_features(query_timestamps)
        return self.relation_module(relation_ids, time_features)

    # ------------------------------------------------------------------
    # Entity encoding pipeline.
    # ------------------------------------------------------------------
    def encode_entities(
        self,
        prompts,
        entity_ids,
        neighbor_ids,
        neighbor_relation_ids,
        neighbor_deltas,
        r_t,
    ):
        device = entity_ids.device
        batch_size = int(entity_ids.shape[0])
        text_embed = self._encode_text(prompts, entity_ids, batch_size, device)
        if self.config.use_tgn:
            struct_embed = self.graph_encoder(
                entity_ids=entity_ids,
                neighbor_ids=neighbor_ids,
                neighbor_relation_ids=neighbor_relation_ids,
                neighbor_deltas=neighbor_deltas,
                query_text=text_embed,
                r_t=r_t,
            )
        else:
            struct_embed = torch.zeros_like(text_embed)
        if not self.config.use_llm:
            # Even with LLM removed we keep a non-zero text-side input for
            # the gate to compare against, otherwise (3-18) collapses.
            text_embed = self.text_fallback_embedding(entity_ids).to(text_embed.dtype)
        fused, gate_weights = self.fusion(text_embed, struct_embed)
        return fused, gate_weights

    def _encode_text(self, prompts, entity_ids, batch_size: int, device):
        if self.config.use_llm:
            embed = self.text_encoder(prompts)
            target_dtype = self._shared_relation_embedding.weight.dtype
            return embed.to(target_dtype) if embed.dtype != target_dtype else embed
        # Provide a deterministic non-zero text embedding even when the LLM
        # branch is disabled so the gating module still has comparable signal.
        return self.text_fallback_embedding(entity_ids)

    # ------------------------------------------------------------------
    # Forward (training) and scoring helper (used by the trainer).
    # ------------------------------------------------------------------
    def forward(self, batch: dict[str, object]) -> dict[str, object]:
        relation_ids = batch["relation_ids"]
        query_timestamps = batch["query_timestamps"]
        r_t = self.compute_r_t(relation_ids, query_timestamps)

        subject_embed, gate_weights = self.encode_entities(
            prompts=batch["subject_prompts"],
            entity_ids=batch["subject_ids"],
            neighbor_ids=batch["subject_neighbor_ids"],
            neighbor_relation_ids=batch["subject_neighbor_relation_ids"],
            neighbor_deltas=batch["subject_neighbor_deltas"],
            r_t=r_t,
        )
        positive_embed, _ = self.encode_entities(
            prompts=batch["positive_object_prompts"],
            entity_ids=batch["positive_object_ids"],
            neighbor_ids=batch["object_neighbor_ids"],
            neighbor_relation_ids=batch["object_neighbor_relation_ids"],
            neighbor_deltas=batch["object_neighbor_deltas"],
            r_t=r_t,
        )
        positive_scores = self.scorer(subject_embed, r_t, positive_embed)

        negative_scores = None
        negative_object_ids = batch["negative_object_ids"]
        if negative_object_ids.numel() > 0 and negative_object_ids.size(1) > 0:
            negative_count = negative_object_ids.size(1)
            flat_negative_ids = batch["negative_object_ids_flat"]
            flat_negative_prompts = batch["negative_object_prompts_flat"]
            negative_embed, _ = self.encode_entities(
                prompts=flat_negative_prompts,
                entity_ids=flat_negative_ids,
                neighbor_ids=batch["negative_neighbor_ids_flat"],
                neighbor_relation_ids=batch["negative_neighbor_relation_ids_flat"],
                neighbor_deltas=batch["negative_neighbor_deltas_flat"],
                r_t=r_t.repeat_interleave(negative_count, dim=0),
            )
            repeated_subject = subject_embed.repeat_interleave(negative_count, dim=0)
            repeated_r_t = r_t.repeat_interleave(negative_count, dim=0)
            negative_scores = self.scorer(repeated_subject, repeated_r_t, negative_embed).reshape(
                negative_object_ids.shape
            )
            negative_scores = negative_scores.masked_fill(~batch["negative_mask"], float("-inf"))

        return {
            "positive_scores": positive_scores,
            "negative_scores": negative_scores,
            "gate_weights": gate_weights,
            "r_t": r_t,
            "subject_embed": subject_embed,
        }
