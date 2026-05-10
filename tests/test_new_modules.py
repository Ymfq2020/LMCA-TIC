"""Tests for the new time encoder and time-aware relation modules."""

from __future__ import annotations

import math

import pytest


torch = pytest.importorskip("torch")

from lmca_tic.config.schemas import ModelConfig
from lmca_tic.models.fusion import AdaptiveFusion
from lmca_tic.models.scoring import BilinearScorer
from lmca_tic.models.temporal_graph import TemporalGraphEncoder
from lmca_tic.models.text_encoder import masked_mean_pool
from lmca_tic.models.time_encoder import SinusoidalTimeEncoder, TimeAwareRelation


def test_sinusoidal_time_encoder_outputs_expected_shape():
    encoder = SinusoidalTimeEncoder(dim=8)
    timestamps = torch.tensor([0.0, 1.0, 100.0, 365.0])
    encoded = encoder(timestamps)
    assert encoded.shape == (4, 8)
    # Even/odd index pattern: sin(0)==0, cos(0)==1
    assert encoded[0, 0].item() == pytest.approx(0.0, abs=1e-6)
    assert encoded[0, 1].item() == pytest.approx(1.0, abs=1e-6)
    assert encoded[0, 2].item() == pytest.approx(0.0, abs=1e-6)
    assert encoded[0, 3].item() == pytest.approx(1.0, abs=1e-6)


def test_time_aware_relation_combines_relation_and_time():
    embedding = torch.nn.Embedding(3, 8)
    module = TimeAwareRelation(num_relations=3, embedding_dim=8, time_encoding_dim=8, relation_embedding=embedding)
    relation_ids = torch.tensor([0, 1, 2])
    time_features = torch.zeros(3, 8)
    output = module(relation_ids, time_features)
    assert output.shape == (3, 8)


def test_temporal_graph_encoder_forward_shapes():
    config = ModelConfig(
        embedding_dim=16,
        time_encoding_dim=8,
        tgn_neighbor_size=4,
        tgn_memory_dim=16,
        tgn_time_decay_init=0.1,
        fusion_hidden_dim=8,
        use_temporal=True,
        use_tgn=True,
        use_gate=True,
        use_4bit=False,
    )
    relation_embedding = torch.nn.Embedding(5, 16)
    time_encoder = SinusoidalTimeEncoder(8)
    encoder = TemporalGraphEncoder(
        num_entities=10,
        config=config,
        relation_embedding=relation_embedding,
        time_encoder=time_encoder,
    )
    entity_ids = torch.tensor([0, 1])
    neighbor_ids = torch.tensor([[2, 3, 0, 0], [4, 0, 0, 0]])
    neighbor_relation_ids = torch.tensor([[0, 1, 0, 0], [2, 0, 0, 0]])
    neighbor_deltas = torch.tensor([[1.0, 2.0, 0.0, 0.0], [3.0, 0.0, 0.0, 0.0]])
    query_text = torch.randn(2, 16)
    r_t = torch.randn(2, 16)
    output = encoder(
        entity_ids=entity_ids,
        neighbor_ids=neighbor_ids,
        neighbor_relation_ids=neighbor_relation_ids,
        neighbor_deltas=neighbor_deltas,
        query_text=query_text,
        r_t=r_t,
    )
    assert output.shape == (2, 16)


def test_adaptive_fusion_sigmoid_gate_returns_per_sample_weight():
    config = ModelConfig(embedding_dim=4, fusion_hidden_dim=4, use_gate=True, use_4bit=False)
    fusion = AdaptiveFusion(config)
    text = torch.randn(3, 4)
    struct = torch.randn(3, 4)
    fused, gate = fusion(text, struct)
    assert fused.shape == (3, 4)
    assert gate.shape == (3, 1)
    assert (gate >= 0).all() and (gate <= 1).all()


def test_adaptive_fusion_disabled_gate_falls_back_to_mean():
    config = ModelConfig(embedding_dim=4, fusion_hidden_dim=4, use_gate=False, use_4bit=False)
    fusion = AdaptiveFusion(config)
    text = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    struct = torch.tensor([[3.0, 3.0, 3.0, 3.0]])
    fused, gate = fusion(text, struct)
    assert torch.allclose(fused, torch.tensor([[2.0, 2.0, 2.0, 2.0]]))
    assert gate.shape == (1, 1)


def test_bilinear_scorer_uses_r_t():
    scorer = BilinearScorer(embedding_dim=4)
    subject = torch.ones(2, 4)
    r_t = torch.zeros(2, 4)
    obj = torch.ones(2, 4)
    score_no_relation = scorer(subject, r_t, obj)
    r_t_nonzero = torch.full((2, 4), 0.5)
    score_with_relation = scorer(subject, r_t_nonzero, obj)
    # Non-zero r_t must change the resulting bilinear score.
    assert not torch.allclose(score_no_relation, score_with_relation)


def test_masked_mean_pool_ignores_padding_tokens():
    hidden_states = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [100.0, 100.0]]])
    attention_mask = torch.tensor([[1, 1, 0]])
    pooled = masked_mean_pool(hidden_states, attention_mask)
    assert torch.allclose(pooled, torch.tensor([[2.0, 2.0]]))
