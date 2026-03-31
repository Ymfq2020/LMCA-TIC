from lmca_tic.models.fusion import normalize_modal_weights


def test_modal_weight_normalization_sums_to_one():
    weights = normalize_modal_weights([0.0, 1.0, 2.0])
    assert round(sum(weights), 6) == 1.0
    assert max(weights) == weights[2]
