from __future__ import annotations

import torch

from src.models.components.spint import MultiLayerCrossAttention
from src.models.components.streaming_encoders import SideFeatureEarlyPoolEncoder
from src.models.components.streaming_spint import StreamingSpintModel
from src.models.components.spint import SpintModel


def test_zero_gamma_exactly_recovers_unbiased_attention():
    torch.manual_seed(7)
    attention = MultiLayerCrossAttention(2, 8, 2, dim_feedforward=16, dropout=0.0).eval()
    query, key_value = torch.randn(3, 2, 8), torch.randn(3, 5, 8)
    baseline, _ = attention(query, key_value)
    biased, _ = attention(
        query, key_value, attention_logit_bias=torch.randn(3, 5),
        logit_bias_gammas=torch.zeros(2),
    )
    assert torch.equal(baseline, biased)


def test_bias_is_unit_aligned_and_broadcast_over_heads_and_queries():
    torch.manual_seed(11)
    attention = MultiLayerCrossAttention(1, 8, 2, dim_feedforward=16, dropout=0.0).eval()
    query, key_value = torch.randn(2, 3, 8), torch.randn(2, 4, 8)
    q = torch.tensor([[0.0, 2.0, -1.0, 0.5], [1.0, -2.0, 0.0, 0.25]])
    observed, _ = attention(query, key_value, attention_logit_bias=q, logit_bias_gammas=torch.ones(1))
    layer = attention.layers[0]
    mask = q[:, None, None, :].expand(-1, layer.cross_attn.num_heads, query.shape[1], -1).reshape(-1, query.shape[1], key_value.shape[1])
    expected, _ = attention(query, key_value, attn_mask=mask)
    assert torch.equal(observed, expected)


def test_e04_splits_five_column_carrier_before_b3s_encoder():
    torch.manual_seed(3)
    decoder = SpintModel(model_dim=8, num_covariates=2, window_size=4, num_heads=2, num_layers=1, num_id_layers=1, tf_drop_rate=0.0)
    encoder = SideFeatureEarlyPoolEncoder(trial_length=6, window_size=4, hidden_dim=8, side_dim=4)
    model = StreamingSpintModel(decoder, encoder, reliability_logit_bias=True).eval()
    neural = torch.randn(2, 4, 3)
    calibration = torch.randn(2, 3, 6, 3)
    carrier = torch.randn(2, 3, 5)
    output, identity = model(neural, calib_trials=calibration, side_features=carrier)
    assert output.shape == (2, 4, 2)
    assert identity.shape == (2, 3, 4)
    assert encoder.side_dim == 4
    assert model.reliability_logit_gammas().shape == (1,)
    assert all(parameter.requires_grad for name, parameter in model.named_parameters() if "reliability_logit_gamma" in name)
