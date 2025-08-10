import torch
import pytest

from ml.models.eyewidth_model import EyeWidthRegressor


def build_dummy_model(ignore_snp=True):
    model = EyeWidthRegressor(
        num_types=3,
        model_dim=32,
        num_heads=4,
        num_layers=2,
        dropout=0.0,
        freq_length=64,
        use_rope=True,
        max_seq_len=256,
        use_gradient_checkpointing=False,
        ignore_snp=ignore_snp,
        predict_logvar=True,
    )
    return model


def make_structured_trace_seq(B=2, L=32, P_signals=8, feat_dim_mid=4):
    """Create a structured sequence tensor matching TraceSequenceProcessor expectations.
    Layout per token: [layer, type, mid_features..., x_dim, z_dim]
    Ensure exactly P_signals tokens have type==0 (signals).
    """
    D = 2 + feat_dim_mid + 2  # layer, type, mid, x, z
    seq = torch.zeros(B, L, D)
    for b in range(B):
        types = torch.ones(L, dtype=torch.long)
        types[:P_signals] = 0
        seq[b, :, 1] = types
        layers = torch.arange(L) % 4
        seq[b, :, 0] = layers
        seq[b, :, 2 : 2 + feat_dim_mid] = torch.rand(L, feat_dim_mid)
        widths = torch.rand(L)
        cum_x = torch.cumsum(widths, dim=0)
        seq[b, :, -2] = cum_x - cum_x.roll(1, 0)
        seq[b, 0, -2] = 0.0
        seq[b, :, -1] = layers.float()
    return seq


def random_batch(B=2, L=32, P_signals=8, Fb=13, F=64):
    trace_seq = make_structured_trace_seq(B=B, L=L, P_signals=P_signals, feat_dim_mid=6)
    direction = torch.randint(0, 2, (B, P_signals))
    boundary = torch.randn(B, Fb)
    P_ports = P_signals * 2
    snp_vert = torch.randn(B, 2, F, P_ports, P_ports, dtype=torch.complex64)
    return trace_seq, direction, boundary, snp_vert


def test_encode_decode_matches_forward_ignore_snp():
    model = build_dummy_model(ignore_snp=True)
    model.eval()
    with torch.no_grad():
        trace_seq, direction, boundary, snp_vert = random_batch()
        v1, lv1, lg1 = model(trace_seq, direction, boundary, snp_vert)
        tokens = model.encode_trace_tokens(trace_seq)
        assert tokens.shape[1] == direction.shape[1]
        v2, lv2, lg2 = model.decode_from_tokens(tokens, direction, boundary, snp_vert)
    assert torch.allclose(v1, v2, atol=1e-5)
    assert torch.allclose(lv1, lv2, atol=1e-5)
    assert torch.allclose(lg1, lg2, atol=1e-5)


def test_encode_decode_matches_forward_with_snp():
    model = build_dummy_model(ignore_snp=False)
    model.eval()
    with torch.no_grad():
        trace_seq, direction, boundary, snp_vert = random_batch()
        v1, lv1, lg1 = model(trace_seq, direction, boundary, snp_vert)
        tokens = model.encode_trace_tokens(trace_seq)
        assert tokens.shape[1] == direction.shape[1]
        v2, lv2, lg2 = model.decode_from_tokens(tokens, direction, boundary, snp_vert)
    assert torch.allclose(v1, v2, atol=1e-5)
    assert torch.allclose(lv1, lv2, atol=1e-5)
    assert torch.allclose(lg1, lg2, atol=1e-5)
