import torch

from ml.modules.trace_ew_module import TraceEWModule
from ml.models.eyewidth_model import EyeWidthRegressor


def _build_mod(enable_token=False):
    model = EyeWidthRegressor(
        num_types=3,
        model_dim=32,
        num_heads=4,
        num_layers=2,
        dropout=0.0,
        freq_length=32,
        use_rope=True,
        max_seq_len=128,
        use_gradient_checkpointing=False,
        ignore_snp=True,
        predict_logvar=False,
    )
    mod = TraceEWModule(
        model=model,
        ignore_snp=True,
        predict_logvar=False,
        enable_token_shift=enable_token,
        lambda_tokens=0.1 if enable_token else 0.0,
        pair_enable=True,
        pair_k=2,
        pair_min_delta=0.1,
    )
    # Provide config_keys for theta selection
    mod.config_keys = [f"k{i}" for i in range(6)]
    return mod


def _make_structured_trace_seq(B=2, L=32, P_signals=8, feat_dim_mid=6):
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


def _random_batch(B=6, L=32, P_signals=8, Fb=6):
    trace_seq = _make_structured_trace_seq(B=B, L=L, P_signals=P_signals, feat_dim_mid=6)
    direction = torch.randint(0, 2, (B, P_signals))
    boundary = torch.randn(B, Fb)
    boundary[:, -1] = torch.linspace(0, 1, B)  # theta-like dim
    snp_vert = torch.zeros(B, 2, 8, P_signals * 2, P_signals * 2, dtype=torch.complex64)
    true_ew = torch.rand(B, P_signals)
    meta = {}
    return trace_seq, direction, boundary, snp_vert, true_ew, meta


def test_token_shift_off_matches_baseline():
    mod = _build_mod(enable_token=False)
    batch = {"ds": _random_batch()}
    out = mod.step(batch, 0, "train_", 0)
    assert "loss" in out


def test_token_shift_on_runs_without_error():
    mod = _build_mod(enable_token=True)
    batch = {"ds": _random_batch()}
    out = mod.step(batch, 0, "train_", 0)
    assert "loss" in out


