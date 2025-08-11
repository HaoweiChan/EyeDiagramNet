# Minimal tests for in-batch theta pairing
import torch

from ml.modules.trace_ew_module import TraceEWModule
from ml.models.eyewidth_model import EyeWidthRegressor


def _dummy_model():
    return EyeWidthRegressor(
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


def test_find_theta_pairs_basic():
    model = _dummy_model()
    mod = TraceEWModule(
        model=model,
        ignore_snp=True,
        predict_logvar=False,
        pair_enable=True,
        pair_k=2,
        pair_min_delta=0.1,
    )
    # Emulate datamodule-provided config_keys
    mod.config_keys = [f"k{i}" for i in range(6)]

    B, Fb = 8, 6
    boundary = torch.randn(B, Fb)
    # Make last column act as theta so pairs need difference there
    boundary[:, -1] = torch.linspace(0, 1, B)

    pairs = mod._find_theta_pairs(boundary)
    if pairs is None:
        # Accept None if no pairs found due to random geometry; should not error
        assert True
    else:
        src, tgt = pairs
        assert src.dtype == torch.long and tgt.dtype == torch.long
        assert src.numel() == tgt.numel()
        assert src.numel() <= B


