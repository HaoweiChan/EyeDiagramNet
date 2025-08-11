import torch

from ml.models.token_shift import DeltaTokenModulator, FourierFeat


def test_fourier_feat_shapes_and_determinism():
    B, in_dim, num_f = 4, 3, 6
    enc = FourierFeat(in_dim=in_dim, num_f=num_f)

    theta = torch.randn(B, in_dim)
    out = enc(theta)
    assert out.shape == (B, in_dim * 2 * num_f)

    # Determinism for identical inputs
    out2 = enc(theta.clone())
    assert torch.allclose(out, out2)


def test_delta_token_modulator_identity_when_same_theta():
    B, N, token_dim = 2, 5, 16
    theta_dim, hidden, num_f = 4, 32, 4
    mod = DeltaTokenModulator(token_dim=token_dim, theta_dim=theta_dim, hidden=hidden, num_f=num_f)

    z = torch.randn(B, N, token_dim)
    theta = torch.randn(B, theta_dim)

    with torch.no_grad():
        z_hat = mod(z, theta, theta)
    # When theta_ref == theta_tgt, expect minimal delta (exact zero if gate is 0 init)
    assert torch.allclose(z_hat, z, atol=1e-6)


def test_delta_token_modulator_shape_and_grad():
    B, N, token_dim = 3, 7, 32
    theta_dim, hidden, num_f = 5, 64, 6
    mod = DeltaTokenModulator(token_dim=token_dim, theta_dim=theta_dim, hidden=hidden, num_f=num_f)

    z = torch.randn(B, N, token_dim, requires_grad=True)
    theta_ref = torch.randn(B, theta_dim)
    theta_tgt = torch.randn(B, theta_dim)

    z_hat = mod(z, theta_ref, theta_tgt)
    assert z_hat.shape == (B, N, token_dim)

    loss = z_hat.pow(2).mean()
    loss.backward()
    assert z.grad is not None


