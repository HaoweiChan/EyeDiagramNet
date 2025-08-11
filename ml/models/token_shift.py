import math
import torch
import torch.nn as nn


class FourierFeat(nn.Module):
    """
    Fourier feature encoding for positional embedding of theta values.

    Projects input theta values into a higher-dimensional space using
    sinusoidal functions at multiple frequency bands (2^k).

    Args:
        in_dim: Dimensionality of input theta values
        num_f: Number of frequency bands to use
    """

    def __init__(self, in_dim: int, num_f: int):
        super().__init__()
        if in_dim <= 0 or num_f <= 0:
            raise ValueError("in_dim and num_f must be positive integers")
        self.in_dim = in_dim
        self.num_f = num_f
        freqs = torch.tensor([2**k for k in range(num_f)], dtype=torch.float32)
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Compute Fourier features for input theta values.

        Args:
            theta: Tensor of shape (B, in_dim)

        Returns:
            Tensor of shape (B, in_dim * 2 * num_f) with concatenated
            [sin(theta*f_k), cos(theta*f_k)] per frequency band
        """
        if theta.dim() != 2 or theta.size(1) != self.in_dim:
            raise ValueError(
                f"Expected theta of shape (B, {self.in_dim}), got {tuple(theta.shape)}"
            )
        batch_size = theta.size(0)

        # Match buffer dtype/device to input to enable autocast/half precision
        freqs = self.freqs.to(device=theta.device, dtype=theta.dtype)

        # Compute projections with broadcast; use reshape for compile-friendliness
        x = theta.reshape(batch_size, self.in_dim, 1) * freqs.reshape(1, 1, self.num_f)
        sin_features = torch.sin(x)
        cos_features = torch.cos(x)

        # Concatenate along frequency axis and flatten
        features = torch.cat((sin_features, cos_features), dim=2)
        features = features.reshape(batch_size, self.in_dim * 2 * self.num_f)
        return features


class DeltaTokenModulator(nn.Module):
    """
    Token modulation network that predicts token shifts based on theta differences.

    Given reference tokens and two theta conditions (reference, target), predicts
    a per-token delta that, when added to the reference tokens, produces tokens
    consistent with the target theta.

    Args:
        token_dim: Dimensionality of token embeddings
        theta_dim: Dimensionality of theta vectors
        hidden: Hidden size for MLPs
        num_f: Number of Fourier frequency bands for theta encoding
    """

    def __init__(self, token_dim: int, theta_dim: int, hidden: int, num_f: int):
        super().__init__()
        if token_dim <= 0 or theta_dim <= 0 or hidden <= 0 or num_f <= 0:
            raise ValueError("All dimensions must be positive integers")

        self.token_dim = token_dim
        self.theta_encoder = FourierFeat(theta_dim, num_f)
        theta_feat_dim = theta_dim * 2 * num_f

        # Encode theta delta into a compact context vector
        self.theta_mlp = nn.Sequential(
            nn.Linear(theta_feat_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )

        # Light attention: context as a global key over per-token queries
        self.query_proj = nn.Linear(token_dim, hidden)
        self.ctx_key = nn.Linear(hidden, hidden)
        self.ctx_to_token = nn.Linear(hidden, token_dim)

        # Precompute attention scale (1/sqrt(H)) to avoid repeated sqrt
        self.register_buffer("attn_scale", torch.tensor(1.0 / math.sqrt(hidden), dtype=torch.float32), persistent=False)

        # Predict delta purely from context so that when ctx == 0 (theta_ref == theta_tgt), delta == 0
        self.delta_mlp = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )

        # Global gate initialized to 0 -> start conservative
        self.gate = nn.Parameter(torch.zeros(1))

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        z_ref: torch.Tensor,
        theta_ref: torch.Tensor,
        theta_tgt: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            z_ref: (B, N, token_dim) reference tokens
            theta_ref: (B, theta_dim) reference theta
            theta_tgt: (B, theta_dim) target theta

        Returns:
            z_hat: (B, N, token_dim) predicted target tokens
        """
        if z_ref.dim() != 3 or z_ref.size(-1) != self.token_dim:
            raise ValueError(
                f"Expected z_ref of shape (B, N, {self.token_dim}), got {tuple(z_ref.shape)}"
            )
        batch_size, num_tokens, _ = z_ref.shape

        if theta_ref.shape != (batch_size, self.theta_encoder.in_dim):
            raise ValueError(
                f"Expected theta_ref of shape ({batch_size}, {self.theta_encoder.in_dim}), got {tuple(theta_ref.shape)}"
            )
        if theta_tgt.shape != (batch_size, self.theta_encoder.in_dim):
            raise ValueError(
                f"Expected theta_tgt of shape ({batch_size}, {self.theta_encoder.in_dim}), got {tuple(theta_tgt.shape)}"
            )

        device, dtype = z_ref.device, z_ref.dtype
        theta_ref = theta_ref.to(device=device, dtype=dtype)
        theta_tgt = theta_tgt.to(device=device, dtype=dtype)

        # Encode thetas and form delta features
        feat_ref = self.theta_encoder(theta_ref)
        feat_tgt = self.theta_encoder(theta_tgt)
        delta_feat = feat_tgt - feat_ref
        ctx = self.theta_mlp(delta_feat)  # (B, hidden)

        # Attention weights over tokens using context as a key
        q = self.query_proj(z_ref)  # (B, N, hidden)
        k = self.ctx_key(ctx).unsqueeze(2)  # (B, hidden, 1)
        # Efficient batched matmul for attention scores
        scores = torch.bmm(q, k).squeeze(2)  # (B, N)
        scale = self.attn_scale.to(device=q.device, dtype=q.dtype)
        scores = scores * scale
        alpha = torch.softmax(scores, dim=1).unsqueeze(2)  # (B, N, 1)

        # Broadcast context to token space and predict delta
        ctx_tok = self.ctx_to_token(ctx)  # (B, token_dim)
        delta_base = self.delta_mlp(ctx_tok)  # (B, token_dim)
        delta = delta_base.unsqueeze(1).expand(-1, num_tokens, -1)  # (B, N, token_dim)
        z_hat = z_ref + torch.sigmoid(self.gate) * alpha * delta
        return z_hat


