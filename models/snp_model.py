import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .layers import ConditionEmbedding

class SNPEmbedding(nn.Module):
    def __init__(
        self,
        model_dim,
        freq_length
    ):
        super().__init__()

        self.freq_length = freq_length

        self.snp_proj = nn.Linear(freq_length * 2, freq_length)
        self.snp_encoder = ConditionEmbedding(encoder_dim=freq_length, embed_dim=model_dim)

        self.tx_token = nn.Parameter(torch.zeros(1, 1, model_dim))
        self.rx_token = nn.Parameter(torch.zeros(1, 1, model_dim))

    def snp_transform(self, x, power=4):
        return x.sign() * x.abs() ** (1 / power)

    def forward(self, snp_vert):
        """Encoder of snp for encoding vertical frequency responses"""
        b, d, f, p = snp_vert.size()
        snp_vert = torch.view_as_real(snp_vert)
        snp_vert = self.snp_transform(snp_vert)

        if d != 2:
            raise ValueError("Invalid input shape: snp_vert must have 2 snp tensors (tx and rx) in dimension 1.")

        # Reshape snp vert to prepare for linear interpolation
        snp_vert = rearrange(snp_vert, "b d pl p2 ri -> (b d pl p2) ri f")
        snp_vert = F.interpolate(snp_vert, size=self.freq_length)
        snp_vert = rearrange(snp_vert, "(b d pl p2) e f -> (b d) pl (p2 e) f", b=b, d=d, pl=p, p2=p)

        # Linearly project snp vert from complex space to hidden space
        snp_vert = self.snp_proj(snp_vert.flatten(2))
        snp_vert = rearrange(snp_vert, "(b d) pl (p2 e) -> (b d) pl (p2 e)", b=b, d=d, pl=p, p2=p)

        # Interleave in/out for intuition of a signal trace
        interleaved = torch.stack((snp_vert[:, :p // 2], snp_vert[:, p // 2:]), dim=2)
        snp_vert = rearrange(interleaved, "b pl d (p2 e) -> (b pl d) (p2 e)", p1=(p // 2), p2=p)

        # Forward snp vert to conditional embedding to condense port interaction information
        hidden_states_snp = self.snp_encoder(snp_vert)
        hidden_states_snp = rearrange(hidden_states_snp, "b d p e -> b d p e", b=b, d=d, p=(p // 2))
        # hidden_states_snp = reduce(hidden_states_snp, "b d p e -> b d e", "sum") # add tx and rx snp states
        # hidden_states_snp = rearrange(hidden_states_snp, "b d e -> b (d e)") # concat tx and rx snp states
        hidden_states_snp[:, 0] = hidden_states_snp[:, 0] + self.tx_token
        hidden_states_snp[:, 1] = hidden_states_snp[:, 1] + self.rx_token
        return hidden_states_snp