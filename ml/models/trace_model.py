from typing import Tuple
from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import positional_encoding_1d, cont_positional_encoding, RotaryTransformerEncoder
from ..data.processors import TraceSequenceProcessor

class TraceSeqTransformer(nn.Module):
    def __init__(
        self,
        num_types,
        model_dim,
        num_heads,
        num_layers,
        dropout,
        use_rope=True,
        max_seq_len=1000,
    ):
        super().__init__()

        self.model_dim = model_dim
        self.use_rope = use_rope
        self.type_projection = nn.Embedding(num_types, model_dim)
        self.input_projection = nn.LazyLinear(model_dim)

        # Mask parameter for invalid tokens
        self.mask_token = nn.Parameter(torch.zeros(1, 1, model_dim))

        # Layer projection using positional encoding
        layer_projection = positional_encoding_1d(model_dim, max_len=100)
        self.register_buffer('layer_projection', layer_projection)

        # Encoder - choose between RoPE and standard transformer
        if use_rope:
            self.seq_encoder = RotaryTransformerEncoder(
                d_model=model_dim,
                nhead=num_heads,
                num_layers=num_layers,
                dropout=dropout,
                max_seq_len=max_seq_len
            )
        else:
            # Standard transformer encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=model_dim,
                nhead=num_heads,
                activation='relu',
                batch_first=True,
                norm_first=True,
                dropout=dropout
            )
            self.seq_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, seq_input):
        """Encoder of sequence input for extracting information between line sequence"""
        # Create a mask to filter null token
        null_mask = (seq_input == -1).any(-1)
        seq_input[null_mask] = 0 # prevent error for embeddings

        # Split sequence using semantic processor
        layers, types, feats, spatials = TraceSequenceProcessor.split_for_model(seq_input)
        layers, types = layers.long(), types.long() # (B, L, 1)

        # Get embedding for layer, type, and feature info
        layer_embeds = self.layer_projection[layers.view(-1, 1)].view(*feats.size()[:-1], -1)  # (B, L, M)
        type_embeds = self.type_projection(types.view(-1, 1)).view(*feats.size()[:-1], -1)    # (B, L, E)
        feat_embeds = self.input_projection(feats)

        # Make continuous positional embedding for spatial coordinates
        x_coords, z_coords = spatials.chunk(2, dim=-1)
        x_embeds = cont_positional_encoding(x_coords.squeeze(-1), self.model_dim // 2)
        z_embeds = cont_positional_encoding(z_coords.squeeze(-1), self.model_dim // 2)
        pos_embeds = torch.cat([x_embeds, z_embeds], dim=-1)

        # Sum all the embeddings and mask those with null type
        valid_embeds = layer_embeds + type_embeds + feat_embeds + pos_embeds
        embeds = (~null_mask).unsqueeze(-1) * valid_embeds + null_mask.unsqueeze(-1) * self.mask_token

        # Forward trace sequence transformer
        hidden_states_seq = self.seq_encoder(embeds)

        # Extract signal port information (type == 0)
        bs, _, c = hidden_states_seq.size()
        masked_types = types.clone()
        masked_types[null_mask] = -1
        idx_sig_ports = (masked_types.squeeze(-1) == 0).nonzero(as_tuple=True)
        hidden_states_seq = hidden_states_seq[idx_sig_ports].view(bs, -1, c)  # (B, P, C)

        return hidden_states_seq

class TraceSeqPretrainModel(nn.Module):
    def __init__(
        self,
        num_types,
        model_dim,
        output_dim,
        num_layers,
        num_heads,
        dropout,
        *args,
        **kwargs
    ):
        super().__init__()

        # Capture the initialization arguments
        frame = inspect.currentframe()
        args, _, _, values = inspect.getargvalues(frame)
        self._init_args = {arg: values[arg] for arg in args if arg != "self"}

        # Vision model for single-channel
        self.vision_model = TraceSeqTransformer(
            model_name=model_name,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            **kwargs,
        )

        # A lightweight linear decoder
        # Maps embedding dim -> (patch_size * patch_size)
        # We assume single-channel => we only decode to patch_size * patch_size.
        self.patch_size = patch_size
        self.decoder = nn.Linear(embed_dim, patch_size * patch_size)

    def get_config(self):
        return self._init_args

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the SimMIMPretrainModel.

        Args:
            pixel_values (torch.Tensor): shape (B, L, H, W), single-channel.

        Returns:
            --- torch.Tensor: Reconstructed patches of shape (B, L, H, W).
        """
        B, L, H, W = pixel_values.shape
        pH = pW = self.patch_size
        nH, nW = H // pH, W // pW

        # 1) Extract embeddings (B, L, E)
        embeddings = self.vision_model(pixel_values)  # shape (B, L, E)

        # 2) Decode back to patches => shape (B, L * num_patches, patch_size * patch_size)
        recon_patches = self.decoder(embeddings)

        # 3) Reshape => (B, L, patch_size, patch_size)
        # recon patches = recon_patches.view(
        #   B, L, self.patch_size, self.patch_size
        # )
        recon_patches = rearrange(recon_patches, "B N (pH pW) -> B N pH pW", pH=pH, pW=pW)

        return recon_patches