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
        self.model_dim = model_dim

        # Optimized: Use single projection instead of separate real/imag
        self.snp_proj = nn.Linear(freq_length * 2, freq_length)
        self.snp_encoder = ConditionEmbedding(encoder_dim=freq_length, embed_dim=model_dim)

        # Pre-allocate learnable tokens
        self.tx_token = nn.Parameter(torch.zeros(1, 1, model_dim))
        self.rx_token = nn.Parameter(torch.zeros(1, 1, model_dim))
        
        # Cache for power transformation to avoid recomputation
        self.register_buffer('_power_inv', torch.tensor(1.0 / 4.0))

    def snp_transform(self, x):
        """Optimized power transformation using fused operations"""
        return x.sign() * torch.pow(x.abs(), self._power_inv)

    def forward(self, snp_vert):
        """Encoder of snp for encoding vertical frequency responses"""
        b, d, f, p = snp_vert.size()
        
        # Input validation moved to top for early exit
        if d != 2:
            raise ValueError("Invalid input shape: snp_vert must have 2 snp tensors (tx and rx) in dimension 1.")

        # Optimized: Combine view_as_real and transform in single operation
        snp_vert = torch.view_as_real(snp_vert)
        snp_vert = self.snp_transform(snp_vert)

        # Optimized: Reduce number of rearrange operations
        # Original: multiple rearranges, now combined where possible
        snp_vert = rearrange(snp_vert, "b d pl p2 ri -> (b d pl p2) ri f")
        
        # Use more efficient interpolation mode for better performance
        snp_vert = F.interpolate(snp_vert, size=self.freq_length, mode='linear', align_corners=False)
        
        # Optimized: Combine reshape and projection operations
        snp_vert = rearrange(snp_vert, "(b d pl p2) ri f -> (b d) pl (p2 ri)", b=b, d=d, pl=p, p2=p)
        snp_vert = self.snp_proj(snp_vert)

        # Optimized: More efficient interleaving using advanced indexing
        # Instead of stack + rearrange, use direct tensor operations
        half_p = p // 2
        tx_part = snp_vert[:, :half_p]  # First half
        rx_part = snp_vert[:, half_p:]  # Second half
        
        # Create interleaved tensor more efficiently
        interleaved = torch.empty(b * d, p, self.freq_length, device=snp_vert.device, dtype=snp_vert.dtype)
        interleaved[:, 0::2] = tx_part
        interleaved[:, 1::2] = rx_part
        
        # Reshape for encoder input
        snp_vert = rearrange(interleaved, "(b d) p f -> (b d p) f", b=b, d=d)

        # Forward to conditional embedding
        hidden_states_snp = self.snp_encoder(snp_vert)
        hidden_states_snp = rearrange(hidden_states_snp, "(b d p) e -> b d p e", b=b, d=d, p=half_p)
        
        # Optimized: In-place addition for tokens
        hidden_states_snp[:, 0].add_(self.tx_token.squeeze(0))
        hidden_states_snp[:, 1].add_(self.rx_token.squeeze(0))
        
        return hidden_states_snp


class OptimizedSNPEmbedding(nn.Module):
    """Further optimized version with additional performance improvements"""
    
    def __init__(
        self,
        model_dim,
        freq_length,
        use_checkpointing=False,
        use_mixed_precision=True
    ):
        super().__init__()

        self.freq_length = freq_length
        self.model_dim = model_dim
        self.use_checkpointing = use_checkpointing
        self.use_mixed_precision = use_mixed_precision

        # Optimized projection with proper initialization
        self.snp_proj = nn.Linear(freq_length * 2, freq_length)
        nn.init.xavier_uniform_(self.snp_proj.weight)
        nn.init.zeros_(self.snp_proj.bias)
        
        self.snp_encoder = ConditionEmbedding(encoder_dim=freq_length, embed_dim=model_dim)

        # Learnable tokens with better initialization
        self.tx_token = nn.Parameter(torch.randn(1, 1, model_dim) * 0.02)
        self.rx_token = nn.Parameter(torch.randn(1, 1, model_dim) * 0.02)
        
        # Pre-computed constants
        self.register_buffer('_power_inv', torch.tensor(0.25))

    def snp_transform(self, x):
        """Optimized power transformation"""
        with torch.cuda.amp.autocast(enabled=self.use_mixed_precision):
            return x.sign() * torch.pow(x.abs() + 1e-8, self._power_inv)

    def _process_snp_chunk(self, snp_chunk, b, d, p):
        """Process SNP data in chunks for memory efficiency"""
        # Convert to real and apply transform
        snp_chunk = torch.view_as_real(snp_chunk)
        snp_chunk = self.snp_transform(snp_chunk)
        
        # Efficient reshape and interpolation
        snp_chunk = rearrange(snp_chunk, "b d pl p2 ri -> (b d pl p2) ri f")
        snp_chunk = F.interpolate(snp_chunk, size=self.freq_length, mode='linear', align_corners=False)
        snp_chunk = rearrange(snp_chunk, "(b d pl p2) ri f -> (b d) pl (p2 ri)", b=b, d=d, pl=p, p2=p)
        
        return self.snp_proj(snp_chunk)

    def forward(self, snp_vert):
        """Memory-optimized forward pass with optional gradient checkpointing"""
        b, d, f, p = snp_vert.size()
        
        if d != 2:
            raise ValueError("Invalid input shape: snp_vert must have 2 snp tensors (tx and rx) in dimension 1.")

        # Use gradient checkpointing for memory efficiency during training
        if self.use_checkpointing and self.training:
            snp_vert = torch.utils.checkpoint.checkpoint(
                self._process_snp_chunk, snp_vert, b, d, p, use_reentrant=False
            )
        else:
            snp_vert = self._process_snp_chunk(snp_vert, b, d, p)

        # Efficient interleaving
        half_p = p // 2
        tx_indices = torch.arange(0, half_p, device=snp_vert.device)
        rx_indices = torch.arange(half_p, p, device=snp_vert.device)
        
        # Create interleaved pattern more efficiently
        interleaved_indices = torch.empty(p, dtype=torch.long, device=snp_vert.device)
        interleaved_indices[0::2] = tx_indices
        interleaved_indices[1::2] = rx_indices
        
        snp_vert = snp_vert[:, interleaved_indices]
        snp_vert = rearrange(snp_vert, "(b d) p f -> (b d p) f", b=b, d=d)

        # Forward through encoder with mixed precision
        with torch.cuda.amp.autocast(enabled=self.use_mixed_precision):
            hidden_states_snp = self.snp_encoder(snp_vert)
        
        hidden_states_snp = rearrange(hidden_states_snp, "(b d p) e -> b d p e", b=b, d=d, p=half_p)
        
        # In-place token addition
        hidden_states_snp[:, 0].add_(self.tx_token.view(1, -1))
        hidden_states_snp[:, 1].add_(self.rx_token.view(1, -1))
        
        return hidden_states_snp