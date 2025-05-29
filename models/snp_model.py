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
        b, d, f, p1, p2 = snp_vert.size()
        
        # Input validation moved to top for early exit
        if d != 2:
            raise ValueError("Invalid input shape: snp_vert must have 2 snp tensors (tx and rx) in dimension 1.")
        
        if p1 != p2:
            raise ValueError("SNP matrix must be square (P x P)")

        # Extract diagonal elements from the S-parameter matrix
        # We focus on the diagonal elements which represent reflection coefficients
        snp_diag = torch.diagonal(snp_vert, dim1=-2, dim2=-1)  # (B, D, F, P)
        
        # Optimized: Combine view_as_real and transform in single operation
        snp_diag = torch.view_as_real(snp_diag)  # (B, D, F, P, 2)
        snp_diag = self.snp_transform(snp_diag)  # (B, D, F, P, 2)

        # Reshape for interpolation: combine real/imaginary and prepare for frequency interpolation
        snp_diag = rearrange(snp_diag, "b d f p ri -> (b d p) (ri f)")
        
        # Interpolate frequency dimension
        snp_diag = F.interpolate(snp_diag.unsqueeze(1), size=self.freq_length * 2, mode='linear', align_corners=False)
        snp_diag = snp_diag.squeeze(1)  # (b*d*p, freq_length*2)
        
        # Apply projection
        snp_diag = self.snp_proj(snp_diag)  # (b*d*p, freq_length)
        
        # Add sequence dimension for AttentionPooling (expects 3D input)
        # Split freq_length into seq_len chunks to create sequence dimension
        seq_len = min(self.freq_length, 32)  # Use manageable sequence length
        chunk_size = self.freq_length // seq_len
        if chunk_size * seq_len < self.freq_length:
            # Pad to make it divisible
            padding_size = seq_len * chunk_size + chunk_size - self.freq_length
            snp_diag = F.pad(snp_diag, (0, padding_size))
            chunk_size = (self.freq_length + padding_size) // seq_len
        
        # Reshape to add sequence dimension
        snp_diag = snp_diag.view(-1, seq_len, chunk_size)  # (b*d*p, seq_len, chunk_size)

        # Forward to conditional embedding
        hidden_states_snp = self.snp_encoder(snp_diag)  # (b*d*p, model_dim)
        
        # Reshape back to desired format
        hidden_states_snp = rearrange(hidden_states_snp, "(b d p) e -> b d p e", b=b, d=d, p=p1)
        
        # Add learnable tokens for tx/rx distinction
        hidden_states_snp[:, 0].add_(self.tx_token.squeeze(0))  # Add tx token to first dimension
        hidden_states_snp[:, 1].add_(self.rx_token.squeeze(0))  # Add rx token to second dimension
        
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
        # Extract diagonal elements from the S-parameter matrix
        snp_chunk = torch.diagonal(snp_chunk, dim1=-2, dim2=-1)  # (B, D, F, P)
        
        # Convert to real and apply transform
        snp_chunk = torch.view_as_real(snp_chunk)  # (B, D, F, P, 2)
        snp_chunk = self.snp_transform(snp_chunk)  # (B, D, F, P, 2)
        
        # Efficient reshape and interpolation
        snp_chunk = rearrange(snp_chunk, "b d f p ri -> (b d p) (ri f)")
        snp_chunk = F.interpolate(snp_chunk.unsqueeze(1), size=self.freq_length * 2, mode='linear', align_corners=False)
        snp_chunk = snp_chunk.squeeze(1)  # (b*d*p, freq_length*2)
        
        snp_chunk = self.snp_proj(snp_chunk)
        
        # Add sequence dimension for AttentionPooling
        seq_len = min(self.freq_length, 32)
        chunk_size = self.freq_length // seq_len
        if chunk_size * seq_len < self.freq_length:
            padding_size = seq_len * chunk_size + chunk_size - self.freq_length
            snp_chunk = F.pad(snp_chunk, (0, padding_size))
            chunk_size = (self.freq_length + padding_size) // seq_len
        
        snp_chunk = snp_chunk.view(-1, seq_len, chunk_size)
        
        return snp_chunk

    def forward(self, snp_vert):
        """Memory-optimized forward pass with optional gradient checkpointing"""
        b, d, f, p1, p2 = snp_vert.size()
        
        if d != 2:
            raise ValueError("Invalid input shape: snp_vert must have 2 snp tensors (tx and rx) in dimension 1.")

        # Use gradient checkpointing for memory efficiency during training
        if self.use_checkpointing and self.training:
            snp_vert = torch.utils.checkpoint.checkpoint(
                self._process_snp_chunk, snp_vert, b, d, p1, use_reentrant=False
            )
        else:
            snp_vert = self._process_snp_chunk(snp_vert, b, d, p1)

        # Forward through encoder with mixed precision
        with torch.cuda.amp.autocast(enabled=self.use_mixed_precision):
            hidden_states_snp = self.snp_encoder(snp_vert)
        
        hidden_states_snp = rearrange(hidden_states_snp, "(b d p) e -> b d p e", b=b, d=d, p=p1)
        
        # In-place token addition
        hidden_states_snp[:, 0].add_(self.tx_token.view(1, -1))
        hidden_states_snp[:, 1].add_(self.rx_token.view(1, -1))
        
        return hidden_states_snp