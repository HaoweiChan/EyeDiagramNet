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

    def _snp_transform(self, x):
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
        p = p1

        # Convert to real and apply power transformation
        snp_vert = torch.view_as_real(snp_vert)  # (B, D, F, P1, P2, 2)
        snp_vert = self._snp_transform(snp_vert)

        # Reshape snp_vert to prepare for linear interpolation
        snp_vert = rearrange(snp_vert, "b d f p1 p2 ri -> (b d p1 p2) ri f")
        snp_vert = F.interpolate(snp_vert, size=self.freq_length, mode='linear', align_corners=False)

        # Linearly project snp_vert from complex space to hidden space
        snp_vert = self.snp_proj(snp_vert.flatten(1))
        snp_vert = rearrange(snp_vert, "(b d p1 p2) e -> (b d) p1 (p2 e)", b=b, d=d, p1=p1, p2=p2)

        # Interleave in/out port information of a signal trace
        half_p = p1 // 2
        interleaved = torch.stack([snp_vert[:, :half_p], snp_vert[:, half_p:]], dim=2)
        snp_vert = rearrange(interleaved, "b p1 d (p2 e) -> (b p1) (d p2) e", p1=half_p, d=2, p2=p)

        # Forward snp_vert to conditional embedding to condense port interaction information
        hidden_states_snp = self.snp_encoder(snp_vert)
        hidden_states_snp = rearrange(hidden_states_snp, "(b p) e -> b d p e", b=b, d=d, p=half_p)

        # Add tx and rx tokens
        hidden_states_snp[:, 0].add_(self.tx_token)
        hidden_states_snp[:, 1].add_(self.rx_token)

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

    def _snp_transform(self, x):
        """Optimized power transformation"""
        with torch.cuda.amp.autocast(enabled=self.use_mixed_precision):
            return x.sign() * torch.pow(x.abs() + 1e-8, self._power_inv)

    def _process_snp_chunk(self, snp_chunk, b, d, p):
        """Process SNP data in chunks for memory efficiency"""
        # Convert to real and apply power transformation
        snp_chunk = torch.view_as_real(snp_chunk)
        snp_chunk = self._snp_transform(snp_chunk)
        
        # Reshape for linear interpolation
        snp_chunk = rearrange(snp_chunk, "b d f p1 p2 ri -> (b d p1 p2) ri f")
        snp_chunk = F.interpolate(snp_chunk, size=self.freq_length, mode='linear', align_corners=False)
        
        # Project from complex space to hidden space
        snp_chunk = self.snp_proj(snp_chunk.flatten(1))
        snp_chunk = rearrange(snp_chunk, "(b d p1 p2) e -> (b d) p1 (p2 e)", b=b, d=d, p1=p, p2=p)
        
        return snp_chunk

    def forward(self, snp_vert):
        """Memory-optimized forward pass with optional gradient checkpointing"""
        b, d, f, p1, p2 = snp_vert.size()
        
        if d != 2:
            raise ValueError("Invalid input shape: snp_vert must have 2 snp tensors (tx and rx) in dimension 1.")
        
        if p1 != p2:
            raise ValueError("SNP matrix must be square (P x P)")
        p = p1

        # Use gradient checkpointing for memory efficiency during training
        if self.use_checkpointing and self.training:
            snp_vert = torch.utils.checkpoint.checkpoint(
                self._process_snp_chunk, snp_vert, b, d, p1, use_reentrant=False
            )
        else:
            snp_vert = self._process_snp_chunk(snp_vert, b, d, p1)

        # Interleave in/out port information of a signal trace
        half_p = p1 // 2
        interleaved = torch.stack([snp_vert[:, :half_p], snp_vert[:, half_p:]], dim=2)
        snp_vert = rearrange(interleaved, "b p1 d (p2 e) -> (b p1) (d p2) e", p1=half_p, d=2, p2=p)

        # Forward through encoder with mixed precision
        with torch.cuda.amp.autocast(enabled=self.use_mixed_precision):
            hidden_states_snp = self.snp_encoder(snp_vert)
        
        hidden_states_snp = rearrange(hidden_states_snp, "(b d p) e -> b d p e", b=b, d=d, p=half_p)
        
        # Add tx and rx tokens (consistent with SNPEmbedding)
        hidden_states_snp[:, 0].add_(self.tx_token)
        hidden_states_snp[:, 1].add_(self.rx_token)
        
        return hidden_states_snp