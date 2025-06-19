import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class SNPDecoder(nn.Module):
    """Memory-optimized decoder for SNP reconstruction"""
    
    def __init__(
        self,
        model_dim,
        freq_length,
        output_freq_length=None,
        decoder_hidden_ratio=2,
        use_checkpointing=False,
        use_mixed_precision=True
    ):
        super().__init__()
        self.model_dim = model_dim
        self.freq_length = freq_length
        self.output_freq_length = output_freq_length or freq_length
        self.use_checkpointing = use_checkpointing
        self.use_mixed_precision = use_mixed_precision
        
        # Decode from embedding space back to frequency space
        # Use MLP to preserve port dimension
        self.embed_decoder = nn.Sequential(
            nn.Linear(model_dim, model_dim * decoder_hidden_ratio),
            nn.GELU(),
            nn.Linear(model_dim * decoder_hidden_ratio, freq_length)
        )
        
        # Project from frequency space to complex SNP space with better initialization
        self.snp_proj = nn.Linear(freq_length, self.output_freq_length * 2)
        nn.init.xavier_uniform_(self.snp_proj.weight)
        nn.init.zeros_(self.snp_proj.bias)
        
        # Power transformation for inverse
        self.register_buffer('_power', torch.tensor(4.0))
    
    def _inverse_snp_transform(self, x):
        """Inverse power transformation with mixed precision"""
        if self.use_mixed_precision and x.is_cuda:
            with torch.amp.autocast(enabled=True):
                return x.sign() * torch.pow(x.abs() + 1e-8, self._power)
        else:
            return x.sign() * torch.pow(x.abs() + 1e-8, self._power)
    
    def _decode_chunk(self, hidden_states, b, half_p):
        """Process decoding in chunks for memory efficiency"""
        # hidden_states shape: (b, half_p, model_dim)
        # Apply decoder to each port embedding
        b_p, e = hidden_states.shape[0], hidden_states.shape[-1]
        
        # Decode from embedding to frequency space
        if self.use_mixed_precision and hidden_states.is_cuda:
            with torch.amp.autocast(enabled=True):
                freq_features = self.embed_decoder(hidden_states.view(-1, e))  # (b*half_p, freq_length)
        else:
            freq_features = self.embed_decoder(hidden_states.view(-1, e))  # (b*half_p, freq_length)
        
        # Project to complex space
        snp_complex = self.snp_proj(freq_features)  # (b*half_p, output_freq_length*2)
        snp_complex = snp_complex.view(b, half_p, -1)  # (b, half_p, output_freq_length*2)
        
        return snp_complex
    
    def forward(self, hidden_states_snp):
        """
        Decode hidden states back to SNP data
        
        Args:
            hidden_states_snp: Encoded SNP embeddings of shape (B, P, E)
                              where P=num_ports/2, E=model_dim
                              Note: No tx/rx dimension for self-supervised learning
        
        Returns:
            snp_vert: Reconstructed SNP data of shape (B, F, P1, P2)
        """
        if hidden_states_snp.dim() == 4:
            # If input has tx/rx dimension (B, D, P, E), flatten it
            b, d, half_p, e = hidden_states_snp.size()
            hidden_states = hidden_states_snp.view(b * d, half_p, e)
            batch_size = b * d
        else:
            # Standard case for self-supervised learning (B, P, E)
            b, half_p, e = hidden_states_snp.size()
            hidden_states = hidden_states_snp
            batch_size = b
        
        # Use gradient checkpointing if enabled
        if self.use_checkpointing and self.training:
            snp_complex = torch.utils.checkpoint.checkpoint(
                self._decode_chunk, hidden_states, batch_size, half_p, use_reentrant=False
            )
        else:
            snp_complex = self._decode_chunk(hidden_states, batch_size, half_p)
        
        # Efficient reconstruction of full P x P matrix
        p = half_p * 2
        device = hidden_states.device
        
        # Pre-allocate output tensor
        snp_vert = torch.zeros(batch_size, self.output_freq_length, p, p, dtype=torch.complex64, device=device)
        
        # Reshape complex values
        snp_complex_reshaped = rearrange(
            snp_complex, "b p (f ri) -> b p f ri", 
            f=self.output_freq_length, ri=2
        )
        snp_complex_full = torch.view_as_complex(snp_complex_reshaped)
        
        # Fill the SNP matrix efficiently
        # Each port corresponds to specific rows in the SNP matrix
        snp_vert[:, :, :half_p, :] = snp_complex_full.permute(0, 2, 1).unsqueeze(3).expand(-1, -1, -1, p)
        snp_vert[:, :, half_p:, :] = snp_complex_full.permute(0, 2, 1).unsqueeze(3).expand(-1, -1, -1, p)
        
        # Apply inverse transformation on real view
        snp_vert_real = torch.view_as_real(snp_vert)
        snp_vert_real = self._inverse_snp_transform(snp_vert_real)
        snp_vert = torch.view_as_complex(snp_vert_real)
        
        # Reshape back to original batch dimension if needed
        if hidden_states_snp.dim() == 4:
            snp_vert = snp_vert.view(b, d, self.output_freq_length, p, p)
        
        return snp_vert 