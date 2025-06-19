import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .layers import ConditionEmbedding

class SNPEmbedding(nn.Module):
    def __init__(
        self,
        model_dim,
        freq_length,
        use_tx_rx_tokens=True
    ):
        super().__init__()

        self.freq_length = freq_length
        self.model_dim = model_dim
        self.use_tx_rx_tokens = use_tx_rx_tokens

        # Optimized: Use single projection instead of separate real/imag
        self.snp_proj = nn.Linear(freq_length * 2, freq_length)
        # Use appropriate number of heads based on freq_length
        # Ensure num_heads divides freq_length evenly
        max_heads = min(8, freq_length // 16)  # Max heads with at least 16 dims per head
        num_heads = max_heads
        while num_heads > 1 and freq_length % num_heads != 0:
            num_heads -= 1
        self.snp_encoder = ConditionEmbedding(encoder_dim=freq_length, embed_dim=model_dim, num_heads=num_heads)

        # Pre-allocate learnable tokens (only if using tx/rx tokens)
        if self.use_tx_rx_tokens:
            self.tx_token = nn.Parameter(torch.zeros(1, 1, model_dim))
            self.rx_token = nn.Parameter(torch.zeros(1, 1, model_dim))
        
        # Cache for power transformation to avoid recomputation
        self.register_buffer('_power_inv', torch.tensor(1.0 / 4.0))

    def _snp_transform(self, x):
        """Optimized power transformation using fused operations"""
        return x.sign() * torch.pow(x.abs(), self._power_inv)

    def forward(self, snp_vert):
        """Encoder of snp for encoding vertical frequency responses"""
        # For self-supervised learning, we may have shape (B, F, P1, P2) without tx/rx dimension
        if snp_vert.dim() == 4:
            # No tx/rx dimension - add it for consistency
            b, f, p1, p2 = snp_vert.size()
            snp_vert = snp_vert.unsqueeze(1)  # (B, 1, F, P1, P2)
            d = 1
        else:
            b, d, f, p1, p2 = snp_vert.size()
            
            # Input validation moved to top for early exit
            if self.use_tx_rx_tokens and d != 2:
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
        hidden_states_snp = rearrange(hidden_states_snp, "(b d p) e -> b d p e", b=b, d=d, p=half_p)

        # Add tx and rx tokens if enabled
        if self.use_tx_rx_tokens and d == 2:
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
        use_mixed_precision=True,
        use_tx_rx_tokens=True
    ):
        super().__init__()

        self.freq_length = freq_length
        self.model_dim = model_dim
        self.use_checkpointing = use_checkpointing
        self.use_mixed_precision = use_mixed_precision
        self.use_tx_rx_tokens = use_tx_rx_tokens

        # Optimized projection with proper initialization
        self.snp_proj = nn.Linear(freq_length * 2, freq_length)
        nn.init.xavier_uniform_(self.snp_proj.weight)
        nn.init.zeros_(self.snp_proj.bias)
        
        # Use appropriate number of heads based on freq_length
        # Ensure num_heads divides freq_length evenly
        max_heads = min(8, freq_length // 16)  # Max heads with at least 16 dims per head
        num_heads = max_heads
        while num_heads > 1 and freq_length % num_heads != 0:
            num_heads -= 1
        self.snp_encoder = ConditionEmbedding(encoder_dim=freq_length, embed_dim=model_dim, num_heads=num_heads)

        # Learnable tokens with better initialization (only if using tx/rx tokens)
        if self.use_tx_rx_tokens:
            self.tx_token = nn.Parameter(torch.randn(1, 1, model_dim) * 0.02)
            self.rx_token = nn.Parameter(torch.randn(1, 1, model_dim) * 0.02)
        
        # Pre-computed constants
        self.register_buffer('_power_inv', torch.tensor(0.25))

    def _snp_transform(self, x):
        """Optimized power transformation"""
        with torch.amp.autocast(device_type=x.device.type, enabled=self.use_mixed_precision):
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
        # For self-supervised learning, we may have shape (B, F, P1, P2) without tx/rx dimension
        if snp_vert.dim() == 4:
            # No tx/rx dimension - add it for consistency
            b, f, p1, p2 = snp_vert.size()
            snp_vert = snp_vert.unsqueeze(1)  # (B, 1, F, P1, P2)
            d = 1
        else:
            b, d, f, p1, p2 = snp_vert.size()
            
            if self.use_tx_rx_tokens and d != 2:
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
        with torch.amp.autocast(device_type=snp_vert.device.type, enabled=self.use_mixed_precision):
            hidden_states_snp = self.snp_encoder(snp_vert)
        
        hidden_states_snp = rearrange(hidden_states_snp, "(b d p) e -> b d p e", b=b, d=d, p=half_p)
        
        # Add tx and rx tokens (consistent with SNPEmbedding)
        if self.use_tx_rx_tokens and d == 2:
            hidden_states_snp[:, 0].add_(self.tx_token)
            hidden_states_snp[:, 1].add_(self.rx_token)
        
        return hidden_states_snp

class SNPDecoder(nn.Module):
    """Decodes SNP embeddings back to S-parameter matrices, handling variable frequency lengths."""
    
    def __init__(
        self,
        model_dim,
        freq_length, # This is now the model's internal frequency dimension
        decoder_hidden_ratio=2,
        use_checkpointing=False,
        use_mixed_precision=True
    ):
        super().__init__()
        self.model_dim = model_dim
        self.freq_length = freq_length
        self.use_checkpointing = use_checkpointing
        self.use_mixed_precision = use_mixed_precision
        
        self.embed_decoder = nn.Sequential(
            nn.Linear(model_dim, model_dim * decoder_hidden_ratio),
            nn.GELU(),
            nn.Linear(model_dim * decoder_hidden_ratio, freq_length)
        )
        
        self.snp_proj = nn.Linear(freq_length, freq_length * 2)
        nn.init.xavier_uniform_(self.snp_proj.weight)
        nn.init.zeros_(self.snp_proj.bias)
        
        self.register_buffer('_power', torch.tensor(4.0))
    
    def _inverse_snp_transform(self, x):
        with torch.amp.autocast(device_type=x.device.type, enabled=self.use_mixed_precision):
            return x.sign() * torch.pow(x.abs() + 1e-8, self._power)
    
    def _decode_chunk(self, hidden_states, b, half_p):
        b_p, e = hidden_states.shape[0], hidden_states.shape[-1]
        
        with torch.amp.autocast(device_type=hidden_states.device.type, enabled=self.use_mixed_precision):
            freq_features = self.embed_decoder(hidden_states.view(-1, e))
            snp_complex = self.snp_proj(freq_features)
        
        snp_complex = snp_complex.view(b, half_p, -1)
        return snp_complex
    
    def forward(self, hidden_states_snp, output_freq_length: int):
        """
        Args:
            hidden_states_snp: Encoded SNP embeddings of shape (B, P, E).
            output_freq_length: The target frequency dimension of the output SNP.
        """
        if hidden_states_snp.dim() == 4:
            b, d, half_p, e = hidden_states_snp.size()
            hidden_states = hidden_states_snp.view(b * d, half_p, e)
            batch_size = b * d
        else:
            b, half_p, e = hidden_states_snp.size()
            hidden_states = hidden_states_snp
            batch_size = b
        
        snp_complex = self._decode_chunk(hidden_states, batch_size, half_p)
        
        # Interpolate to the desired output frequency length
        snp_for_interp = rearrange(snp_complex, 'b p (f c) -> (b p) c f', c=2)
        snp_interp = F.interpolate(snp_for_interp, size=output_freq_length, mode='linear', align_corners=False)
        snp_complex_interp = rearrange(snp_interp, '(b p) c f -> b p (f c)', b=batch_size)

        p = half_p * 2
        device = hidden_states.device
        snp_vert = torch.zeros(batch_size, output_freq_length, p, p, dtype=torch.complex64, device=device)
        
        snp_complex_full = torch.view_as_complex(
            rearrange(snp_complex_interp, "b p (f ri) -> b p f ri", f=output_freq_length)
        )
        
        snp_vert[:, :, :half_p, :] = snp_complex_full.permute(0, 2, 1).unsqueeze(3).expand(-1, -1, -1, p)
        snp_vert[:, :, half_p:, :] = snp_complex_full.permute(0, 2, 1).unsqueeze(3).expand(-1, -1, -1, p)
        
        snp_vert_real = torch.view_as_real(snp_vert)
        snp_vert_real = self._inverse_snp_transform(snp_vert_real)
        snp_vert = torch.view_as_complex(snp_vert_real)
        
        if hidden_states_snp.dim() == 4:
            snp_vert = snp_vert.view(b, d, output_freq_length, p, p)
        
        return snp_vert

class ImprovedSNPDecoder(nn.Module):
    """
    Enhanced decoder with better architecture for phase reconstruction.
    Uses separate pathways for magnitude and phase with skip connections.
    """
    
    def __init__(
        self,
        model_dim,
        freq_length,
        decoder_hidden_ratio=4,  # Increased capacity
        num_decoder_layers=3,    # Deeper decoder
        use_skip_connections=True,
        use_separate_phase_mag=True,
        dropout_rate=0.1,
        use_checkpointing=False,
        use_mixed_precision=True
    ):
        super().__init__()
        self.model_dim = model_dim
        self.freq_length = freq_length
        self.use_checkpointing = use_checkpointing
        self.use_mixed_precision = use_mixed_precision
        self.use_skip_connections = use_skip_connections
        self.use_separate_phase_mag = use_separate_phase_mag
        
        hidden_dim = model_dim * decoder_hidden_ratio
        
        # Multi-layer decoder with residual connections
        self.decoder_layers = nn.ModuleList()
        
        # First layer
        self.decoder_layers.append(nn.Sequential(
            nn.Linear(model_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        ))
        
        # Intermediate layers with skip connections
        for _ in range(num_decoder_layers - 2):
            self.decoder_layers.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_rate)
            ))
        
        # Final projection to frequency space
        self.freq_proj = nn.Linear(hidden_dim, freq_length)
        
        if self.use_separate_phase_mag:
            # Separate heads for magnitude and phase
            self.magnitude_head = nn.Sequential(
                nn.Linear(freq_length, freq_length),
                nn.ReLU()  # Magnitude is always positive
            )
            
            self.phase_head = nn.Sequential(
                nn.Linear(freq_length, freq_length),
                nn.Tanh()  # Bounded phase representation
            )
        else:
            # Direct complex projection
            self.snp_proj = nn.Linear(freq_length, freq_length * 2)
            nn.init.xavier_uniform_(self.snp_proj.weight)
            nn.init.zeros_(self.snp_proj.bias)
        
        # Learnable power for inverse transform
        self.power = nn.Parameter(torch.tensor(4.0))
        
    def _inverse_snp_transform(self, x):
        with torch.amp.autocast(device_type=x.device.type, enabled=self.use_mixed_precision):
            return x.sign() * torch.pow(x.abs() + 1e-8, self.power.abs())
    
    def _decode_features(self, hidden_states):
        """Multi-layer decoding with skip connections"""
        b_p, e = hidden_states.shape
        
        x = hidden_states
        skip_connection = None
        
        for i, layer in enumerate(self.decoder_layers):
            x = layer(x)
            
            # Skip connection from input (if enabled)
            if i == 0 and self.use_skip_connections:
                skip_connection = x
            elif i == len(self.decoder_layers) - 1 and skip_connection is not None:
                x = x + skip_connection
        
        # Project to frequency dimension
        freq_features = self.freq_proj(x)
        
        return freq_features
    
    def forward(self, hidden_states_snp, output_freq_length: int):
        """
        Enhanced forward pass with better phase handling.
        """
        if hidden_states_snp.dim() == 4:
            b, d, half_p, e = hidden_states_snp.size()
            hidden_states = hidden_states_snp.view(b * d, half_p, e)
            batch_size = b * d
        else:
            b, half_p, e = hidden_states_snp.size()
            hidden_states = hidden_states_snp
            batch_size = b
        
        # Decode features
        with torch.amp.autocast(device_type=hidden_states.device.type, enabled=self.use_mixed_precision):
            freq_features = self._decode_features(hidden_states.view(-1, e))
        
        if self.use_separate_phase_mag:
            # Separate magnitude and phase prediction
            magnitude = self.magnitude_head(freq_features)
            phase_features = self.phase_head(freq_features)
            
            # Convert phase features to actual phases
            phase = phase_features * torch.pi  # Scale to [-pi, pi]
            
            # Reshape for interpolation
            magnitude = magnitude.view(batch_size, half_p, -1)
            phase = phase.view(batch_size, half_p, -1)
            
            # Interpolate magnitude and phase separately
            mag_for_interp = rearrange(magnitude, 'b p f -> (b p) 1 f')
            phase_for_interp = rearrange(phase, 'b p f -> (b p) 1 f')
            
            mag_interp = F.interpolate(mag_for_interp, size=output_freq_length, mode='linear', align_corners=False)
            phase_interp = F.interpolate(phase_for_interp, size=output_freq_length, mode='linear', align_corners=False)
            
            # Convert back to complex
            mag_interp = rearrange(mag_interp, '(b p) 1 f -> b p f', b=batch_size)
            phase_interp = rearrange(phase_interp, '(b p) 1 f -> b p f', b=batch_size)
            
            # Apply inverse power transform to magnitude
            mag_transformed = torch.pow(mag_interp + 1e-8, self.power.abs())
            
            # Reconstruct complex numbers
            snp_complex_full = mag_transformed * torch.exp(1j * phase_interp)
            
        else:
            # Original approach with direct complex projection
            snp_complex = self.snp_proj(freq_features)
            snp_complex = snp_complex.view(batch_size, half_p, -1)
            
            # Interpolate
            snp_for_interp = rearrange(snp_complex, 'b p (f c) -> (b p) c f', c=2)
            snp_interp = F.interpolate(snp_for_interp, size=output_freq_length, mode='linear', align_corners=False)
            snp_complex_interp = rearrange(snp_interp, '(b p) c f -> b p (f c)', b=batch_size)
            
            snp_complex_full = torch.view_as_complex(
                rearrange(snp_complex_interp, "b p (f ri) -> b p f ri", f=output_freq_length)
            )
            
            # Apply inverse transform
            snp_complex_full = torch.view_as_complex(
                self._inverse_snp_transform(torch.view_as_real(snp_complex_full))
            )
        
        # Reconstruct full S-parameter matrix
        p = half_p * 2
        device = hidden_states.device
        snp_vert = torch.zeros(batch_size, output_freq_length, p, p, dtype=torch.complex64, device=device)
        
        snp_vert[:, :, :half_p, :] = snp_complex_full.permute(0, 2, 1).unsqueeze(3).expand(-1, -1, -1, p)
        snp_vert[:, :, half_p:, :] = snp_complex_full.permute(0, 2, 1).unsqueeze(3).expand(-1, -1, -1, p)
        
        if hidden_states_snp.dim() == 4:
            snp_vert = snp_vert.view(b, d, output_freq_length, p, p)
        
        return snp_vert