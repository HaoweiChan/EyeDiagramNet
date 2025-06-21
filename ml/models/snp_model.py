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
        
        # Cache for power transformation to avoid recomputation
        self.register_buffer('_power_inv', torch.tensor(1.0 / 4.0))

    def _snp_transform(self, x):
        """Optimized power transformation using fused operations"""
        return x.sign() * torch.pow(x.abs(), self._power_inv)

    def forward(self, snp_vert, tx_token=None, rx_token=None):
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

        # Add tx and rx tokens if enabled and provided
        if self.use_tx_rx_tokens and d == 2:
            if tx_token is None or rx_token is None:
                raise ValueError("tx_token and rx_token must be provided when use_tx_rx_tokens is True.")
            hidden_states_snp[:, 0].add_(tx_token)
            hidden_states_snp[:, 1].add_(rx_token)

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

    def forward(self, snp_vert, tx_token=None, rx_token=None):
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
        
        # Add tx and rx tokens if enabled and provided
        if self.use_tx_rx_tokens and d == 2:
            if tx_token is None or rx_token is None:
                raise ValueError("tx_token and rx_token must be provided when use_tx_rx_tokens is True.")
            # Use positional arguments for checkpointing compatibility
            hidden_states_snp[:, 0] = hidden_states_snp[:, 0] + tx_token
            hidden_states_snp[:, 1] = hidden_states_snp[:, 1] + rx_token
        
        return hidden_states_snp

class SNPDecoder(nn.Module):
    """
    Physics-aware decoder that respects trace-port structure and frequency continuity.
    Uses global scale prediction and local waveform modeling.
    """
    
    def __init__(
        self,
        model_dim,
        freq_length,
        num_attention_layers=2,
        num_heads=8,
        mlp_hidden_ratio=4,
        dropout_rate=0.1,
        use_frequency_conv=True,
        conv_kernel_size=5,
        use_checkpointing=False,
        use_mixed_precision=True,
        enforce_reciprocity=True
    ):
        super().__init__()
        self.model_dim = model_dim
        self.freq_length = freq_length
        self.use_checkpointing = use_checkpointing
        self.use_mixed_precision = use_mixed_precision
        self.use_frequency_conv = use_frequency_conv
        self.enforce_reciprocity = enforce_reciprocity
        
        # Global scale prediction from mean pooling
        self.scale_predictor = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(model_dim, 1),
            nn.Softplus()  # Ensure positive scale
        )
        
        # Trace-level self-attention (each trace sees all other traces)
        self.trace_attention = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=model_dim,
                nhead=num_heads,
                dim_feedforward=model_dim * 4,
                dropout=dropout_rate,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )
            for _ in range(num_attention_layers)
        ])
        
        # Trace-to-port expansion (each trace generates its two ports)
        self.trace_to_ports = nn.Sequential(
            nn.Linear(model_dim, model_dim * 2),
            nn.LayerNorm(model_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        # Port-pair interaction modeling (for S-parameter matrix)
        # Input: concatenated port embeddings from two traces
        pairwise_dim = model_dim * 2
        
        # Normalized waveform prediction (magnitude in [0, 1])
        self.magnitude_waveform = nn.Sequential(
            nn.Linear(pairwise_dim, pairwise_dim * mlp_hidden_ratio),
            nn.LayerNorm(pairwise_dim * mlp_hidden_ratio),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(pairwise_dim * mlp_hidden_ratio, freq_length),
            nn.Sigmoid()  # Normalized magnitude in [0, 1]
        )
        
        # Phase prediction with continuity
        self.phase_predictor = nn.Sequential(
            nn.Linear(pairwise_dim, pairwise_dim * mlp_hidden_ratio),
            nn.LayerNorm(pairwise_dim * mlp_hidden_ratio),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(pairwise_dim * mlp_hidden_ratio, freq_length)
        )
        
        # Frequency smoothing via 1D convolution
        if self.use_frequency_conv:
            padding = (conv_kernel_size - 1) // 2
            self.freq_smooth_mag = nn.Conv1d(1, 1, kernel_size=conv_kernel_size, padding=padding, bias=False)
            self.freq_smooth_phase = nn.Conv1d(1, 1, kernel_size=conv_kernel_size, padding=padding, bias=False)
            # Initialize as identity
            nn.init.constant_(self.freq_smooth_mag.weight, 1.0 / conv_kernel_size)
            nn.init.constant_(self.freq_smooth_phase.weight, 1.0 / conv_kernel_size)
        
        # Learnable power for inverse transform
        self.power = nn.Parameter(torch.tensor(4.0))
        
    def _apply_trace_attention(self, trace_embeddings):
        """Apply self-attention between traces"""
        x = trace_embeddings
        for layer in self.trace_attention:
            if self.use_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
            else:
                with torch.amp.autocast(device_type=x.device.type, enabled=self.use_mixed_precision):
                    x = layer(x)
        return x
    
    def _expand_traces_to_ports(self, trace_embeddings):
        """Expand each trace embedding to its two port embeddings"""
        b, num_traces, e = trace_embeddings.shape
        
        # Each trace generates two port embeddings
        port_pairs = self.trace_to_ports(trace_embeddings)  # (B, num_traces, 2*E)
        port_pairs = port_pairs.view(b, num_traces, 2, e)  # (B, num_traces, 2, E)
        
        # Rearrange to get all ports
        # First half: first port of each trace, Second half: second port of each trace
        ports_first = port_pairs[:, :, 0, :]  # (B, num_traces, E)
        ports_second = port_pairs[:, :, 1, :]  # (B, num_traces, E)
        
        return ports_first, ports_second
    
    def _create_sparam_features(self, ports_first, ports_second):
        """Create S-parameter features for all port pairs"""
        b, num_traces, e = ports_first.shape
        
        # For S-parameters, we need all combinations of ports
        # But respecting the trace structure
        all_ports = torch.cat([ports_first, ports_second], dim=1)  # (B, 2*num_traces, E)
        
        # Create pairwise features
        num_ports = 2 * num_traces
        port_i = all_ports.unsqueeze(2).expand(b, num_ports, num_ports, e)
        port_j = all_ports.unsqueeze(1).expand(b, num_ports, num_ports, e)
        
        pairwise_features = torch.cat([port_i, port_j], dim=-1)  # (B, P, P, 2E)
        
        return pairwise_features
    
    def _smooth_frequency(self, features, feature_type='magnitude'):
        """Apply frequency smoothing using 1D convolution"""
        if not self.use_frequency_conv:
            return features
            
        b, *spatial_dims, f = features.shape
        features_flat = features.view(-1, 1, f)  # (B*..., 1, F)
        
        if feature_type == 'magnitude':
            smoothed = self.freq_smooth_mag(features_flat)
        else:  # phase
            smoothed = self.freq_smooth_phase(features_flat)
            
        return smoothed.view(b, *spatial_dims, f)
    
    def forward(self, hidden_states_snp, output_freq_length: int):
        """
        Decode trace embeddings to S-parameter matrix with physics awareness.
        
        Args:
            hidden_states_snp: Trace embeddings of shape (B, num_traces, E) or (B, D, num_traces, E)
            output_freq_length: Target frequency dimension
            
        Returns:
            Reconstructed S-parameters of shape (B, F, P, P) or (B, D, F, P, P)
        """
        # Handle different input dimensions
        if hidden_states_snp.dim() == 4:
            b, d, num_traces, e = hidden_states_snp.size()
            hidden_states = hidden_states_snp.view(b * d, num_traces, e)
            batch_size = b * d
            has_d_dim = True
        else:
            b, num_traces, e = hidden_states_snp.size()
            hidden_states = hidden_states_snp
            batch_size = b
            has_d_dim = False
        
        # Predict global scale from mean trace embedding
        mean_embedding = hidden_states.mean(dim=1)  # (B, E)
        global_scale = self.scale_predictor(mean_embedding)  # (B, 1)
        
        # Apply trace-level attention
        attended_traces = self._apply_trace_attention(hidden_states)
        
        # Expand traces to port pairs
        ports_first, ports_second = self._expand_traces_to_ports(attended_traces)
        
        # Create S-parameter features for all port pairs
        pairwise_features = self._create_sparam_features(ports_first, ports_second)
        
        # Flatten for MLP processing
        b_flat = batch_size
        p = 2 * num_traces
        pairwise_flat = pairwise_features.view(-1, pairwise_features.shape[-1])
        
        # Predict normalized magnitude waveform and phase
        with torch.amp.autocast(device_type=hidden_states.device.type, enabled=self.use_mixed_precision):
            mag_waveform = self.magnitude_waveform(pairwise_flat)  # (B*P*P, F), normalized to [0,1]
            phase_raw = self.phase_predictor(pairwise_flat)  # (B*P*P, F)
        
        # Reshape
        mag_waveform = mag_waveform.view(b_flat, p, p, self.freq_length)
        phase_raw = phase_raw.view(b_flat, p, p, self.freq_length)
        
        # Apply frequency smoothing for continuity
        mag_waveform = self._smooth_frequency(mag_waveform, 'magnitude')
        phase_raw = self._smooth_frequency(phase_raw, 'phase')
        
        # Convert phase to [-pi, pi] using tanh for stability
        phase = torch.tanh(phase_raw) * torch.pi
        
        # Apply global scale to magnitude
        # Scale is in log domain to handle large dynamic range
        log_scale = torch.log(global_scale + 1e-8).view(b_flat, 1, 1, 1)
        magnitude = mag_waveform * torch.exp(log_scale)
        
        # Interpolate to target frequency
        mag_for_interp = rearrange(magnitude, 'b p1 p2 f -> (b p1 p2) 1 f')
        phase_for_interp = rearrange(phase, 'b p1 p2 f -> (b p1 p2) 1 f')
        
        mag_interp = F.interpolate(mag_for_interp, size=output_freq_length, mode='linear', align_corners=False)
        phase_interp = F.interpolate(phase_for_interp, size=output_freq_length, mode='linear', align_corners=False)
        
        # Reshape back
        mag_interp = rearrange(mag_interp, '(b p1 p2) 1 f -> b f p1 p2', b=b_flat, p1=p, p2=p)
        phase_interp = rearrange(phase_interp, '(b p1 p2) 1 f -> b f p1 p2', b=b_flat, p1=p, p2=p)
        
        # Apply inverse power transform
        mag_transformed = torch.pow(mag_interp + 1e-8, self.power.abs())
        
        # Reconstruct complex S-parameters
        snp_vert = mag_transformed * torch.exp(1j * phase_interp)
        
        # Enforce reciprocity (S_ij = S_ji) if enabled
        if self.enforce_reciprocity:
            snp_vert = (snp_vert + snp_vert.transpose(-2, -1).conj()) / 2
        
        # Reshape back if needed
        if has_d_dim:
            snp_vert = snp_vert.view(b, d, output_freq_length, p, p)
        
        return snp_vert