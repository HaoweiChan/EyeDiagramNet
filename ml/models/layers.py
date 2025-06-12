import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# Utility Functions
# =============================================================================

def positional_encoding_1d(dims: int, max_len: int):
    position = torch.arange(max_len).unsqueeze(1)
    
    denom = torch.pow(10000.0, -torch.arange(0, dims, 2).float() / dims)
    denom = denom.view(1, -1)
    
    encoding = torch.zeros(max_len, dims)
    encoding[:, 0::2] = torch.sin(position * denom)
    encoding[:, 1::2] = torch.cos(position * denom)
    
    return encoding

def cont_positional_encoding(positions: torch.Tensor, embed_dim: int):
    batch_size, seq_length = positions.size()
    pos_encoding = torch.zeros(batch_size, seq_length, embed_dim, device=positions.device)
    
    div_term = torch.exp(torch.arange(0, embed_dim, 2, device=positions.device) * -(math.log(10000.0) / embed_dim))
    positions = positions.unsqueeze(-1)
    
    pos_encoding[..., 0::2] = torch.sin(positions * div_term)
    pos_encoding[..., 1::2] = torch.cos(positions * div_term)
    
    return pos_encoding

# =============================================================================
# Basic Building Blocks
# =============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x_fp32 = x.float()
        x_normed = (
            x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + self.eps)
        ).type_as(x)
        return self.scale * x_normed

# =============================================================================
# Positional Embedding Modules
# =============================================================================

class RotaryPositionalEmbeddings(nn.Module):
    """Rotary Positional Embeddings implementation"""
    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute the rotation matrix
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Cache for efficiency
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None

    def _update_cos_sin_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        if seq_len > self._seq_len_cached:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=device, dtype=dtype)
            inv_freq = self.inv_freq.to(device=device, dtype=dtype)
            freqs = torch.outer(t, inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cos_cached = emb.cos()
            self._sin_cached = emb.sin()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply rotary positional embeddings to input tensor.
        
        Args:
            x: Input tensor of shape [batch, seq_len, num_heads, head_dim]
            
        Returns:
            Tensor with rotary embeddings applied
        """
        batch_size, seq_len, num_heads, head_dim = x.shape
        
        self._update_cos_sin_cache(seq_len, x.device, x.dtype)
        
        # Get cos and sin for current sequence length and ensure they're on the right device
        cos = self._cos_cached[:seq_len, :head_dim].to(device=x.device, dtype=x.dtype)
        sin = self._sin_cached[:seq_len, :head_dim].to(device=x.device, dtype=x.dtype)
        
        # Reshape for broadcasting
        cos = cos.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, head_dim]
        sin = sin.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, head_dim]
        
        # Split x into two halves
        x1 = x[..., :head_dim // 2]
        x2 = x[..., head_dim // 2:]
        
        # Apply rotation
        rotated = torch.cat([
            x1 * cos[..., :head_dim // 2] - x2 * sin[..., head_dim // 2:],
            x2 * cos[..., head_dim // 2:] + x1 * sin[..., :head_dim // 2]
        ], dim=-1)
        
        return rotated

# Try to import from torchtune, fallback to our implementation
try:
    from torchtune.modules import RotaryPositionalEmbeddings as TorchtuneRoPE
    RoPE = TorchtuneRoPE
except ImportError:
    RoPE = RotaryPositionalEmbeddings

# =============================================================================
# Transformer Modules
# =============================================================================

class RotaryTransformerLayer(nn.Module):
    """Transformer layer with Rotary Positional Embeddings"""
    def __init__(self, d_model, nhead, dropout, rope: RotaryPositionalEmbeddings):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        # Projections
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # Share a single RoPE instance
        self.rope = rope

        # Layer normalization and FFN
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: (B, L, D)
        B, L, D = x.shape

        # 1) Project to q, k, v
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)  # each (B, L, D)

        # 2) Reshape for heads: (B, L, H, head_dim)
        q = q.view(B, L, self.nhead, self.head_dim)
        k = k.view(B, L, self.nhead, self.head_dim)
        v = v.view(B, L, self.nhead, self.head_dim)

        # 3) Apply RoPE to q & k
        q = self.rope(q)
        k = self.rope(k)

        # 4) Compute attention
        # Bring into (B, H, L, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention
        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, H, L, L)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply to v
        out = attn @ v  # (B, H, L, head_dim)
        out = out.transpose(1, 2).reshape(B, L, D)  # (B, L, D)
        out = self.out_proj(self.dropout(out))

        # 5) Residual + norm + FFN
        x = self.norm1(x + out)
        x = self.norm2(x + self.ffn(x))
        return x

class RotaryTransformerEncoder(nn.Module):
    """Multi-layer transformer encoder with RoPE"""
    def __init__(self, d_model, nhead, num_layers, dropout=0.1, max_seq_len=2048):
        super().__init__()
        
        # Shared RoPE instance
        self.rope = RoPE(dim=d_model // nhead, max_seq_len=max_seq_len)
        
        # Stack of transformer layers
        self.layers = nn.ModuleList([
            RotaryTransformerLayer(d_model, nhead, dropout, self.rope)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# =============================================================================
# Attention Modules
# =============================================================================

class AttentionPooling(nn.Module):
    # Copied from https://github.com/deep-floyd/IF/blob/2f91391f27dd3c468bf174be5805b4cc92980c0b/deepfloyd_if/model/nn.py#L54
    def __init__(self, num_heads, embed_dim, dtype=None):
        super().__init__()
        self.dtype = dtype
        self.positional_embedding = nn.Parameter(torch.randn(1, embed_dim) / embed_dim**0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim, dtype=self.dtype)
        self.q_proj = nn.Linear(embed_dim, embed_dim, dtype=self.dtype)
        self.v_proj = nn.Linear(embed_dim, embed_dim, dtype=self.dtype)
        self.num_heads = num_heads
        self.dim_per_head = embed_dim // self.num_heads
        
    def forward(self, x):
        bs, length, width = x.size()
        
        def shape(x):
            # (bs, length, width) --> (bs, length, n heads, dim per head)
            x = x.view(bs, -1, self.num_heads, self.dim_per_head)
            x = x.transpose(1, 2)
            x = x.reshape(bs * self.num_heads, -1, self.dim_per_head)
            x = x.transpose(1, 2)
            return x
        
        class_token = x.mean(dim=1, keepdim=True) + self.positional_embedding.to(x.dtype)
        x = torch.cat([class_token, x], dim=1)  # (bs, length+1, width)

        q = shape(self.q_proj(class_token))
        k = shape(self.k_proj(x))
        v = shape(self.v_proj(x))

        scale = 1 / math.sqrt(math.sqrt(self.dim_per_head))
        weight = torch.einsum("bct,bcs->bts", q * scale, k * scale)  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        
        a = torch.einsum("bts,bcs->bct", weight, v)
        a = a.reshape(bs, -1, 1).transpose(1, 2)
        
        return a[:, 0, :]  # cls token

class ConditionEmbedding(nn.Module):
    def __init__(self, encoder_dim: int, embed_dim: int, num_heads: int = 64):
        super().__init__()
        self.norm1 = nn.LayerNorm(encoder_dim)
        self.pool = AttentionPooling(num_heads, encoder_dim)
        self.proj = nn.Linear(encoder_dim, embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, hidden_states):
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.pool(hidden_states)
        hidden_states = self.proj(hidden_states)
        hidden_states = self.norm2(hidden_states)
        return hidden_states

# =============================================================================
# Uncertainty Quantification Modules
# =============================================================================

class DeepEnsemble(nn.Module):
    """Deep Ensemble wrapper for uncertainty quantification"""
    def __init__(self, model_class, model_args, n_models=5):
        super().__init__()
        self.n_models = n_models
        self.models = nn.ModuleList([
            model_class(**model_args) for _ in range(n_models)
        ])
    
    def forward(self, *args, **kwargs):
        # During training, use only one model (randomly selected)
        if self.training:
            idx = torch.randint(0, self.n_models, (1,)).item()
            return self.models[idx](*args, **kwargs)
        else:
            # During inference, use all models
            outputs = []
            for model in self.models:
                outputs.append(model(*args, **kwargs))
            return outputs
    
    def predict_with_uncertainty(self, *args, **kwargs):
        """Ensemble prediction with uncertainty estimation"""
        self.eval()
        
        predictions = []
        log_vars = []
        logits_list = []
        
        with torch.no_grad():
            for model in self.models:
                values, log_var, logits = model(*args, **kwargs)
                predictions.append(values)
                log_vars.append(log_var)
                logits_list.append(logits)
        
        # Stack predictions
        predictions = torch.stack(predictions, dim=0)  # (n_models, B, P)
        log_vars = torch.stack(log_vars, dim=0)  # (n_models, B, P)
        logits_list = torch.stack(logits_list, dim=0)  # (n_models, B, P)
        
        # Compute statistics
        mean_values = predictions.mean(dim=0)  # (B, P)
        epistemic_var = predictions.var(dim=0)  # (B, P) - model uncertainty
        mean_log_var = log_vars.mean(dim=0)  # (B, P)
        aleatoric_var = torch.exp(mean_log_var)  # (B, P) - data uncertainty
        total_var = epistemic_var + aleatoric_var  # Total uncertainty
        mean_logits = logits_list.mean(dim=0)  # (B, P)
        
        return mean_values, total_var, aleatoric_var, epistemic_var, mean_logits

# =============================================================================
# Loss Balancing Modules
# =============================================================================

class UncertaintyWeightedLoss(nn.Module):
    """Learnable uncertainty weighting for multi-task learning"""
    def __init__(self, task_names):
        super().__init__()
        self.task_names = task_names
        self.log_vars = nn.Parameter(torch.zeros(len(task_names)))

    def forward(self, losses, hidden_states=None):
        total_loss = 0
        for i, (task_name, loss) in enumerate(losses.items()):
            precision = torch.exp(-self.log_vars[i])
            total_loss += precision * loss + self.log_vars[i]
        return total_loss

class LearnableLossWeighting(nn.Module):
    """Simple learnable loss weighting"""
    def __init__(self, task_names):
        super().__init__()
        self.task_names = task_names
        self.weights = nn.Parameter(torch.ones(len(task_names)))

    def forward(self, losses, hidden_states=None):
        total_loss = 0
        weights = F.softmax(self.weights, dim=0) * len(self.weights)
        for i, (task_name, loss) in enumerate(losses.items()):
            total_loss += weights[i] * loss
        return total_loss

class GradNormLossBalancer(nn.Module):
    """Gradient normalization for loss balancing"""
    def __init__(self, task_names, alpha=1.5):
        super().__init__()
        self.task_names = task_names
        self.alpha = alpha
        self.log_weights = nn.Parameter(torch.zeros(len(task_names)))
        self.initial_losses = None

    def forward(self, losses, hidden_states=None):
        # Convert losses dict to match expected format
        loss_dict = losses
        shared_rep = hidden_states
        
        weights = torch.softmax(self.log_weights, dim=0)
        total_loss = sum(weights[i] * loss_dict[name] for i, name in enumerate(self.task_names))

        # If no shared representation or in no_grad context, skip GradNorm logic
        if (shared_rep is None or 
            not shared_rep.requires_grad or 
            any(not loss.requires_grad for loss in loss_dict.values())):
            return total_loss

        # Compute gradient norms
        grads = []
        for i, name in enumerate(self.task_names):
            g = torch.autograd.grad(
                loss_dict[name], shared_rep, 
                retain_graph=True, create_graph=True
            )[0]
            grads.append(g.norm())

        # Initialize initial loss values on first forward pass
        if self.initial_losses is None:
            self.initial_losses = [loss_dict[name].detach() for name in self.task_names]

        # Compute target norms based on relative loss progress
        loss_ratios = [
            loss_dict[name].detach() / self.initial_losses[i] 
            for i, name in enumerate(self.task_names)
        ]
        mean_ratio = sum(loss_ratios) / len(loss_ratios)
        target_grads = [
            g.detach() * (r / mean_ratio) ** self.alpha 
            for g, r in zip(grads, loss_ratios)
        ]

        # Compute gradnorm loss
        gradnorm_loss = sum(
            F.l1_loss(grads[i], target_grads[i]) 
            for i in range(len(self.task_names))
        )
        
        # Backward gradnorm loss to update balancing weights
        if gradnorm_loss.requires_grad:
            gradnorm_loss.backward(retain_graph=True)
        
        return total_loss

# =============================================================================
# Specialized Processing Modules
# =============================================================================

class StructuredGatedBoundaryProcessor(nn.Module):
    """
    Structured processor for boundary conditions with separate MLPs for different parameter types.
    
    Boundary condition structure:
    - Electrical (RLC): R_tx, R_rx, C_tx, C_rx, L_tx, L_rx (6 params)
    - Signal: pulse_amplitude, bits_per_sec, vmask (3 params)  
    - CTLE (optional): AC_gain, DC_gain, fp1, fp2 (4 params, may contain NaNs)
    """
    
    def __init__(self, model_dim, electrical_dim=6, signal_dim=3, ctle_dim=4):
        super().__init__()
        
        self.electrical_dim = electrical_dim
        self.signal_dim = signal_dim
        self.ctle_dim = ctle_dim
        
        # Separate MLPs for different parameter types
        self.electrical_mlp = nn.Sequential(
            nn.Linear(electrical_dim, model_dim // 2),
            nn.GELU(),
            nn.Linear(model_dim // 2, model_dim // 2)
        )
        
        self.signal_mlp = nn.Sequential(
            nn.Linear(signal_dim, model_dim // 4),
            nn.GELU(),
            nn.Linear(model_dim // 4, model_dim // 4)
        )
        
        self.ctle_mlp = nn.Sequential(
            nn.Linear(ctle_dim, model_dim // 4),
            nn.GELU(),
            nn.Linear(model_dim // 4, model_dim // 4)
        )
        
        # Final fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim)
        )
        
    def forward(self, boundary_conditions):
        """
        Args:
            boundary_conditions: Tensor of shape (B, 13) 
                                Expected order: [electrical(6), signal(3), ctle(4)]
        
        Returns:
            Processed boundary features of shape (B, model_dim)
        """
        # Split boundary conditions into components
        electrical = boundary_conditions[:, :self.electrical_dim]  # (B, 6)
        signal = boundary_conditions[:, self.electrical_dim:self.electrical_dim + self.signal_dim]  # (B, 3) 
        ctle = boundary_conditions[:, -self.ctle_dim:]  # (B, 4)
        
        # Simple NaN handling: replace with zeros (CTLE MLP learns that zeros = no CTLE)
        ctle = torch.nan_to_num(ctle, nan=0.0)
        
        # Process each component through its dedicated MLP
        electrical_feat = self.electrical_mlp(electrical)  # (B, model_dim // 2)
        signal_feat = self.signal_mlp(signal)  # (B, model_dim // 4)
        ctle_feat = self.ctle_mlp(ctle)  # (B, model_dim // 4)
        
        # Concatenate all features
        combined_feat = torch.cat([
            electrical_feat, 
            signal_feat, 
            ctle_feat
        ], dim=-1)  # (B, model_dim)
        
        # Final fusion
        output = self.fusion(combined_feat)  # (B, model_dim)
        
        return output