import math
import torch
import torch.nn as nn

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

class AttentionPooling(nn.Module):
    # Copied from https://github.com/deep-floyd/IF/blob/2f91391f27dd3c468bf174be5805b4cc92980c0b/deepfloyd_if/model/nn.py#L54
    def __init__(self, num_heads, embed_dim, dtype=None):
        super().__init__()
        self.dtype = dtype
        self.positional_embedding = nn.Parameter(torch.randn(1, embed_dim) / embed_dim**0.5)
        self.proj = nn.Linear(embed_dim, embed_dim, dtype=self.dtype)
        self.v_proj = nn.Linear(embed_dim, embed_dim, dtype=self.dtype)
        self.q_proj = nn.Linear(embed_dim, embed_dim, dtype=self.dtype)
        self.num_heads = num_heads
        self.dim_per_head = embed_dim // self.num_heads
        
    def forward(self, x):
        bs, length, width = x.size()
        
        def shape(x):
            # (bs, length, width) --> (bs, n heads, length, dim per head)
            x = x.view(bs, length, self.num_heads, self.dim_per_head)
            x = x.permute(0, 2, 1, 3)
            return x
        
        class_token = x.mean(dim=1, keepdim=True) + self.positional_embedding.to(x.dtype)
        x = torch.cat([class_token, x], dim=1)  # (bs, length+1, width)

        q = shape(self.q_proj(class_token))
        k = shape(self.proj(x))
        v = shape(self.v_proj(x))

        scale = 1 / math.sqrt(self.dim_per_head)
        weight = torch.einsum("bhfc,bhsc->bhfs", q * scale, k * scale)  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight, dim=-1).type(weight.dtype)
        
        a = torch.einsum("bhts,bhsc->bhst", weight, v)
        a = a.reshape(bs, -1, 1).transpose(1, 2)
        
        return a[:, 0, :]  # cls token

# Refer to huggingface.diffusers.models.embeddings.TextTimeEmbedding
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