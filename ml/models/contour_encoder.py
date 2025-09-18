"""
Enhanced Variable Token Encoder for Permutation-Invariant Variable Representation

Implements variable-as-token encoding where each design parameter becomes a token
with enhanced semantic embeddings that provide generalization while distinguishing variables.

Key Features:
- **Role Embeddings**: Semantic function (HEIGHT, WIDTH, DIELECTRIC_CONSTANT, CONDUCTIVITY, etc.)
- **Type Embeddings**: Circuit element type (D=Dielectric, S=Signal, G=Ground, none=Geometric)
- **Instance Embeddings**: Distinguish variables with same role (H_a vs H_b, both HEIGHT)
- **Material Projections**: Learned projections from actual material properties (dk, df, conductivity)

This design provides:
- **Generalization**: No dependency on specific variable names (H_a, H_b → both HEIGHT)
- **Variable Distinction**: Instance embeddings distinguish same-role variables
- **Composition Support**: Model can learn H_real = -0.5*H_a + 2*H_b relationships
- **Material Identity**: From actual property values, not arbitrary labels

Benefits:
- Works across different naming schemes (H_a/H_b same as H_first/H_second)
- Supports variable composition learning through set aggregation
- Maintains contour plotting ability (vary specific instances)
- Future-proof to new material properties
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import math

from ..data.variable_registry import VariableRegistry, VariableRole


class VariableTokenEncoder(nn.Module):
    """Convert variables to permutation-invariant token representation."""
    
    def __init__(
        self,
        variable_registry: Optional[VariableRegistry] = None,
        token_dim: int = 64,
        role_embed_dim: int = 16,
        type_embed_dim: int = 16,
        instance_embed_dim: int = 8,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.registry = variable_registry or VariableRegistry()
        self.token_dim = token_dim
        
        # Create embeddings for semantic information
        self.role_embeddings = nn.Embedding(
            len(VariableRole), role_embed_dim
        )
        
        # Instance embeddings to distinguish variables with same role
        max_instances = max(self.registry.get_max_instances_per_role(), 8)  # At least 8 for safety
        self.instance_embeddings = nn.Embedding(
            max_instances, instance_embed_dim
        )
        
        # Type embeddings for circuit element types
        self.circuit_types = self.registry.get_circuit_types()
        self.type_to_idx = {ctype: i for i, ctype in enumerate(self.circuit_types)}
        
        self.type_embeddings = nn.Embedding(
            len(self.circuit_types), type_embed_dim
        )
        
        # Material property projection network
        # Dynamically sized based on registry's material property schema
        material_prop_dim = len(self.registry.get_material_property_schema())
        material_embed_dim = type_embed_dim  # Same size as type embeddings for consistency
        self.material_proj = nn.Sequential(
            nn.Linear(material_prop_dim, material_embed_dim),  # Dynamic size -> embedding
            nn.LayerNorm(material_embed_dim),
            nn.ReLU(),
            nn.Linear(material_embed_dim, material_embed_dim)
        )
        
        # Create mappings
        self.role_to_idx = {role: i for i, role in enumerate(VariableRole)}
        
        # Token fusion MLP  
        # input = value + role_embed + type_embed + instance_embed + material_embed
        input_dim = 1 + role_embed_dim + type_embed_dim + instance_embed_dim + material_embed_dim
        
        layers = []
        current_dim = input_dim
        
        for i in range(num_layers):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            current_dim = hidden_dim
        
        layers.append(nn.Linear(current_dim, token_dim))
        
        self.token_mlp = nn.Sequential(*layers)
        
        # Initialize embeddings
        self._init_embeddings()
    
    def _init_embeddings(self):
        """Initialize embeddings with reasonable scales."""
        nn.init.normal_(self.role_embeddings.weight, std=0.1)
        nn.init.normal_(self.type_embeddings.weight, std=0.1)
        nn.init.normal_(self.instance_embeddings.weight, std=0.1)
        
        # Initialize material projection with Xavier initialization
        for layer in self.material_proj:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def _extract_material_properties_as_tensor(self, variables: Dict[str, torch.Tensor], variable_name: str, device: torch.device) -> torch.Tensor:
        """Extract material properties as tensor using registry method."""
        # Use registry method to get properties in canonical order
        material_prop_values = self.registry.get_material_properties_as_tensor_values(variables, variable_name)
        
        # Convert to tensor
        material_props = torch.tensor(material_prop_values, device=device, dtype=torch.float32)
        
        return material_props
    
    def forward(
        self, 
        variables: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, int]]:
        """
        Encode variables as permutation-invariant tokens.
        
        Args:
            variables: Dict mapping variable names to values
                      Values can be scalars or tensors of shape [batch_size]
        
        Returns:
            tokens: [batch_size, n_variables, token_dim] or [n_variables, token_dim]
            token_info: Mapping from token index to variable name
        """
        batch_size = None
        device = None
        
        # Determine batch size and device from first variable
        for name, value in variables.items():
            if isinstance(value, torch.Tensor):
                if value.dim() == 1:
                    batch_size = value.size(0)
                device = value.device
            break
        
        tokens = []
        token_info = {}
        
        for token_idx, (name, value) in enumerate(variables.items()):
            if name not in self.registry:
                continue  # Skip unknown variables
                
            token_info[token_idx] = name
            variable = self.registry.get_variable(name)
            
            # Convert value to tensor and normalize
            if not isinstance(value, torch.Tensor):
                value = torch.tensor(value, dtype=torch.float32)
            
            if device is not None:
                value = value.to(device)
                
            # Scale the value according to registry
            scaled_value = self._scale_tensor_value(name, value)
            
            # Ensure proper shape
            if batch_size is not None:
                if scaled_value.dim() == 0:
                    scaled_value = scaled_value.unsqueeze(0).expand(batch_size)
                scaled_value = scaled_value.unsqueeze(-1)  # [batch_size, 1]
            else:
                if scaled_value.dim() == 0:
                    scaled_value = scaled_value.unsqueeze(0)  # [1]
            
            # Get embeddings
            role_idx = self.role_to_idx[variable.role]
            
            # Infer circuit type and instance index using registry methods
            circuit_type = self.registry.infer_circuit_type(name)
            type_idx = self.type_to_idx[circuit_type]
            instance_idx = self.registry.get_instance_index(name)
            
            # Create embedding tensors
            if device is None:
                device = scaled_value.device
                
            role_emb = self.role_embeddings(torch.tensor(role_idx, device=device))
            type_emb = self.type_embeddings(torch.tensor(type_idx, device=device))
            instance_emb = self.instance_embeddings(torch.tensor(instance_idx, device=device))
            
            # Get material properties and create material embedding using registry method
            material_props = self._extract_material_properties_as_tensor(variables, name, device)
            material_emb = self.material_proj(material_props)
            
            # Expand embeddings to batch size if needed
            if batch_size is not None:
                role_emb = role_emb.unsqueeze(0).expand(batch_size, -1)
                type_emb = type_emb.unsqueeze(0).expand(batch_size, -1)
                instance_emb = instance_emb.unsqueeze(0).expand(batch_size, -1)
                material_emb = material_emb.unsqueeze(0).expand(batch_size, -1)
            
            # Concatenate value and embeddings (no name embedding)
            token_input = torch.cat([
                scaled_value, role_emb, type_emb, instance_emb, material_emb
            ], dim=-1)
            
            # Pass through MLP to create token
            token = self.token_mlp(token_input)
            tokens.append(token)
        
        if not tokens:
            # Handle empty case
            if batch_size is not None:
                return torch.zeros(batch_size, 0, self.token_dim), {}
            else:
                return torch.zeros(0, self.token_dim), {}
        
        # Stack tokens
        if batch_size is not None:
            tokens = torch.stack(tokens, dim=1)  # [batch_size, n_variables, token_dim]
        else:
            tokens = torch.stack(tokens, dim=0)  # [n_variables, token_dim]
        
        return tokens, token_info
    
    def _scale_tensor_value(self, name: str, value: torch.Tensor) -> torch.Tensor:
        """Apply scaling transformation to tensor values."""
        variable = self.registry.get_variable(name)
        
        if variable.scale.value == "linear":
            return value
        elif variable.scale.value == "log":
            return torch.log(torch.clamp(value, min=1e-10))
        elif variable.scale.value == "zscore":
            if 'mean' not in variable.scale_params or 'std' not in variable.scale_params:
                # Use default scaling if not fitted
                return value
            mean = variable.scale_params['mean']
            std = variable.scale_params['std']
            return (value - mean) / (std + 1e-8)
        else:
            return value


class DeepSetsEncoder(nn.Module):
    """Simple DeepSets encoder for permutation-invariant aggregation."""
    
    def __init__(
        self,
        token_dim: int = 64,
        hidden_dim: int = 128,
        output_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Encoder MLP (applied per token)
        encoder_layers = []
        current_dim = token_dim
        
        for i in range(num_layers):
            encoder_layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            current_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder MLP (applied after aggregation)
        decoder_layers = []
        current_dim = hidden_dim
        
        for i in range(num_layers):
            out_dim = output_dim if i == num_layers - 1 else hidden_dim
            decoder_layers.extend([
                nn.Linear(current_dim, out_dim),
            ])
            if i < num_layers - 1:
                decoder_layers.extend([
                    nn.LayerNorm(out_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ])
            current_dim = out_dim
        
        self.decoder = nn.Sequential(*decoder_layers)
        
    def forward(self, tokens: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            tokens: [batch_size, n_tokens, token_dim] or [n_tokens, token_dim]
            mask: Optional [batch_size, n_tokens] boolean mask
        
        Returns:
            encoded: [batch_size, output_dim] or [output_dim]
        """
        if tokens.numel() == 0:
            # Handle empty token case
            batch_size = tokens.size(0) if tokens.dim() == 3 else 1
            return torch.zeros(batch_size, self.decoder[-1].out_features, device=tokens.device)
        
        # Apply encoder to each token
        encoded_tokens = self.encoder(tokens)  # Same shape as input
        
        # Aggregate (sum for permutation invariance)
        if mask is not None:
            # Apply mask before aggregation
            mask = mask.unsqueeze(-1)  # [batch_size, n_tokens, 1]
            encoded_tokens = encoded_tokens * mask
        
        aggregated = encoded_tokens.sum(dim=-2)  # Sum over token dimension
        
        # Apply decoder
        output = self.decoder(aggregated)
        
        return output


class SetTransformerEncoder(nn.Module):
    """Set Transformer encoder for more expressive set aggregation."""
    
    def __init__(
        self,
        token_dim: int = 64,
        hidden_dim: int = 128,
        output_dim: int = 256,
        num_heads: int = 4,
        num_blocks: int = 2,
        num_seeds: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_seeds = num_seeds
        
        # Learnable seed vectors for pooling
        self.seeds = nn.Parameter(torch.randn(num_seeds, hidden_dim))
        
        # Input projection
        self.input_proj = nn.Linear(token_dim, hidden_dim)
        
        # ISAB blocks (Induced Set Attention Block)
        self.isab_blocks = nn.ModuleList([
            ISABBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_blocks)
        ])
        
        # Pooling by Multihead Attention (PMA)
        self.pma = PMABlock(hidden_dim, num_heads, dropout)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, tokens: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            tokens: [batch_size, n_tokens, token_dim] or [n_tokens, token_dim]
            mask: Optional [batch_size, n_tokens] boolean mask
        
        Returns:
            encoded: [batch_size, output_dim] or [output_dim]
        """
        if tokens.numel() == 0:
            batch_size = tokens.size(0) if tokens.dim() == 3 else 1
            return torch.zeros(batch_size, self.output_proj.out_features, device=tokens.device)
        
        # Ensure 3D input
        if tokens.dim() == 2:
            tokens = tokens.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
            
        batch_size = tokens.size(0)
        
        # Project to hidden dimension
        h = self.input_proj(tokens)  # [batch_size, n_tokens, hidden_dim]
        
        # Apply ISAB blocks
        for isab in self.isab_blocks:
            h = isab(h, mask=mask)
        
        # Pool with learned seeds
        seeds = self.seeds.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, num_seeds, hidden_dim]
        pooled = self.pma(seeds, h, mask=mask)  # [batch_size, num_seeds, hidden_dim]
        
        # Aggregate seeds if multiple
        if self.num_seeds > 1:
            pooled = pooled.mean(dim=1)  # [batch_size, hidden_dim]
        else:
            pooled = pooled.squeeze(1)  # [batch_size, hidden_dim]
        
        # Final projection
        output = self.output_proj(pooled)  # [batch_size, output_dim]
        
        if squeeze_output:
            output = output.squeeze(0)
            
        return output


class ISABBlock(nn.Module):
    """Induced Set Attention Block."""
    
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim) 
        
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention
        attn_mask = None
        if mask is not None:
            # Convert boolean mask to attention mask
            attn_mask = ~mask  # MultiheadAttention uses True for masked positions
        
        attn_out, _ = self.attention(x, x, x, key_padding_mask=attn_mask)
        x = self.norm1(x + attn_out)
        
        # Feedforward
        ff_out = self.feedforward(x)
        x = self.norm2(x + ff_out)
        
        return x


class PMABlock(nn.Module):
    """Pooling by Multihead Attention."""
    
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, seeds: torch.Tensor, x: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            seeds: [batch_size, num_seeds, hidden_dim]
            x: [batch_size, n_tokens, hidden_dim]  
            mask: Optional [batch_size, n_tokens]
        """
        attn_mask = None
        if mask is not None:
            attn_mask = ~mask
            
        attn_out, _ = self.attention(seeds, x, x, key_padding_mask=attn_mask)
        output = self.norm(seeds + attn_out)
        
        return output

