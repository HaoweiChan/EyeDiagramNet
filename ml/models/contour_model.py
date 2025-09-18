"""
Variable-Agnostic Contour Prediction Model

Main model combining variable token encoding, sequence encoding, and prediction heads
for learning continuous eye width contours over arbitrary variable pairs.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Union

from .contour_encoder import VariableTokenEncoder, DeepSetsEncoder, SetTransformerEncoder
from .layers import positional_encoding_1d
from ..data.variable_registry import VariableRegistry


class SequenceEncoder(nn.Module):
    """Encode sequence structure tokens (D/S/G types, layers, multipliers)."""
    
    def __init__(
        self,
        vocab_size: int = 64,  # Number of unique sequence token types
        embed_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        max_sequence_length: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        
        # Token embeddings for sequence elements
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(embed_dim, max_sequence_length, dropout)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output projection to match variable encoder output
        self.output_proj = nn.Linear(embed_dim, hidden_dim)
        
    def forward(self, sequence_tokens: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            sequence_tokens: [batch_size, seq_len] or [seq_len] integer token IDs
            mask: Optional [batch_size, seq_len] boolean mask
        
        Returns:
            encoded: [batch_size, hidden_dim] or [hidden_dim] 
        """
        # Handle single sequence case
        if sequence_tokens.dim() == 1:
            sequence_tokens = sequence_tokens.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Token embeddings
        embedded = self.token_embedding(sequence_tokens)
        
        # Add positional encoding
        embedded = self.pos_encoding(embedded)
        
        # Convert mask for transformer (True = masked)
        if mask is not None:
            transformer_mask = ~mask
        else:
            transformer_mask = None
        
        # Apply transformer
        encoded = self.transformer(embedded, src_key_padding_mask=transformer_mask)
        
        # Pool sequence representation (mean over non-masked positions)
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).expand_as(encoded)
            masked_encoded = encoded * mask_expanded
            seq_lengths = mask.sum(dim=-1, keepdim=True).clamp(min=1)
            pooled = masked_encoded.sum(dim=1) / seq_lengths
        else:
            pooled = encoded.mean(dim=1)
        
        # Final projection
        output = self.output_proj(pooled)
        
        if squeeze_output:
            output = output.squeeze(0)
            
        return output


class PredictorHead(nn.Module):
    """Prediction head for eye width with optional uncertainty."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
        predict_uncertainty: bool = False
    ):
        super().__init__()
        
        self.predict_uncertainty = predict_uncertainty
        
        # Shared layers
        shared_layers = []
        current_dim = input_dim
        
        for i in range(num_layers - 1):
            shared_layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            current_dim = hidden_dim
        
        self.shared = nn.Sequential(*shared_layers)
        
        # Output heads
        self.eye_width_head = nn.Linear(current_dim, 1)
        
        if predict_uncertainty:
            self.uncertainty_head = nn.Sequential(
                nn.Linear(current_dim, 1),
                nn.Softplus()  # Ensure positive uncertainty
            )
        
        # Initialize final layer with small weights for stability
        nn.init.xavier_normal_(self.eye_width_head.weight, gain=0.1)
        nn.init.constant_(self.eye_width_head.bias, 0.0)
        
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: [batch_size, input_dim] or [input_dim]
            
        Returns:
            eye_width: [batch_size, 1] or [1] predicted eye width
            uncertainty: [batch_size, 1] or [1] prediction uncertainty (if enabled)
        """
        shared_repr = self.shared(x)
        
        eye_width = self.eye_width_head(shared_repr)
        
        if self.predict_uncertainty:
            uncertainty = self.uncertainty_head(shared_repr)
            return eye_width, uncertainty
        else:
            return eye_width


class ContourPredictor(nn.Module):
    """
    Variable-agnostic contour prediction model.
    
    Learns f(variables, sequence) -> eye_width mapping that can be queried
    for arbitrary variable pairs to generate smooth contours.
    """
    
    def __init__(
        self,
        variable_registry: Optional[VariableRegistry] = None,
        # Variable encoder parameters
        token_dim: int = 64,
        variable_encoder_type: str = "deepsets",  # "deepsets" or "set_transformer"
        variable_output_dim: int = 256,
        # Sequence encoder parameters  
        sequence_vocab_size: int = 64,
        sequence_embed_dim: int = 64,
        sequence_hidden_dim: int = 128,
        sequence_num_layers: int = 2,
        max_sequence_length: int = 256,
        # Predictor parameters
        predictor_hidden_dim: int = 256,
        predictor_num_layers: int = 3,
        predict_uncertainty: bool = False,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.registry = variable_registry or VariableRegistry()
        self.predict_uncertainty = predict_uncertainty
        
        # Variable encoder
        self.variable_encoder = VariableTokenEncoder(
            variable_registry=self.registry,
            token_dim=token_dim,
            dropout=dropout
        )
        
        # Set aggregation encoder
        if variable_encoder_type == "deepsets":
            self.set_encoder = DeepSetsEncoder(
                token_dim=token_dim,
                output_dim=variable_output_dim,
                dropout=dropout
            )
        elif variable_encoder_type == "set_transformer":
            self.set_encoder = SetTransformerEncoder(
                token_dim=token_dim,
                output_dim=variable_output_dim,
                dropout=dropout
            )
        else:
            raise ValueError(f"Unknown variable encoder type: {variable_encoder_type}")
        
        # Sequence encoder
        self.sequence_encoder = SequenceEncoder(
            vocab_size=sequence_vocab_size,
            embed_dim=sequence_embed_dim,
            hidden_dim=sequence_hidden_dim,
            num_layers=sequence_num_layers,
            max_sequence_length=max_sequence_length,
            dropout=dropout
        )
        
        # Predictor head
        combined_dim = variable_output_dim + sequence_hidden_dim
        self.predictor = PredictorHead(
            input_dim=combined_dim,
            hidden_dim=predictor_hidden_dim,
            num_layers=predictor_num_layers,
            dropout=dropout,
            predict_uncertainty=predict_uncertainty
        )
    
    def forward(
        self,
        variables: Dict[str, torch.Tensor],
        sequence_tokens: torch.Tensor,
        sequence_mask: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for contour prediction.
        
        Args:
            variables: Dict mapping variable names to values
            sequence_tokens: [batch_size, seq_len] or [seq_len] sequence token IDs
            sequence_mask: Optional [batch_size, seq_len] mask for sequence
        
        Returns:
            eye_width: Predicted eye width values
            uncertainty: Prediction uncertainty (if predict_uncertainty=True)
        """
        # Encode variables (permutation-invariant)
        var_tokens, token_info = self.variable_encoder(variables)
        z_vars = self.set_encoder(var_tokens)
        
        # Encode sequence structure
        z_seq = self.sequence_encoder(sequence_tokens, mask=sequence_mask)
        
        # Handle dimension alignment for batching
        if z_vars.dim() != z_seq.dim():
            if z_vars.dim() == 1 and z_seq.dim() == 2:
                z_vars = z_vars.unsqueeze(0).expand(z_seq.size(0), -1)
            elif z_vars.dim() == 2 and z_seq.dim() == 1:
                z_seq = z_seq.unsqueeze(0).expand(z_vars.size(0), -1)
        
        # Combine variable and sequence representations
        z_combined = torch.cat([z_vars, z_seq], dim=-1)
        
        # Predict eye width (and uncertainty)
        return self.predictor(z_combined)
    
    def predict_contour_2d(
        self,
        var1_name: str,
        var2_name: str,
        fixed_variables: Dict[str, float],
        sequence_tokens: torch.Tensor,
        var1_range: Tuple[float, float],
        var2_range: Tuple[float, float],
        resolution: int = 50,
        device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate 2D contour for any pair of variables.
        
        Args:
            var1_name: Name of first variable (x-axis)
            var2_name: Name of second variable (y-axis) 
            fixed_variables: Values for all other variables
            sequence_tokens: Sequence structure tokens
            var1_range: (min, max) range for var1
            var2_range: (min, max) range for var2
            resolution: Grid resolution
            device: Target device for computation
            
        Returns:
            var1_grid: [resolution] grid values for var1
            var2_grid: [resolution] grid values for var2  
            predictions: [resolution, resolution] predicted eye widths
        """
        self.eval()
        
        if device is None:
            device = next(self.parameters()).device
        
        # Create coordinate grids
        var1_values = torch.linspace(var1_range[0], var1_range[1], resolution, device=device)
        var2_values = torch.linspace(var2_range[0], var2_range[1], resolution, device=device)
        
        predictions = torch.zeros(resolution, resolution, device=device)
        
        with torch.no_grad():
            for i, v1 in enumerate(var1_values):
                for j, v2 in enumerate(var2_values):
                    # Create variable assignment
                    variables = fixed_variables.copy()
                    variables[var1_name] = v1.item()
                    variables[var2_name] = v2.item()
                    
                    # Convert to tensors
                    var_tensors = {k: torch.tensor(v, device=device) for k, v in variables.items()}
                    
                    # Predict
                    if self.predict_uncertainty:
                        eye_width, _ = self.forward(var_tensors, sequence_tokens.to(device))
                    else:
                        eye_width = self.forward(var_tensors, sequence_tokens.to(device))
                    
                    predictions[i, j] = eye_width.squeeze()
        
        return var1_values, var2_values, predictions
    
    def get_variable_importance(
        self,
        variables: Dict[str, torch.Tensor],
        sequence_tokens: torch.Tensor,
        method: str = "gradient"
    ) -> Dict[str, float]:
        """
        Compute variable importance scores using gradients.
        
        Args:
            variables: Variable values to analyze
            sequence_tokens: Sequence structure
            method: Importance computation method
            
        Returns:
            importance_scores: Dict mapping variable names to importance scores
        """
        self.eval()
        
        # Convert variables to tensors with gradients
        var_tensors = {}
        for name, value in variables.items():
            if not isinstance(value, torch.Tensor):
                value = torch.tensor(value, dtype=torch.float32)
            var_tensors[name] = value.requires_grad_(True)
        
        # Forward pass
        if self.predict_uncertainty:
            output, _ = self.forward(var_tensors, sequence_tokens)
        else:
            output = self.forward(var_tensors, sequence_tokens)
        
        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=output.sum(),
            inputs=list(var_tensors.values()),
            create_graph=False,
            retain_graph=False
        )
        
        # Compute importance scores
        importance_scores = {}
        for i, (name, grad) in enumerate(zip(var_tensors.keys(), gradients)):
            if method == "gradient":
                importance_scores[name] = grad.abs().mean().item()
            elif method == "gradient_x_input":
                importance_scores[name] = (grad * var_tensors[name]).abs().mean().item()
            else:
                raise ValueError(f"Unknown importance method: {method}")
        
        return importance_scores


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence tokens using layers.py utilities."""
    
    def __init__(self, embed_dim: int, max_len: int = 256, dropout: float = 0.1):
        super().__init__()
        
        self.dropout = nn.Dropout(dropout)
        
        # Use common positional encoding from layers.py
        pe = positional_encoding_1d(embed_dim, max_len)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, embed_dim] or [seq_len, embed_dim]
        """
        if x.dim() == 3:
            # x: [batch_size, seq_len, embed_dim]
            seq_len = x.size(1)
            x = x + self.pe[:seq_len, :].unsqueeze(0)
        else:
            # x: [seq_len, embed_dim]
            seq_len = x.size(0)
            x = x + self.pe[:seq_len, :]
            
        return self.dropout(x)
