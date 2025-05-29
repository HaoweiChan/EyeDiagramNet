import torch
import numpy as np
from typing import Dict, Union, Tuple

class TraceSequenceProcessor:
    """Handles semantic parsing of trace sequence data for 3D cross-section analysis.
    
    Data format: [Layer, Type, W, H, Length, feat1, ..., featN, x_dim, z_dim]
    Where:
    - Layer: Integer layer number (categorical)
    - Type: Structure type S/G/D encoded as 0/1/2 (categorical) 
    - W, H, Length: Geometric dimensions (continuous)
    - feat1...featN: Additional local features (continuous)
    - x_dim, z_dim: Spatial coordinates (continuous, uses positional encoding)
    """
    
    # Field indices
    LAYER_IDX = 0
    TYPE_IDX = 1
    GEOM_START = 2  # W, H, Length
    GEOM_END = 5
    SPATIAL_START = -2  # x_dim, z_dim
    
    @classmethod
    def split_features(cls, seq_input: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Split sequence input into semantic components.
        
        Args:
            seq_input: Tensor of shape (B, L, F) where F is total feature dimension
            
        Returns:
            Dictionary with keys: 'layers', 'types', 'geometry', 'features', 'spatial'
        """
        return {
            'layers': seq_input[:, :, cls.LAYER_IDX],
            'types': seq_input[:, :, cls.TYPE_IDX], 
            'geometry': seq_input[:, :, cls.GEOM_START:cls.GEOM_END],  # W, H, Length
            'features': seq_input[:, :, cls.GEOM_END:cls.SPATIAL_START],  # Additional features
            'spatial': seq_input[:, :, cls.SPATIAL_START:]  # x_dim, z_dim
        }
    
    @classmethod
    def get_scalable_features(cls, seq_input: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """Get features that should be scaled (exclude categorical and spatial).
        
        Args:
            seq_input: Input sequence tensor or array
            
        Returns:
            Features that should be scaled: geometry + additional features
        """
        return seq_input[:, :, cls.GEOM_START:cls.SPATIAL_START]
    
    @classmethod
    def get_scalable_slice(cls) -> slice:
        """Get slice for scalable features."""
        return slice(cls.GEOM_START, cls.SPATIAL_START)
    
    @classmethod
    def get_categorical_features(cls, seq_input: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """Get categorical features (layer and type)."""
        return seq_input[:, :, :cls.GEOM_START]
    
    @classmethod
    def get_spatial_features(cls, seq_input: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """Get spatial coordinate features."""
        return seq_input[:, :, cls.SPATIAL_START:]
    
    @classmethod
    def reconstruct_sequence(cls, layers: torch.Tensor, types: torch.Tensor, 
                           geometry: torch.Tensor, features: torch.Tensor, 
                           spatial: torch.Tensor) -> torch.Tensor:
        """Reconstruct full sequence from semantic components."""
        return torch.cat([
            layers.unsqueeze(-1), 
            types.unsqueeze(-1), 
            geometry, 
            features, 
            spatial
        ], dim=-1)
    
    @classmethod
    def split_for_model(cls, seq_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split sequence for model forward pass (matching current trace_model.py format).
        
        Returns:
            layers, types, feats, spatials - matching current model expectations
        """
        feat_dim = seq_input.size(-1) - 4  # Total features excluding layer, type, x_dim, z_dim
        layers = seq_input[:, :, cls.LAYER_IDX:cls.LAYER_IDX+1]  # Keep dim
        types = seq_input[:, :, cls.TYPE_IDX:cls.TYPE_IDX+1]     # Keep dim  
        feats = seq_input[:, :, cls.GEOM_START:cls.SPATIAL_START]  # All middle features
        spatials = seq_input[:, :, cls.SPATIAL_START:]            # x_dim, z_dim
        
        return layers, types, feats, spatials
    
    @classmethod 
    def get_feature_dims(cls, total_dim: int) -> Dict[str, int]:
        """Get dimensions for each feature type."""
        return {
            'layer': 1,
            'type': 1, 
            'geometry': cls.GEOM_END - cls.GEOM_START,  # 3 (W, H, Length)
            'features': total_dim - 4 - (cls.GEOM_END - cls.GEOM_START),  # Additional features
            'spatial': 2  # x_dim, z_dim
        } 