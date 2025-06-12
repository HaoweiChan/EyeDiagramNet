import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Union
from lightning.pytorch.utilities.rank_zero import rank_zero_info

class CSVProcessor:
    def __init__(self, patterns: List[str] = None, padding_value: int = -1):
        self.patterns = patterns or [
            "input_for_AI*.csv",
            "*AI_input_data*.csv",
            "*AI_input*.csv",
            "*ai*.csv",
        ]
        self.padding = padding_value

    def locate(self, data_dirs: Union[Dict[str, str], List[str]]) -> Union[Dict[str, Path], List[Path]]:
        is_dict = isinstance(data_dirs, dict)
        out = {} if is_dict else []
        items = data_dirs.items() if is_dict else enumerate(data_dirs)

        for key, dir_path in items:
            rank_zero_info(f"Parsing data from {dir_path}")
            matches = [
                p
                for pat in self.patterns
                for p in Path(dir_path).glob(pat)
            ]
            if len(matches) != 1:
                raise FileNotFoundError(
                    f"Expected exactly one match in {dir_path}, got {matches}"
                )
            if is_dict:
                out[key] = matches[0]
            else:
                out.append(matches[0])
        return out

    def parse(self, csv_path: Union[Path, str]) -> Tuple[np.ndarray, np.ndarray]:
        df = pd.read_csv(csv_path, index_col=0, header=0)
        types = ("S", "G", "D")
        type_cols = df.columns[df.columns.str.contains("Type_")]
        df[type_cols] = df[type_cols].replace({t: i for i, t in enumerate(types)})

        mats = [self._spatial_feats(row.dropna()) for _, row in df.iterrows()]
        max_len = max(m.shape[0] for m in mats)
        padded = np.stack(
            [
                np.pad(
                    m,
                    ((0, max_len - m.shape[0]), (0, 0)),
                    constant_values=self.padding
                )
                for m in mats
            ]
        )
        return df.index.values, padded

    def _spatial_feats(self, case: pd.Series) -> np.ndarray:
        idx = case.index
        layer_mask = idx.str.contains("Layer_")
        width_mask = idx.str.contains("W_")
        height_mask = idx.str.contains("H_")

        layers = case[layer_mask].astype(int).values
        widths = case[width_mask].values
        heights = case[height_mask].values

        # per-trace fields
        layer_change = np.r_[True, np.diff(layers) != 0]
        _, layer_count = np.unique(layers, return_counts=True)

        # x coordinate: cumulative widths (shifted by 1)
        cum_x = np.r_[0, widths.cumsum()[:-1]]
        x_dim = cum_x - np.repeat(cum_x[layer_change], layer_count)

        # z coordinate: bottom height of each layer
        cum_h = heights[layer_change].cumsum()
        cum_h = np.roll(cum_h, 1)
        cum_h[0] = 0
        z_dim = np.repeat(cum_h, layer_count)

        # original feature block
        layer_idx = np.flatnonzero(layer_mask)
        feat_dim = layer_idx[1] - layer_idx[0]
        data_col = case.values.reshape(-1, feat_dim)
        
        return np.hstack([data_col, x_dim[:, None], z_dim[:, None]])

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