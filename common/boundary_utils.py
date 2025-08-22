"""
Boundary processing utilities for inference compatibility.

This module handles boundary parameter processing to ensure compatibility
between inference JSON files and training data configurations.
"""

import json
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Any

from .param_types import SampleResult, to_new_param_name


def save_scaler_with_config_keys(scalers: tuple, config_keys: list, save_path: Path) -> None:
    """
    Save scalers along with training config_keys metadata.
    
    Args:
        scalers: Tuple of (seq_scaler, fix_scaler)
        config_keys: List of parameter keys used during training
        save_path: Path to save the enhanced scaler file
    """
    import torch
    
    scaler_data = {
        'scalers': scalers,
        'config_keys': config_keys,
        'version': '1.0'  # For future compatibility
    }
    torch.save(scaler_data, save_path)


def load_scaler_with_config_keys(load_path: Path) -> Tuple[tuple, list]:
    """
    Load scalers and extract training config_keys metadata.
    
    Args:
        load_path: Path to the scaler file
        
    Returns:
        Tuple of (scalers, config_keys)
        
    Raises:
        ValueError: If config_keys metadata is not found
    """
    import torch
    
    try:
        # Try loading as enhanced scaler with metadata
        scaler_data = torch.load(load_path, weights_only=False)
        
        if isinstance(scaler_data, dict) and 'scalers' in scaler_data and 'config_keys' in scaler_data:
            return scaler_data['scalers'], scaler_data['config_keys']
    except:
        pass
    
    # Fallback: Try loading as legacy scaler (tuple only)
    try:
        scalers = torch.load(load_path, weights_only=False)
        if isinstance(scalers, tuple) and len(scalers) == 2:
            raise ValueError(
                f"Legacy scaler format detected at {load_path}. "
                "This scaler does not contain config_keys metadata. "
                "Please retrain the model with the updated scaler saving logic, "
                "or manually provide config_keys for inference."
            )
    except Exception as e:
        if "config_keys metadata" in str(e):
            raise e
        else:
            raise ValueError(f"Cannot load scaler from {load_path}: {e}")
    
    raise ValueError(f"Unrecognized scaler format at {load_path}")


def process_boundary_for_inference(
    boundary_json_path: str, 
    training_config_keys: List[str]
) -> Tuple[SampleResult, List[float], List[str]]:
    """
    Process boundary JSON for inference, ensuring compatibility with training config.
    
    Args:
        boundary_json_path: Path to boundary JSON file
        training_config_keys: Config keys used during training (from scaler metadata)
        
    Returns:
        Tuple of (boundary_sample_result, boundary_values_ordered, final_config_keys)
        
    Raises:
        ValueError: If boundary parameters don't match training requirements
    """
    # Load boundary JSON
    with open(boundary_json_path, 'r') as f:
        loaded = json.load(f)
    
    # Get base boundary parameters
    boundary = loaded.get('boundary', {})
    if isinstance(boundary, dict):
        boundary = to_new_param_name(boundary)
    
    # Get CTLE parameters
    ctle = loaded.get('CTLE', {})
    
    # Only include CTLE parameters that were in training config_keys
    filtered_ctle = {}
    ctle_keys_in_training = [key for key in ctle.keys() if key in training_config_keys]
    for key in ctle_keys_in_training:
        filtered_ctle[key] = ctle[key]
    
    # Combine boundary + filtered CTLE
    full_boundary = boundary | filtered_ctle
    boundary_result = SampleResult(**full_boundary)
    
    # Check that all training config_keys are available in boundary
    missing_keys = [key for key in training_config_keys if key not in boundary_result.keys()]
    if missing_keys:
        raise ValueError(
            f"Boundary JSON is missing required parameters: {missing_keys}. "
            f"Training config_keys: {training_config_keys}. "
            f"Available in boundary: {list(boundary_result.keys())}. "
            f"Ensure the boundary JSON contains all parameters used during training."
        )
    
    # Extract values in training order
    boundary_values = [boundary_result.get(key, np.nan) for key in training_config_keys]
    
    return boundary_result, boundary_values, training_config_keys


def validate_boundary_dimensions(
    boundary_values: List[float], 
    scaler_tuple: tuple, 
    config_keys: List[str]
) -> None:
    """
    Validate that boundary dimensions match scaler expectations.
    
    Args:
        boundary_values: List of boundary parameter values
        scaler_tuple: Tuple of (seq_scaler, fix_scaler)  
        config_keys: Config keys for better error messages
        
    Raises:
        ValueError: If dimensions don't match
    """
    seq_scaler, fix_scaler = scaler_tuple
    expected_dim = len(boundary_values)
    
    # Get scaler dimension
    if hasattr(fix_scaler, '_min') and fix_scaler._min is not None:
        scaler_dim = len(fix_scaler._min)
    elif hasattr(fix_scaler, 'min_') and fix_scaler.min_ is not None:
        scaler_dim = len(fix_scaler.min_)
    else:
        raise ValueError("Cannot determine scaler dimensions - invalid scaler format")
    
    if expected_dim != scaler_dim:
        raise ValueError(
            f"Boundary parameter dimension mismatch! "
            f"Boundary has {expected_dim} parameters: {config_keys}, "
            f"but scaler expects {scaler_dim} parameters. "
            f"This usually means the boundary JSON doesn't match the training data format."
        )


def get_directions_from_boundary_json(boundary_json_path: str, default_ports: int = None) -> np.ndarray:
    """
    Extract direction information from boundary JSON.
    
    Args:
        boundary_json_path: Path to boundary JSON file
        default_ports: Default number of ports if directions not specified
        
    Returns:
        numpy array of directions
    """
    with open(boundary_json_path, 'r') as f:
        loaded = json.load(f)
    
    if 'directions' in loaded:
        return np.array(loaded['directions'])
    elif default_ports is not None:
        return np.ones(default_ports // 2, dtype=int)
    else:
        raise ValueError(
            "No 'directions' found in boundary JSON and no default_ports provided. "
            "Please specify directions in the JSON or provide default_ports."
        )
