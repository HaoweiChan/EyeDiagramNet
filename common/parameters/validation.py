"""
Parameter validation utilities for EyeDiagramNet.

This module provides validation functions to ensure parameter
compatibility and dimensional consistency across the system.
"""

import logging
import numpy as np
from typing import List, Tuple


def validate_boundary_dimensions(
    boundary_values: np.ndarray, 
    scaler_tuple: tuple, 
    config_keys: List[str]
) -> None:
    """
    Validate that boundary dimensions match scaler expectations and values are within range.
    
    Args:
        boundary_values: Numpy array of boundary parameter values
        scaler_tuple: Tuple of (seq_scaler, fix_scaler)  
        config_keys: Config keys for better error messages
        
    Raises:
        ValueError: If dimensions don't match or scaler is invalid
    """
    seq_scaler, fix_scaler = scaler_tuple
    
    # Ensure boundary_values is a numpy array for consistent processing
    if isinstance(boundary_values, list):
        boundary_values = np.array(boundary_values)
        
    expected_dim = len(boundary_values)
    
    # Get scaler dimension and ranges
    if hasattr(fix_scaler, 'min_') and fix_scaler.min_ is not None:
        min_vals = fix_scaler.min_.numpy()
        max_vals = fix_scaler.max_.numpy()
        scaler_dim = len(fix_scaler.min_)
    elif hasattr(fix_scaler, '_min') and fix_scaler._min is not None:
        min_vals = fix_scaler._min.numpy()
        max_vals = fix_scaler._max.numpy()
        scaler_dim = len(fix_scaler._min)
    else:
        raise ValueError("Cannot determine scaler dimensions - invalid scaler format")
    
    if expected_dim != scaler_dim:
        raise ValueError(
            f"Boundary parameter dimension mismatch! "
            f"Boundary has {expected_dim} parameters: {config_keys}, "
            f"but scaler expects {scaler_dim} parameters. "
            f"This usually means the boundary JSON doesn't match the training data format."
        )

    # Check if boundary values are within the training range and log warnings if not
    for i, value in enumerate(boundary_values):
        if not np.isnan(value):
            min_val, max_val = min_vals[i], max_vals[i]
            # Add a small tolerance for floating point comparisons
            if not (min_val - 1e-6 <= value <= max_val + 1e-6):
                logging.warning(
                    f"Parameter '{config_keys[i]}' value {value:.4f} is outside the training range "
                    f"[{min_val:.4f}, {max_val:.4f}]. Inference results may be less reliable."
                )

def validate_scaler_metadata(scaler_data: dict) -> Tuple[bool, List[str]]:
    """
    Validate that scaler metadata has the required structure.
    
    Args:
        scaler_data: Dictionary containing scaler and metadata
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    error_messages = []
    
    # Check required keys
    required_keys = ['scalers', 'config_keys']
    for key in required_keys:
        if key not in scaler_data:
            error_messages.append(f"Missing required key: {key}")
    
    # Validate scalers structure
    if 'scalers' in scaler_data:
        scalers = scaler_data['scalers']
        if not isinstance(scalers, tuple) or len(scalers) != 2:
            error_messages.append("'scalers' must be a tuple of length 2")
    
    # Validate config_keys structure  
    if 'config_keys' in scaler_data:
        config_keys = scaler_data['config_keys']
        if not isinstance(config_keys, list):
            error_messages.append("'config_keys' must be a list")
        elif len(config_keys) == 0:
            error_messages.append("'config_keys' cannot be empty")
    
    # Check version compatibility if present
    if 'version' in scaler_data:
        version = scaler_data['version']
        supported_versions = ['1.0']
        if version not in supported_versions:
            error_messages.append(f"Unsupported scaler version: {version}")
    
    is_valid = len(error_messages) == 0
    return is_valid, error_messages

