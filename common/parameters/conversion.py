"""
Parameter conversion utilities for EyeDiagramNet.

This module handles parameter name conversion, boundary processing,
and format transformations for compatibility between different parts
of the system.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any

from .types import SampleResult


def to_new_param_name(d: dict) -> dict:
    """
    Convert old parameter names to new ones for backward compatibility.
    
    Args:
        d: Dictionary with potentially old parameter names
        
    Returns:
        Dictionary with updated parameter names
        
    Example:
        >>> to_new_param_name({'R_tx': 50, 'R_rx': 1000})
        {'R_drv': 50, 'R_odt': 1000}
    """
    key_map = {
        'R_tx': 'R_drv', 'C_tx': 'C_drv', 'L_tx': 'L_drv',
        'R_rx': 'R_odt', 'C_rx': 'C_odt', 'L_rx': 'L_odt',
    }
    for old_key, new_key in key_map.items():
        if old_key in d:
            d[new_key] = d.pop(old_key)
    return d


def convert_configs_to_boundaries(configs_list: list, config_keys: list) -> np.ndarray:
    """
    Convert a list of config dictionaries directly to a pure numpy array of boundaries.
    
    Args:
        configs_list: List of lists of config dictionaries
        config_keys: List of parameter keys for array conversion
        
    Returns:
        numpy.ndarray: Pure numerical array of shape (n_samples, n_configs_per_sample, n_parameters)
    """
    boundaries_list = []
    for configs in configs_list:
        # Convert each list of config dicts to numerical arrays
        sample_boundaries = []
        for config_dict in configs:
            # Extract values in the order specified by config_keys
            values = [config_dict.get(key, np.nan) for key in config_keys]
            sample_boundaries.append(values)
            
        boundaries_list.append(sample_boundaries)
    
    # Convert to pure numpy array with proper shape
    return np.array(boundaries_list, dtype=np.float64)


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


def get_directions_from_boundary_json(boundary_json_path: str, default_ports: int = None) -> np.ndarray:
    """
    Extract direction information from boundary JSON.
    
    Args:
        boundary_json_path: Path to boundary JSON file
        default_ports: Default number of ports if directions not specified
        
    Returns:
        numpy array of directions
        
    Raises:
        ValueError: If directions not found and no default provided
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


def extract_boundary_parameters(
    boundary_dict: Dict[str, Any], 
    config_keys: List[str], 
    include_ctle: bool = True
) -> Tuple[Dict[str, Any], List[float]]:
    """
    Extract and order boundary parameters according to config_keys.
    
    Args:
        boundary_dict: Dictionary containing boundary parameters
        config_keys: Ordered list of parameter keys to extract
        include_ctle: Whether to include CTLE parameters if present
        
    Returns:
        Tuple of (processed_boundary_dict, ordered_values)
    """
    # Apply parameter name conversion
    processed_boundary = to_new_param_name(boundary_dict.copy())
    
    # Add CTLE defaults if requested
    if include_ctle:
        ctle_defaults = {
            "AC_gain": np.nan, 
            "DC_gain": np.nan, 
            "fp1": np.nan, 
            "fp2": np.nan
        }
        for key, default_val in ctle_defaults.items():
            if key in config_keys and key not in processed_boundary:
                processed_boundary[key] = default_val
    
    # Extract values in specified order
    ordered_values = [processed_boundary.get(key, np.nan) for key in config_keys]
    
    return processed_boundary, ordered_values


def validate_parameter_compatibility(
    boundary_params: Dict[str, Any], 
    training_config_keys: List[str]
) -> Tuple[bool, List[str], List[str]]:
    """
    Validate that boundary parameters are compatible with training configuration.
    
    Args:
        boundary_params: Dictionary of boundary parameters
        training_config_keys: List of parameter keys used during training
        
    Returns:
        Tuple of (is_compatible, missing_keys, extra_keys)
    """
    boundary_keys = set(boundary_params.keys())
    training_keys = set(training_config_keys)
    
    missing_keys = list(training_keys - boundary_keys)
    extra_keys = list(boundary_keys - training_keys)
    
    is_compatible = len(missing_keys) == 0
    
    return is_compatible, missing_keys, extra_keys

