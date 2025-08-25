"""
Parameter validation utilities for EyeDiagramNet.

This module provides validation functions to ensure parameter
compatibility and dimensional consistency across the system.
"""

from typing import List, Tuple
import numpy as np


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


def validate_boundary_compatibility(
    boundary_params: dict, 
    training_config_keys: List[str],
    strict: bool = True
) -> Tuple[bool, List[str], List[str]]:
    """
    Validate that boundary parameters are compatible with training configuration.
    
    Args:
        boundary_params: Dictionary of boundary parameters
        training_config_keys: Config keys used during training
        strict: If True, all training keys must be present
        
    Returns:
        Tuple of (is_compatible, missing_keys, extra_keys)
    """
    boundary_keys = set(boundary_params.keys())
    training_keys = set(training_config_keys)
    
    missing_keys = list(training_keys - boundary_keys)
    extra_keys = list(boundary_keys - training_keys)
    
    if strict:
        is_compatible = len(missing_keys) == 0
    else:
        # In non-strict mode, allow extra keys but not missing keys
        is_compatible = len(missing_keys) == 0
    
    return is_compatible, missing_keys, extra_keys


def validate_parameter_ranges(
    parameter_values: dict,
    parameter_constraints: dict = None
) -> Tuple[bool, List[str]]:
    """
    Validate that parameter values are within acceptable ranges.
    
    Args:
        parameter_values: Dictionary of parameter names and values
        parameter_constraints: Optional constraints dict with min/max values
        
    Returns:
        Tuple of (all_valid, invalid_parameters)
    """
    if parameter_constraints is None:
        # Default constraints for common parameters
        parameter_constraints = {
            'R_drv': {'min': 0, 'max': 1000},
            'R_odt': {'min': 0, 'max': 1e12},
            'C_drv': {'min': 0, 'max': 1e-9},
            'C_odt': {'min': 0, 'max': 1e-9},
            'L_drv': {'min': 0, 'max': 1e-6},
            'L_odt': {'min': 0, 'max': 1e-6},
            'pulse_amplitude': {'min': 0, 'max': 5},
            'bits_per_sec': {'min': 1e6, 'max': 1e12},
            'vmask': {'min': 0, 'max': 1},
            'AC_gain': {'min': 0, 'max': 100},
            'DC_gain': {'min': 0, 'max': 100},
            'fp1': {'min': 1e6, 'max': 1e12},
            'fp2': {'min': 1e6, 'max': 1e12},
        }
    
    invalid_parameters = []
    
    for param_name, value in parameter_values.items():
        if param_name in parameter_constraints:
            constraints = parameter_constraints[param_name]
            min_val = constraints.get('min', -np.inf)
            max_val = constraints.get('max', np.inf)
            
            # Skip validation for NaN values (they're often used as defaults)
            if not np.isnan(value):
                if value < min_val or value > max_val:
                    invalid_parameters.append(
                        f"{param_name}={value} not in range [{min_val}, {max_val}]"
                    )
    
    all_valid = len(invalid_parameters) == 0
    return all_valid, invalid_parameters


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


def validate_config_keys_consistency(config_keys: List[str]) -> Tuple[bool, List[str]]:
    """
    Validate that config_keys are consistent and well-formed.
    
    Args:
        config_keys: List of parameter keys to validate
        
    Returns:
        Tuple of (is_valid, issues)
    """
    issues = []
    
    if not config_keys:
        issues.append("config_keys list is empty")
        return False, issues
    
    # Check for duplicates
    if len(config_keys) != len(set(config_keys)):
        duplicates = [key for key in config_keys if config_keys.count(key) > 1]
        issues.append(f"Duplicate keys found: {list(set(duplicates))}")
    
    # Check for invalid characters in keys
    invalid_keys = []
    for key in config_keys:
        if not isinstance(key, str):
            invalid_keys.append(f"{key} (not a string)")
        elif not key.replace('_', '').replace('.', '').isalnum():
            invalid_keys.append(f"{key} (contains invalid characters)")
    
    if invalid_keys:
        issues.append(f"Invalid key names: {invalid_keys}")
    
    # Check for known parameter patterns
    known_prefixes = ['R_', 'C_', 'L_', 'AC_', 'DC_', 'fp']
    known_suffixes = ['_drv', '_odt', '_tx', '_rx']
    known_standalone = ['pulse_amplitude', 'bits_per_sec', 'vmask']
    
    unrecognized_keys = []
    for key in config_keys:
        is_recognized = (
            any(key.startswith(prefix) for prefix in known_prefixes) or
            any(key.endswith(suffix) for suffix in known_suffixes) or
            key in known_standalone
        )
        if not is_recognized:
            unrecognized_keys.append(key)
    
    if unrecognized_keys:
        issues.append(f"Unrecognized parameter names: {unrecognized_keys}")
    
    is_valid = len(issues) == 0
    return is_valid, issues

