"""
Unified parameter utilities module for EyeDiagramNet.

This module consolidates all parameter-related utilities into a single,
well-organized package with clear separation of concerns.
"""

# Core parameter types
from .types import (
    Parameter,
    DiscreteParameter,
    LinearParameter, 
    LogParameter,
    SampleResult,
    ParameterSet,
    RandomToggledParameterSet,
    CombinedParameterSet,
    DiscreteParameterSet,
    constraint_fp2_ge_fp1
)

# Parameter conversion utilities
from .conversion import (
    to_new_param_name,
    convert_configs_to_boundaries,
    process_boundary_for_inference,
    get_directions_from_boundary_json
)

# Parameter validation utilities  
from .validation import (
    validate_boundary_dimensions,
    validate_boundary_compatibility
)

# Scaler metadata utilities
from .scaler_metadata import (
    save_scaler_with_config_keys,
    load_scaler_with_config_keys
)

__all__ = [
    # Types
    'Parameter',
    'DiscreteParameter',
    'LinearParameter',
    'LogParameter', 
    'SampleResult',
    'ParameterSet',
    'RandomToggledParameterSet',
    'CombinedParameterSet',
    'DiscreteParameterSet',
    'constraint_fp2_ge_fp1',
    
    # Conversion
    'to_new_param_name',
    'convert_configs_to_boundaries',
    'process_boundary_for_inference',
    'get_directions_from_boundary_json',
    
    # Validation
    'validate_boundary_dimensions',
    'validate_boundary_compatibility',
    
    # Scaler metadata
    'save_scaler_with_config_keys',
    'load_scaler_with_config_keys'
]
