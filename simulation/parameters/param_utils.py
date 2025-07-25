"""Parameter utilities for training data collection."""

from typing import List
from simulation.parameters.bound_param import PARAM_SETS_MAP, ParameterSet

def parse_param_types(param_type_str: str) -> List[str]:
    """Parse comma-separated parameter type string into a list of valid types."""
    valid_types = list(PARAM_SETS_MAP.keys())
    param_types = [ptype.strip() for ptype in param_type_str.split(',')]
    
    processed_types = []
    for ptype in param_types:
        if ptype in valid_types:
            processed_types.append(ptype)
            continue
        
        # Be lenient: if user provides "DER", auto-correct to "DER_PARAMS"
        ptype_with_suffix = f"{ptype}_PARAMS"
        if ptype_with_suffix in valid_types:
            print(f"[CONFIG WARNING] Corrected param_type '{ptype}' to '{ptype_with_suffix}'.")
            processed_types.append(ptype_with_suffix)
            continue
            
        # If still not found, raise an error
        raise ValueError(f"Invalid parameter type: {ptype}. Valid types: {valid_types}")
            
    return processed_types

def modify_params_for_inductance(param_set, enable_inductance):
    """Modify parameter set to zero out inductance if disabled"""
    if enable_inductance:
        return param_set
    
    # Create a copy of the parameter set with L_drv and L_odt set to zero
    from simulation.parameters.bound_param import ParameterSet, DiscreteParameter
    
    # Handle different parameter set types
    if hasattr(param_set, '_params'):
        # Regular ParameterSet
        new_params = param_set._params.copy()
    elif hasattr(param_set, 'static_set') and hasattr(param_set, 'toggling_set'):
        # CombinedParameterSet - need to modify both parts
        static_params = param_set.static_set._params.copy()
        toggling_params = param_set.toggling_set._params.copy()
        
        # Modify static params
        if 'L_drv' in static_params:
            static_params['L_drv'] = DiscreteParameter(values=[0.0])
        if 'L_odt' in static_params:
            static_params['L_odt'] = DiscreteParameter(values=[0.0])
            
        # Modify toggling params  
        if 'L_drv' in toggling_params:
            toggling_params['L_drv'] = DiscreteParameter(values=[0.0])
        if 'L_odt' in toggling_params:
            toggling_params['L_odt'] = DiscreteParameter(values=[0.0])
        
        # Recreate the parameter sets
        from simulation.parameters.bound_param import RandomToggledParameterSet, CombinedParameterSet
        new_static_set = ParameterSet(**static_params)
        new_toggling_set = RandomToggledParameterSet(
            toggle_probability=param_set.toggling_set.toggle_probability,
            **toggling_params
        )
        return CombinedParameterSet(new_static_set, new_toggling_set)
    else:
        # Unknown parameter set type, try to get params directly
        try:
            new_params = param_set.params.copy()
        except:
            print(f"Warning: Cannot modify inductance for parameter set type: {type(param_set)}")
            return param_set
    
    if 'L_drv' in new_params:
        new_params['L_drv'] = DiscreteParameter(values=[0.0])
    if 'L_odt' in new_params:
        new_params['L_odt'] = DiscreteParameter(values=[0.0])
    
    return ParameterSet(**new_params) 