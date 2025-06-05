"""Parameter utilities for training data collection."""

def parse_param_types(param_type_str):
    """Parse comma-separated parameter types"""
    param_types = [ptype.strip() for ptype in param_type_str.split(',')]
    valid_types = ['DDR_PARAMS', 'HBM2_PARAMS', 'UCIE_PARAMS', 'MIX_PARAMS', 'CTLE_PARAMS']
    
    for ptype in param_types:
        if ptype not in valid_types:
            raise ValueError(f"Invalid parameter type: {ptype}. Valid types: {valid_types}")
    
    return param_types

def modify_params_for_inductance(param_set, enable_inductance):
    """Modify parameter set to zero out inductance if disabled"""
    if enable_inductance:
        return param_set
    
    # Create a copy of the parameter set with L_tx and L_rx set to zero
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
        if 'L_tx' in static_params:
            static_params['L_tx'] = DiscreteParameter(values=[0.0])
        if 'L_rx' in static_params:
            static_params['L_rx'] = DiscreteParameter(values=[0.0])
            
        # Modify toggling params  
        if 'L_tx' in toggling_params:
            toggling_params['L_tx'] = DiscreteParameter(values=[0.0])
        if 'L_rx' in toggling_params:
            toggling_params['L_rx'] = DiscreteParameter(values=[0.0])
        
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
    
    if 'L_tx' in new_params:
        new_params['L_tx'] = DiscreteParameter(values=[0.0])
    if 'L_rx' in new_params:
        new_params['L_rx'] = DiscreteParameter(values=[0.0])
    
    return ParameterSet(**new_params) 