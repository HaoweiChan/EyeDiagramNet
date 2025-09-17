"""
Variable Registry System for Variable-Agnostic Contour Prediction

Provides centralized metadata for all tunable design and material parameters
with semantic roles, grouping, and scaling information.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import torch
from enum import Enum

class VariableRole(Enum):
    """Semantic roles for design variables."""
    HEIGHT = "height"
    WIDTH = "width" 
    LENGTH = "length"
    METAL_COND = "metal_cond"
    DIEL_DK = "diel_dk"
    DIEL_DF = "diel_df"
    DIEL_COND = "diel_cond"

class ScaleType(Enum):
    """Scaling/transformation types for variables."""
    LINEAR = "linear"
    LOG = "log" 
    ZSCORE = "zscore"

@dataclass
class Variable:
    """Metadata for a single design/material parameter."""
    name: str                           # e.g., "W_wg1", "H_ubm", "M_1_cond"
    role: VariableRole                  # Semantic category
    group: Optional[str] = None         # Material group e.g., "D_1", "M_1"
    scale: ScaleType = ScaleType.LINEAR # Transformation type
    
    # Computed scaling parameters (fitted during setup)
    scale_params: Dict[str, float] = field(default_factory=dict)

class VariableRegistry:
    """Centralized registry for all tunable parameters."""
    
    def __init__(self):
        self.variables: Dict[str, Variable] = {}
        self._name_to_idx: Dict[str, int] = {}
        self._role_to_names: Dict[VariableRole, List[str]] = {}
        self._group_to_names: Dict[str, List[str]] = {}

    def _parse_variable_name(self, name: str) -> Tuple[VariableRole, Optional[str], ScaleType]:
        """
        Parse variable name to determine role, group, and scaling.
        
        Naming patterns:
        - H_*, W_*, L_* -> HEIGHT, WIDTH, LENGTH (geometric)
        - M_N_cond -> METAL_COND (metal conductivity, group M_N)
        - D_N_dk, D_N_df, D_N_cond -> DIEL_DK, DIEL_DF, DIEL_COND (dielectric properties, group D_N)
        - S_*, G_* -> Could be signal/ground related parameters
        
        Returns:
            Tuple of (role, group, scale_type)
        """
        name_upper = name.upper()
        
        # Height variables (H_*)
        if name_upper.startswith('H_'):
            return (VariableRole.HEIGHT, None, ScaleType.LOG)
        
        # Width variables (W_*)
        elif name_upper.startswith('W_'):
            return (VariableRole.WIDTH, None, ScaleType.LOG)
        
        # Length variables (L_*)
        elif name_upper.startswith('L_'):
            return (VariableRole.LENGTH, None, ScaleType.LOG)
        
        # Metal conductivity (M_N_cond)
        elif name_upper.startswith('M_') and name_upper.endswith('_COND'):
            # Extract group (e.g., M_1 from M_1_cond)
            parts = name.split('_')
            if len(parts) >= 3:
                group = '_'.join(parts[:2])  # M_1
            else:
                group = 'M_1'  # Default
            
            return (VariableRole.METAL_COND, group, ScaleType.LOG)
        
        # Dielectric constant (D_N_dk)
        elif name_upper.startswith('D_') and name_upper.endswith('_DK'):
            # Extract group (e.g., D_1 from D_1_dk)
            parts = name.split('_')
            if len(parts) >= 3:
                group = '_'.join(parts[:2])  # D_1
            else:
                group = 'D_1'  # Default
            
            return (VariableRole.DIEL_DK, group, ScaleType.LINEAR)
        
        # Dielectric dissipation factor (D_N_df)
        elif name_upper.startswith('D_') and name_upper.endswith('_DF'):
            # Extract group (e.g., D_1 from D_1_df)
            parts = name.split('_')
            if len(parts) >= 3:
                group = '_'.join(parts[:2])  # D_1
            else:
                group = 'D_1'  # Default
            
            return (VariableRole.DIEL_DF, group, ScaleType.LOG)
        
        # Dielectric conductivity (D_N_cond)
        elif name_upper.startswith('D_') and name_upper.endswith('_COND'):
            # Extract group (e.g., D_1 from D_1_cond)
            parts = name.split('_')
            if len(parts) >= 3:
                group = '_'.join(parts[:2])  # D_1
            else:
                group = 'D_1'  # Default
            
            return (VariableRole.DIEL_COND, group, ScaleType.LOG)
        
        # Default fallback for unknown patterns
        else:
            # Try to guess based on first letter
            if name_upper.startswith('S_'):
                # Signal-related - treat as width
                return (VariableRole.WIDTH, None, ScaleType.LOG)
            elif name_upper.startswith('G_'):
                # Ground-related - treat as width
                return (VariableRole.WIDTH, None, ScaleType.LOG)
            else:
                # Complete unknown - default to linear width
                return (VariableRole.WIDTH, None, ScaleType.LINEAR)

    def auto_register_variable(self, name: str) -> Variable:
        """
        Automatically register a variable based on its name pattern.
        
        Args:
            name: Variable name to parse and register
            
        Returns:
            The registered Variable object
        """
        if name in self.variables:
            return self.variables[name]
        
        # Parse the variable name
        role, group, scale_type = self._parse_variable_name(name)
        
        # Create and register the variable
        variable = Variable(
            name=name,
            role=role,
            group=group,
            scale=scale_type
        )
        
        self.register_variable(variable)
        return variable

    def register_variable(self, variable: Variable):
        """Register a new variable in the system."""
        self.variables[variable.name] = variable
        
        # Update indices
        self._name_to_idx[variable.name] = len(self._name_to_idx)
        
        # Update role mapping
        if variable.role not in self._role_to_names:
            self._role_to_names[variable.role] = []
        self._role_to_names[variable.role].append(variable.name)
        
        # Update group mapping
        if variable.group:
            if variable.group not in self._group_to_names:
                self._group_to_names[variable.group] = []
            self._group_to_names[variable.group].append(variable.name)

    def get_variable(self, name: str) -> Variable:
        """Get variable metadata by name, auto-registering if not found."""
        if name not in self.variables:
            # Auto-register the variable based on naming pattern
            return self.auto_register_variable(name)
        return self.variables[name]
    
    def get_variables_by_role(self, role: VariableRole) -> List[str]:
        """Get all variable names with a specific role."""
        return self._role_to_names.get(role, [])
    
    def get_variables_by_group(self, group: str) -> List[str]:
        """Get all variable names in a specific group."""
        return self._group_to_names.get(group, [])

    def scale_value(self, name: str, value: Union[float, np.ndarray, torch.Tensor]) -> Union[float, np.ndarray, torch.Tensor]:
        """Apply scaling transformation to a variable value."""
        variable = self.get_variable(name)
        
        if variable.scale == ScaleType.LINEAR:
            return value
        elif variable.scale == ScaleType.LOG:
            return np.log(np.maximum(value, 1e-10))  # Avoid log(0)
        elif variable.scale == ScaleType.ZSCORE:
            if 'mean' not in variable.scale_params or 'std' not in variable.scale_params:
                raise ValueError(f"Z-score scaling requires fitted parameters for {name}")
            return (value - variable.scale_params['mean']) / variable.scale_params['std']
        else:
            raise ValueError(f"Unknown scale type: {variable.scale}")

    def inverse_scale_value(self, name: str, scaled_value: Union[float, np.ndarray, torch.Tensor]) -> Union[float, np.ndarray, torch.Tensor]:
        """Inverse scaling transformation."""
        variable = self.get_variable(name)
        
        if variable.scale == ScaleType.LINEAR:
            return scaled_value
        elif variable.scale == ScaleType.LOG:
            return np.exp(scaled_value)
        elif variable.scale == ScaleType.ZSCORE:
            if 'mean' not in variable.scale_params or 'std' not in variable.scale_params:
                raise ValueError(f"Z-score scaling requires fitted parameters for {name}")
            return scaled_value * variable.scale_params['std'] + variable.scale_params['mean']
        else:
            raise ValueError(f"Unknown scale type: {variable.scale}")

    def fit_scaling_parameters(self, data: Dict[str, np.ndarray]):
        """Fit scaling parameters (e.g., mean/std for z-score) from data."""
        for name, values in data.items():
            if name in self.variables:
                variable = self.variables[name]
                if variable.scale == ScaleType.ZSCORE:
                    variable.scale_params = {
                        'mean': np.mean(values),
                        'std': np.std(values)
                    }

    def get_default_values(self, variable_names: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Get default values for variables (returns zeros since no bounds stored).
        
        Args:
            variable_names: List of variable names to get defaults for.
                          If None, returns defaults for all currently registered variables.
        
        Returns:
            Dict mapping variable names to default values (0.0).
        """
        defaults = {}
        
        if variable_names is not None:
            # Get defaults for specified variables (auto-register if needed)
            for name in variable_names:
                self.get_variable(name)  # Will auto-register if needed
                defaults[name] = 0.0  # Generic default
        else:
            # Get defaults for all currently registered variables
            for name in self.variables.keys():
                defaults[name] = 0.0  # Generic default
        
        return defaults
    
    def sample_variables(
        self, 
        variable_bounds: Dict[str, Tuple[float, float]],
        n_samples: int,
        method: str = 'uniform'
    ) -> Dict[str, np.ndarray]:
        """
        Sample variable values within provided bounds.
        
        Args:
            variable_bounds: Dict mapping variable names to (min, max) bounds
            n_samples: Number of samples to generate
            method: Sampling method ('uniform' or 'lhs')
            
        Returns:
            Dict mapping variable names to arrays of sampled values
        """
        samples = {}
        
        for name, (low, high) in variable_bounds.items():
            # Ensure variable is registered (will auto-register if needed)
            self.get_variable(name)
            
            if method == 'uniform':
                samples[name] = np.random.uniform(low, high, n_samples)
            elif method == 'lhs':
                # Latin Hypercube Sampling (would need scipy)
                samples[name] = np.random.uniform(low, high, n_samples)  # Fallback to uniform
            else:
                raise ValueError(f"Unknown sampling method: {method}")
        
        return samples

    @property 
    def num_variables(self) -> int:
        """Total number of registered variables."""
        return len(self.variables)
    
    @property
    def variable_names(self) -> List[str]:
        """List of all variable names."""
        return list(self.variables.keys())
    
    def __len__(self) -> int:
        return self.num_variables
    
    def __contains__(self, name: str) -> bool:
        """
        Check if a variable is in the registry or can be auto-registered.
        
        This will return True for any variable name that follows recognized patterns,
        even if not currently registered.
        """
        if name in self.variables:
            return True
        
        # Check if the name follows a recognizable pattern
        try:
            self._parse_variable_name(name)
            return True
        except Exception:
            return False
    
    def __iter__(self):
        return iter(self.variables.items())

    def preview_variables(self, names: List[str]) -> Dict[str, Dict[str, any]]:
        """
        Preview what variables would be created from a list of names without registering them.
        
        Args:
            names: List of variable names to preview
            
        Returns:
            Dict mapping variable names to their parsed properties
        """
        preview = {}
        
        for name in names:
            if name in self.variables:
                # Already registered
                var = self.variables[name]
                preview[name] = {
                    'status': 'registered',
                    'role': var.role.value,
                    'group': var.group,
                    'scale': var.scale.value
                }
            else:
                # Would be auto-registered
                try:
                    role, group, scale_type = self._parse_variable_name(name)
                    
                    preview[name] = {
                        'status': 'would_auto_register',
                        'role': role.value,
                        'group': group,
                        'scale': scale_type.value
                    }
                except Exception as e:
                    preview[name] = {
                        'status': 'unrecognized',
                        'error': str(e)
                    }
        
        return preview
    
    def register_variables_from_list(self, names: List[str]) -> List[Variable]:
        """
        Register multiple variables from a list of names.
        
        Args:
            names: List of variable names to register
            
        Returns:
            List of registered Variable objects
        """
        variables = []
        for name in names:
            var = self.get_variable(name)  # Will auto-register if needed
            variables.append(var)
        return variables


# No default global registry instance - users should create their own VariableRegistry
