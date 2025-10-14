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
    bounds: Optional[Tuple[float, float]] = None  # Value range (min, max)
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

    def register_variable(self, variable: Union[Variable, str], **kwargs):
        """
        Register a new variable in the system.
        
        Args:
            variable: Either a Variable object or a variable name string
            **kwargs: If variable is a string, these become Variable attributes
                     (bounds, role, group, scale, etc.)
        """
        # Handle both Variable objects and string names with kwargs
        if isinstance(variable, str):
            # Create Variable from name and kwargs
            name = variable
            bounds = kwargs.get('bounds', None)
            role = kwargs.get('role', None)
            group = kwargs.get('group', None)
            scale = kwargs.get('scale', ScaleType.LINEAR)
            
            # Parse role from string if needed
            if isinstance(role, str):
                role = VariableRole(role.lower())
            elif role is None:
                # Auto-detect role from name
                role, auto_group, auto_scale = self._parse_variable_name(name)
                if group is None:
                    group = auto_group
                if scale == ScaleType.LINEAR:  # Only override if default
                    scale = auto_scale
            
            variable = Variable(
                name=name,
                role=role,
                bounds=bounds,
                group=group,
                scale=scale
            )
        
        self.variables[variable.name] = variable
        
        # Update indices
        if variable.name not in self._name_to_idx:
            self._name_to_idx[variable.name] = len(self._name_to_idx)
        
        # Update role mapping
        if variable.role not in self._role_to_names:
            self._role_to_names[variable.role] = []
        if variable.name not in self._role_to_names[variable.role]:
            self._role_to_names[variable.role].append(variable.name)
        
        # Update group mapping
        if variable.group:
            if variable.group not in self._group_to_names:
                self._group_to_names[variable.group] = []
            if variable.name not in self._group_to_names[variable.group]:
                self._group_to_names[variable.group].append(variable.name)

    def get_variable(self, name: str) -> Variable:
        """Get variable metadata by name, auto-registering if not found."""
        if name not in self.variables:
            # Auto-register the variable based on naming pattern
            return self.auto_register_variable(name)
        return self.variables[name]
    
    def update_variable_bounds(self, name: str, bounds: Tuple[float, float]):
        """
        Update bounds for an existing variable or create new variable with bounds.
        
        Args:
            name: Variable name
            bounds: (min, max) value range
        """
        if name in self.variables:
            # Update existing variable
            self.variables[name].bounds = bounds
        else:
            # Register new variable with bounds
            self.register_variable(name, bounds=bounds)
    
    def get_bounds(self, name: str) -> Optional[Tuple[float, float]]:
        """
        Get bounds for a variable.
        
        Args:
            name: Variable name
            
        Returns:
            (min, max) bounds tuple if available, None otherwise
        """
        if name not in self.variables:
            # Auto-register if needed
            self.get_variable(name)
        
        variable = self.variables[name]
        return variable.bounds
    
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
    
    def infer_circuit_type(self, variable_name: str) -> str:
        """
        Infer circuit element type from variable name.
        
        Args:
            variable_name: Variable name to analyze
            
        Returns:
            Circuit type: "D" (dielectric), "S" (signal), "G" (ground), or "none" (geometric)
        """
        if variable_name.startswith('D_'):
            return "D"  # Dielectric
        elif variable_name.startswith('S_'):
            return "S"  # Signal
        elif variable_name.startswith('G_'):
            return "G"  # Ground
        else:
            return "none"  # Geometric or other variables
    
    def extract_material_properties(self, variables: Dict[str, Union[float, np.ndarray, 'torch.Tensor']], variable_name: str) -> Dict[str, float]:
        """
        Extract all material properties for a given variable.
        
        This method dynamically discovers material properties based on naming patterns.
        Future material properties can be added by extending the property detection logic.
        
        Args:
            variables: Dictionary of all variable values
            variable_name: Name of the variable to get material properties for
            
        Returns:
            Dictionary mapping property names to values (e.g., {'dk': 4.2, 'df': 0.02, 'conductivity': 5.8e7})
        """
        import torch
        
        # Determine the material group from variable name
        if '_' in variable_name:
            parts = variable_name.split('_')
            if len(parts) >= 2:
                material_group = f"{parts[0]}_{parts[1]}"  # e.g., "D_1", "S_1", "G_1"
            else:
                material_group = parts[0]
        else:
            material_group = None
            
        def _extract_value(var_val):
            """Helper to extract scalar value from various types."""
            if isinstance(var_val, torch.Tensor):
                return var_val.mean().item() if var_val.numel() > 1 else var_val.item()
            elif isinstance(var_val, np.ndarray):
                return np.mean(var_val) if var_val.size > 1 else float(var_val)
            else:
                return float(var_val)
        
        material_props = {}
        
        if material_group:
            # Dynamically discover all material properties for this group
            material_prefix = f"{material_group}_"
            
            # Look for all variables that belong to this material group
            for var_name, var_value in variables.items():
                if var_name.startswith(material_prefix):
                    # Extract the property name (e.g., "dk", "df", "cond", etc.)
                    property_name = var_name[len(material_prefix):]
                    material_props[property_name] = _extract_value(var_value)
        
        # Add default values for standard properties if not found
        # This maintains backward compatibility while allowing new properties
        if 'dk' not in material_props:
            material_props['dk'] = 1.0  # Default dielectric constant
        if 'df' not in material_props:
            material_props['df'] = 0.0  # Default dissipation factor
        if 'cond' not in material_props:
            material_props['cond'] = 0.0  # Default conductivity
            
        # Rename 'cond' to 'conductivity' for clarity (maintain API compatibility)
        if 'cond' in material_props:
            material_props['conductivity'] = material_props.pop('cond')
        
        return material_props
    
    def get_circuit_types(self) -> List[str]:
        """Get all possible circuit element types."""
        return ["D", "S", "G", "none"]
    
    def get_instance_index(self, var_name: str) -> int:
        """
        Get consistent instance index for variable within its role group.
        
        Variables with the same role are ordered alphabetically to ensure
        consistent instance indices across different datasets/naming schemes.
        
        Args:
            var_name: Variable name to get instance index for
            
        Returns:
            Instance index (0-based) within the role group
        """
        if var_name not in self.variables:
            return 0  # Default for unknown variables
            
        variable = self.get_variable(var_name)
        
        # Get all variables with the same role
        same_role_vars = [name for name in self.variable_names 
                         if self.get_variable(name).role == variable.role]
        
        # Sort alphabetically for consistent ordering across datasets
        same_role_vars.sort()
        
        # Return position in sorted list
        try:
            return same_role_vars.index(var_name)
        except ValueError:
            return 0  # Fallback to first instance
    
    def get_max_instances_per_role(self) -> int:
        """
        Get maximum number of variables for any single role.
        
        This is used to size the instance embedding table.
        
        Returns:
            Maximum number of variables that share the same role
        """
        from collections import Counter
        
        if not self.variable_names:
            return 1  # Default minimum
            
        # Count variables per role
        role_counts = Counter(
            self.get_variable(name).role for name in self.variable_names
        )
        
        return max(role_counts.values(), default=1)
    
    def get_role_groups(self) -> Dict[str, List[str]]:
        """
        Get variables grouped by their roles.
        
        Returns:
            Dictionary mapping role names to lists of variable names with that role
        """
        role_groups = {}
        
        for var_name in self.variable_names:
            variable = self.get_variable(var_name)
            role_name = variable.role.value
            
            if role_name not in role_groups:
                role_groups[role_name] = []
            role_groups[role_name].append(var_name)
        
        # Sort each group alphabetically
        for role_name in role_groups:
            role_groups[role_name].sort()
            
        return role_groups
    
    def get_material_property_schema(self) -> List[str]:
        """
        Get the canonical ordering of material properties for neural network input.
        
        This defines the expected order of material properties when creating tensors.
        New material properties can be added to the end of this list to maintain 
        backward compatibility.
        
        Returns:
            List of property names in canonical order for tensor creation
        """
        return ['dk', 'df', 'conductivity']  # Standard electromagnetic properties
        
        # Future extensions could include:
        # return ['dk', 'df', 'conductivity', 'thickness', 'roughness', 'thermal_conductivity']
    
    def get_material_properties_as_tensor_values(self, variables: Dict[str, Union[float, np.ndarray, 'torch.Tensor']], variable_name: str) -> List[float]:
        """
        Get material properties as ordered list of values for tensor creation.
        
        Args:
            variables: Dictionary of all variable values
            variable_name: Name of the variable to get material properties for
            
        Returns:
            List of property values in canonical order (matching get_material_property_schema)
        """
        material_props_dict = self.extract_material_properties(variables, variable_name)
        schema = self.get_material_property_schema()
        
        # Extract values in canonical order, using defaults for missing properties
        values = []
        for prop_name in schema:
            values.append(material_props_dict.get(prop_name, 0.0))
        
        return values


# No default global registry instance - users should create their own VariableRegistry
