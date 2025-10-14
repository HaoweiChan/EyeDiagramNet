import random
import traceback
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
from lightning import LightningDataModule
from lightning.pytorch.utilities.rank_zero import rank_zero_info

from ..utils.scaler import MinMaxScaler
from ..utils.dataloader import get_loader_from_dataset
from .processors import CSVProcessor
from .variable_registry import VariableRegistry
from common.parameters import convert_configs_to_boundaries
from common.pickle_utils import load_pickle_directory


class ContourProcessor:
    """Processor for handling sequence.csv and variation.csv files.
    
    Combines structural sequence information with material properties and dimensions
    to create resolved contour data for ML training.
    """
    
    def __init__(self):
        # Material type mappings
        self.type_mapping = {'D': 0, 'S': 1, 'G': 2}  # Dielectric, Signal, Ground
        
        # Dynamic column groups - will be populated during parsing
        self.height_cols = []
        self.width_cols = []
        self.length_cols = []
        
        # Dynamic variation file column mappings - will be created during sync
        self.var_height_map = {}
        self.var_width_map = {}
        self.var_length_map = {}

    def locate_files(self, data_dir: Union[str, Path]) -> Tuple[Path, Path]:
        """Locate *_sequence_input.csv and *_variations_variable.csv files with matching prefix."""
        data_path = Path(data_dir)
        
        # Find all sequence and variation files
        sequence_files = list(data_path.glob("*_sequence_input.csv"))
        variation_files = list(data_path.glob("*_variations_variable.csv"))
        
        if not sequence_files:
            raise FileNotFoundError(f"No *_sequence_input.csv files found in {data_path}")
        if not variation_files:
            raise FileNotFoundError(f"No *_variations_variable.csv files found in {data_path}")
        
        # Extract prefixes
        seq_prefixes = {f.name.replace("_sequence_input.csv", ""): f for f in sequence_files}
        var_prefixes = {f.name.replace("_variations_variable.csv", ""): f for f in variation_files}
        
        # Find matching prefixes
        common_prefixes = set(seq_prefixes.keys()) & set(var_prefixes.keys())
        
        if not common_prefixes:
            raise FileNotFoundError(
                f"No matching sequence/variation file pairs found in {data_path}. "
                f"Sequence prefixes: {list(seq_prefixes.keys())}, "
                f"Variation prefixes: {list(var_prefixes.keys())}"
            )
        
        # Use the first matching prefix (or you could add logic to select a specific one)
        prefix = sorted(common_prefixes)[0]
        sequence_file = seq_prefixes[prefix]
        variation_file = var_prefixes[prefix]
        
        if len(common_prefixes) > 1:
            rank_zero_info(f"Multiple matching file pairs found. Using prefix: '{prefix}'")
        
        return sequence_file, variation_file

    def parse_sequence(self, sequence_path: Path) -> pd.DataFrame:
        """Parse sequence.csv file and discover geometric feature columns."""
        df = pd.read_csv(sequence_path)
        
        # Store original Type strings and convert to numeric for compatibility
        df['Type_Original'] = df['Type'].copy()  # Keep original strings like 'D_1', 'S_1', 'G_1'
        df['Type'] = df['Type'].str.extract('([DSG])')[0].map(self.type_mapping)
        
        # Dynamically discover geometric feature columns
        self._discover_geometric_features(df.columns)
        
        # Fill NaN values with 0 (for geometric multipliers)
        geometric_cols = self.height_cols + self.width_cols + self.length_cols
        for col in geometric_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0.0)
        
        rank_zero_info(f"Parsed sequence file: {df.shape[0]} segments")
        rank_zero_info(f"Found geometric features:")
        rank_zero_info(f"  Heights ({len(self.height_cols)}): {self.height_cols}")
        rank_zero_info(f"  Widths ({len(self.width_cols)}): {self.width_cols}")
        rank_zero_info(f"  Lengths ({len(self.length_cols)}): {self.length_cols}")
        return df

    def parse_variation(self, variation_path: Path) -> pd.DataFrame:
        """Parse variation.csv file."""
        df = pd.read_csv(variation_path)
        
        # Remove empty rows
        df = df.dropna(how='all')
        
        rank_zero_info(f"Parsed variation file: {df.shape[0]} cases")
        return df

    def _discover_geometric_features(self, columns: pd.Index) -> None:
        """Discover H_, W_, L_ feature columns from sequence.csv."""
        self.height_cols = [col for col in columns if col.startswith('H_')]
        self.width_cols = [col for col in columns if col.startswith('W_')]
        self.length_cols = [col for col in columns if col.startswith('L_')]
        
        # Sort for consistent ordering
        self.height_cols.sort()
        self.width_cols.sort() 
        self.length_cols.sort()

    def _sync_variation_features(self, variation_df: pd.DataFrame) -> None:
        """Sync sequence features with variation file features and create mappings."""
        var_columns = variation_df.columns.tolist()
        
        # Find matching variation columns for each sequence feature type
        var_heights = [col for col in var_columns if col.startswith('H_')]
        var_widths = [col for col in var_columns if col.startswith('W_')]
        var_lengths = [col for col in var_columns if col.startswith('L_')]
        
        # Create mappings - try to match by similarity or order
        self.var_height_map = self._create_feature_mapping(self.height_cols, var_heights, 'H_')
        self.var_width_map = self._create_feature_mapping(self.width_cols, var_widths, 'W_')
        self.var_length_map = self._create_feature_mapping(self.length_cols, var_lengths, 'L_')
        
        # Log only non-trivial mappings
        non_trivial_mappings = {}
        for prefix, mapping in [('Height', self.var_height_map), ('Width', self.var_width_map), ('Length', self.var_length_map)]:
            non_trivial = {k: v for k, v in mapping.items() if k != v}
            if non_trivial:
                non_trivial_mappings[prefix] = non_trivial
        
        if non_trivial_mappings:
            rank_zero_info(f"Feature mappings (non-identical only): {non_trivial_mappings}")

    def _create_feature_mapping(self, seq_features: List[str], var_features: List[str], prefix: str) -> Dict[str, str]:
        """Create mapping between sequence features and variation features."""
        mapping = {}
        
        if not seq_features or not var_features:
            return mapping
            
        # Sort both lists for consistent mapping
        seq_sorted = sorted(seq_features)
        var_sorted = sorted(var_features)
        
        # Strategy 1: Try exact name matching (after prefix)
        for seq_feat in seq_sorted:
            seq_suffix = seq_feat[2:]  # Remove 'H_', 'W_', or 'L_' prefix
            
            # Look for exact matches in variation features
            exact_match = None
            for var_feat in var_sorted:
                var_suffix = var_feat[2:]  # Remove prefix
                if seq_suffix == var_suffix:
                    exact_match = var_feat
                    break
            
            if exact_match:
                mapping[seq_feat] = exact_match
                continue
        
        # Strategy 2: For unmapped features, try partial matching or positional mapping
        unmapped_seq = [f for f in seq_sorted if f not in mapping]
        unmapped_var = [f for f in var_sorted if f not in mapping.values()]
        
        # Simple positional mapping for remaining features
        for i, seq_feat in enumerate(unmapped_seq):
            if i < len(unmapped_var):
                mapping[seq_feat] = unmapped_var[i]
                rank_zero_info(f"  Positional mapping: {seq_feat} -> {unmapped_var[i]}")
        
        # Warn about unmapped features
        unmapped_seq_final = [f for f in seq_sorted if f not in mapping]
        if unmapped_seq_final:
            rank_zero_info(f"  WARNING: Unmapped sequence {prefix} features: {unmapped_seq_final}")
            
        unmapped_var_final = [f for f in var_sorted if f not in mapping.values()]
        if unmapped_var_final:
            rank_zero_info(f"  WARNING: Unused variation {prefix} features: {unmapped_var_final}")
        
        return mapping

    def resolve_case(self, sequence_df: pd.DataFrame, variation_row: pd.Series) -> np.ndarray:
        """Resolve a single case by combining sequence multipliers with variation values."""
        n_segments = len(sequence_df)
        
        # Extract material properties for each dielectric type
        material_props = self._extract_material_properties(variation_row)
        
        # Extract base dimensions
        base_dims = self._extract_base_dimensions(variation_row)
        
        # Create resolved features array
        # Structure: [segment, layer, type, resolved_heights, resolved_widths, resolved_length, material_props]
        resolved_features = []
        
        for _, seg_row in sequence_df.iterrows():
            features = []
            
            # Basic segment info
            features.extend([seg_row['Segment'], seg_row['Layer'], seg_row['Type']])
            
            # Resolve heights (multiply multipliers with base values)
            resolved_heights = []
            for h_col in self.height_cols:
                multiplier = seg_row.get(h_col, 0.0)
                base_key = self.var_height_map.get(h_col)
                base_value = base_dims.get(base_key, 0.0) if base_key else 0.0
                resolved_heights.append(multiplier * base_value)
            features.extend(resolved_heights)
            
            # Resolve widths
            resolved_widths = []
            for w_col in self.width_cols:
                multiplier = seg_row.get(w_col, 0.0)
                base_key = self.var_width_map.get(w_col)
                base_value = base_dims.get(base_key, 0.0) if base_key else 0.0
                resolved_widths.append(multiplier * base_value)
            features.extend(resolved_widths)
            
            # Resolve length
            resolved_length = []
            for l_col in self.length_cols:
                multiplier = seg_row.get(l_col, 0.0)
                base_key = self.var_length_map.get(l_col)
                base_value = base_dims.get(base_key, 0.0) if base_key else 0.0
                resolved_length.append(multiplier * base_value)
            features.extend(resolved_length)
            
            # Add material properties based on segment type
            seg_type = seg_row['Type']
            seg_type_original = seg_row['Type_Original']
            type_material_props = self._get_material_props_for_type(material_props, seg_type, seg_type_original)
            features.extend(type_material_props)
            
            resolved_features.append(features)
        
        return np.array(resolved_features, dtype=np.float32)

    def _extract_material_properties(self, variation_row: pd.Series) -> Dict:
        """Extract material properties from variation row."""
        props = {}
        
        # Metal conductivity
        props['metal_cond'] = variation_row.get('M_1_cond', 0.0)
        
        # Dielectric properties for each type
        for i in range(1, 7):  # D_1 to D_6
            props[f'D_{i}'] = {
                'dk': variation_row.get(f'D_{i}_dk', 0.0),
                'df': variation_row.get(f'D_{i}_df', 0.0), 
                'cond': variation_row.get(f'D_{i}_cond', 0.0)
            }
        
        return props

    def _extract_base_dimensions(self, variation_row: pd.Series) -> Dict:
        """Extract base dimensions from variation row using dynamic mappings."""
        dims = {}
        
        # Heights - use discovered mappings
        for var_height in self.var_height_map.values():
            dims[var_height] = variation_row.get(var_height, 0.0)
            
        # Widths - use discovered mappings
        for var_width in self.var_width_map.values():
            dims[var_width] = variation_row.get(var_width, 0.0)
            
        # Lengths - use discovered mappings
        for var_length in self.var_length_map.values():
            dims[var_length] = variation_row.get(var_length, 0.0)
        
        return dims

    def _get_material_props_for_type(self, material_props: Dict, seg_type: int, seg_type_original: str) -> List[float]:
        """Get material properties for a specific segment type."""
        if seg_type == 1 or seg_type == 2:  # Signal or Ground (metal)
            return [material_props['metal_cond'], 0.0, 0.0]  # [conductivity, dk=0, df=0]
        elif seg_type == 0:  # Dielectric
            # Extract the specific dielectric type (e.g., 'D_1', 'D_2', etc.)
            dielectric_type = seg_type_original  # e.g., 'D_1', 'D_2'
            
            # Use the specific dielectric properties - must exist
            if dielectric_type in material_props:
                d_props = material_props[dielectric_type]
                return [d_props['cond'], d_props['dk'], d_props['df']]
            else:
                raise ValueError(f"Dielectric type '{dielectric_type}' found in sequence.csv but missing "
                               f"material properties in variation.csv. Available dielectric types: "
                               f"{[k for k in material_props.keys() if k.startswith('D_')]}")
        else:
            return [0.0, 0.0, 0.0]

    def process(self, data_dir: Union[str, Path]) -> Tuple[np.ndarray, List[int]]:
        """Process sequence and variation files to create resolved contour data."""
        sequence_file, variation_file = self.locate_files(data_dir)
        
        sequence_df = self.parse_sequence(sequence_file)
        variation_df = self.parse_variation(variation_file)
        
        # Sync features between sequence and variation files
        self._sync_variation_features(variation_df)
        
        resolved_cases = []
        case_ids = []
        
        for _, var_row in variation_df.iterrows():
            case_id = int(var_row['case_id'])
            resolved_case = self.resolve_case(sequence_df, var_row)
            resolved_cases.append(resolved_case)
            case_ids.append(case_id)
        
        # Stack all cases
        resolved_data = np.stack(resolved_cases) if resolved_cases else np.array([])
        
        rank_zero_info(f"Processed {len(resolved_cases)} cases with shape: {resolved_data.shape}")
        return resolved_data, case_ids
    
    def get_feature_info(self) -> Dict[str, int]:
        """Get information about the number of each feature type."""
        return {
            'n_heights': len(self.height_cols),
            'n_widths': len(self.width_cols),
            'n_lengths': len(self.length_cols)
        }

class ContourDataset(Dataset):
    """
    Dataset for variable-as-token contour prediction.
    
    Converts traditional contour data to variable-token format for training
    the variable-agnostic contour predictor.
    """
    
    def __init__(
        self,
        sequence_data: np.ndarray,  # Sequence structure tokens
        variable_data: Dict[str, np.ndarray],  # Variable values per case/boundary
        eye_widths: np.ndarray,  # Target eye widths
        case_ids: List[int],
        variable_registry: Optional[VariableRegistry] = None,
        train: bool = False,
        # Random subspace sampling parameters
        enable_random_subspace: bool = True,
        min_active_variables: int = 2,
        max_active_variables: int = 6,
        perturbation_scale: float = 0.02  # 2% of variable range
    ):
        super().__init__()
        
        self.registry = variable_registry or VariableRegistry()
        self.train = train
        
        # Store sequence data
        self.sequence_data = torch.from_numpy(sequence_data).long()
        
        # Store variable data
        self.variable_data = {}
        for name, values in variable_data.items():
            if name in self.registry:
                self.variable_data[name] = torch.from_numpy(values).float()
        
        # Store targets
        self.eye_widths = torch.from_numpy(eye_widths).float()
        self.case_ids = case_ids
        
        # Determine dimensions
        if len(self.eye_widths.shape) == 2:
            self.n_cases, self.repetition = self.eye_widths.shape
        else:
            self.n_cases = self.eye_widths.shape[0]
            self.repetition = 1
        
        # Random subspace parameters
        self.enable_random_subspace = enable_random_subspace and train
        self.min_active_variables = min_active_variables
        self.max_active_variables = max_active_variables
        self.perturbation_scale = perturbation_scale
        
        rank_zero_info(f"ContourVariableDataset initialized: {self.n_cases} cases, "
                      f"{self.repetition} boundary conditions, "
                      f"{len(self.variable_data)} variables")
        rank_zero_info(f"Variables: {list(self.variable_data.keys())}")
        
    def __len__(self) -> int:
        return self.n_cases * self.repetition
    
    def __getitem__(self, index: int) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor], List[str]]]:
        # Calculate seq_index and bnd_index
        seq_index = index // self.repetition
        bnd_index = index % self.repetition
        
        # Get sequence tokens
        sequence_tokens = self.sequence_data[seq_index]
        
        # Get variable values for this case/boundary
        variables = {}
        for name, values in self.variable_data.items():
            if len(values.shape) == 2:  # [n_cases, repetition]
                variables[name] = values[seq_index, bnd_index]
            else:  # [n_cases] - same value for all boundaries
                variables[name] = values[seq_index]
        
        # Get target eye width
        # eye_widths shape: [samples, boundaries, signal_lines] or [samples, boundaries]
        # After indexing [seq_index, bnd_index], we get [signal_lines] or scalar
        target = self.eye_widths[seq_index, bnd_index]
        
        # If target is multi-dimensional (e.g., [signal_lines]), take MIN
        # For contour prediction, we want the minimum eye width across signal lines
        if isinstance(target, torch.Tensor) and target.dim() > 0:
            target = target.min()
        
        # Apply random subspace perturbations if training
        active_variables = list(variables.keys())
        if self.enable_random_subspace:
            variables, active_variables = self._apply_random_subspace_perturbation(variables)
        
        # Ensure target is a 1D tensor of shape [1]
        if isinstance(target, torch.Tensor):
            if target.dim() == 0:
                # Scalar tensor - unsqueeze to [1]
                target = target.unsqueeze(0)
            elif target.dim() > 1:
                # Multi-dimensional - take mean and unsqueeze
                target = target.mean().unsqueeze(0)
            # else: already 1D, keep as is
        else:
            # Convert to tensor
            target = torch.tensor([target], dtype=torch.float32)
        
        return {
            'variables': variables,
            'sequence_tokens': sequence_tokens,
            'targets': target,
            'case_id': self.case_ids[seq_index],
            'active_variables': active_variables
        }
    
    def _apply_random_subspace_perturbation(
        self, 
        variables: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], List[str]]:
        """Apply random subspace perturbations for variable-agnostic training."""
        variable_names = list(variables.keys())
        
        # Randomly choose number of variables to perturb
        n_active = random.randint(
            min(self.min_active_variables, len(variable_names)),
            min(self.max_active_variables, len(variable_names))
        )
        
        # Randomly select which variables to perturb
        active_variables = random.sample(variable_names, n_active)
        
        # Apply perturbations
        perturbed_variables = {}
        for name, value in variables.items():
            if name in active_variables and name in self.registry:
                # Get variable info
                var_info = self.registry.get_variable(name)
                
                # Check if bounds are available, otherwise compute from data
                if var_info.bounds is not None:
                    var_range = var_info.bounds[1] - var_info.bounds[0]
                    noise_std = self.perturbation_scale * var_range
                    min_bound, max_bound = var_info.bounds
                else:
                    # Compute bounds from current variable data
                    if name in self.variable_data:
                        all_values = self.variable_data[name].flatten()
                        min_bound = float(all_values.min())
                        max_bound = float(all_values.max())
                        var_range = max_bound - min_bound
                        noise_std = self.perturbation_scale * var_range if var_range > 0 else 0.01
                    else:
                        # Fallback: use small perturbation relative to value
                        var_range = float(value.abs().mean()) if value.numel() > 0 else 1.0
                        noise_std = self.perturbation_scale * var_range
                        min_bound, max_bound = None, None
                
                # Add Gaussian noise
                noise = torch.randn_like(value) * noise_std
                perturbed_value = value + noise
                
                # Clamp to bounds if available
                if min_bound is not None and max_bound is not None:
                    perturbed_value = torch.clamp(
                        perturbed_value,
                        min_bound,
                        max_bound
                    )
                
                perturbed_variables[name] = perturbed_value
            else:
                # Keep inactive variables unchanged
                perturbed_variables[name] = value
        
        return perturbed_variables, active_variables
    
    def get_variable_statistics(self) -> Dict[str, Dict[str, float]]:
        """Compute statistics for each variable for analysis."""
        stats = {}
        
        for name, values in self.variable_data.items():
            flat_values = values.flatten()
            stats[name] = {
                'mean': float(flat_values.mean()),
                'std': float(flat_values.std()),
                'min': float(flat_values.min()),
                'max': float(flat_values.max()),
                'count': len(flat_values)
            }
        
        return stats

class ContourDataModule(LightningDataModule):
    """Lightning DataModule for variable-as-token contour prediction."""
    
    def __init__(
        self,
        data_dirs: Union[Dict[str, str], List[str]],
        label_dir: str,
        variable_registry: Optional[VariableRegistry] = None,
        batch_size: int = 32,
        test_size: float = 0.2,
        scaler_path: Optional[str] = None,
        # Random subspace training parameters
        enable_random_subspace: bool = True,
        min_active_variables: int = 2,
        max_active_variables: int = 6,
        perturbation_scale: float = 0.02
    ):
        super().__init__()
        self.data_dirs = data_dirs
        self.label_dir = Path(label_dir)
        self.registry = variable_registry or VariableRegistry()
        self.batch_size = batch_size
        self.test_size = test_size
        self.scaler_path = scaler_path
        
        # Random subspace parameters
        self.enable_random_subspace = enable_random_subspace
        self.min_active_variables = min_active_variables
        self.max_active_variables = max_active_variables
        self.perturbation_scale = perturbation_scale
        
        # Initialize datasets
        self.train_dataset: Dict[str, ContourDataset] = {}
        self.val_dataset: Dict[str, ContourDataset] = {}
        self.test_dataset: Dict[str, ContourDataset] = {}

    def setup(self, stage: Optional[str] = None, nan: int = -1):
        """Setup datasets by converting contour data to variable-token format."""
        # Initialize scalers
        fit_scaler = True
        try:
            if self.scaler_path is None:
                raise FileNotFoundError("No scaler path provided")
            self.seq_scaler, self.fix_scaler = torch.load(self.scaler_path, weights_only=False)
            rank_zero_info(f"Loaded scalers from {self.scaler_path}")
            fit_scaler = False
        except (FileNotFoundError, AttributeError, EOFError) as e:
            if stage in ["test", "predict"] or stage is None:
                error_msg = f"Cannot find or load scaler file for {stage or 'test/predict'} mode"
                if self.scaler_path:
                    error_msg += f" at path: {self.scaler_path}"
                else:
                    error_msg += " (no scaler_path provided)"
                error_msg += f". Original error: {e}"
                rank_zero_info(f"ERROR: {error_msg}")
                raise FileNotFoundError(error_msg)
            
            self.seq_scaler = MinMaxScaler(nan=nan)
            self.fix_scaler = MinMaxScaler(nan=nan)
            rank_zero_info("Could not find scalers on disk, creating new ones for training.")

        # Process contour data from sequence.csv and variation.csv files
        if isinstance(self.data_dirs, list):
            data_dirs_dict = {f"contour_{i}": dir_path for i, dir_path in enumerate(self.data_dirs)}
        else:
            data_dirs_dict = self.data_dirs

        for name, data_dir in data_dirs_dict.items():
            try:
                # Process contour data
                processor = ContourProcessor()
                contour_data, case_ids = processor.process(data_dir)
                
                # Load eye width labels
                labels = load_pickle_directory(self.label_dir, name)
                
                # Filter data to only include cases with labels
                label_keys = set(labels.keys())
                keep_idx = [i for i, cid in enumerate(case_ids) if cid in label_keys]
                
                if not keep_idx:
                    rank_zero_info(f"No matching labels for {name}; skipping.")
                    continue
                
                contour_data_filtered = contour_data[keep_idx]
                case_ids_filtered = [case_ids[i] for i in keep_idx]
                sorted_keys = case_ids_filtered
                sorted_vals = [labels[k] for k in sorted_keys]
                
                # Align data by selecting entries with consistent length
                lengths = [len(v[0]) for v in sorted_vals if v and v[0] is not None]
                if not lengths:
                    rank_zero_info(f"No valid label entries for {name}; skipping.")
                    continue
                max_len = max(lengths)
                
                keep_indices = [i for i, s in enumerate(sorted_vals) if len(s[0]) == max_len]
                sorted_vals = [sorted_vals[i] for i in keep_indices]
                contour_data_filtered = contour_data_filtered[keep_indices]
                
                configs_list, directions_list, eye_widths_list, _, metas_list = zip(*sorted_vals)
                config_keys = metas_list[0]['config_keys']
                boundaries = convert_configs_to_boundaries(configs_list, config_keys)
                
                directions, eye_widths = map(np.array, (directions_list, eye_widths_list))
                eye_widths[eye_widths < 0] = 0
                
                # Convert contour data to variable-token format
                variable_data, sequence_data = self._convert_to_variable_format(
                    contour_data_filtered, boundaries, processor, config_keys
                )
                
                rank_zero_info(f"{name}| sequences {sequence_data.shape} | variables {len(variable_data)} | eye_width {eye_widths.shape}")
                
                # Create train/val/test splits
                indices = np.arange(len(sequence_data))
                
                if stage == "test":
                    test_idx = indices
                    seq_test = sequence_data[test_idx]
                    var_test = {k: v[test_idx] for k, v in variable_data.items()}
                    y_test = eye_widths[test_idx]
                    case_ids_test = [case_ids_filtered[i] for i in test_idx]
                    
                    self.test_dataset[name] = ContourDataset(
                        sequence_data=seq_test,
                        variable_data=var_test,
                        eye_widths=y_test,
                        case_ids=case_ids_test,
                        variable_registry=self.registry,
                        train=False,
                        enable_random_subspace=False
                    )
                    continue
                
                # Check dataset size for validation split
                total_size = eye_widths.shape[0] * eye_widths.shape[1]
                estimated_batch_size = max(1, self.batch_size // max(1, len(data_dirs_dict)))
                estimated_batches = max(1, total_size // estimated_batch_size)
                
                if estimated_batches < 10:
                    # Small dataset - use all for training
                    rank_zero_info(f"Dataset '{name}' too small for validation, using all {total_size} samples for training")
                    seq_tr, seq_val = sequence_data, np.array([])
                    var_tr = variable_data
                    var_val = {k: np.array([]) for k in variable_data.keys()}
                    y_tr, y_val = eye_widths, np.array([])
                    case_ids_tr = case_ids_filtered
                    case_ids_val = []
                else:
                    # Normal train/val split
                    train_idx, val_idx = train_test_split(
                        indices, test_size=self.test_size, shuffle=True, random_state=42
                    )
                    
                    seq_tr, seq_val = sequence_data[train_idx], sequence_data[val_idx]
                    var_tr = {k: v[train_idx] for k, v in variable_data.items()}
                    var_val = {k: v[val_idx] for k, v in variable_data.items()}
                    y_tr, y_val = eye_widths[train_idx], eye_widths[val_idx]
                    case_ids_tr = [case_ids_filtered[i] for i in train_idx]
                    case_ids_val = [case_ids_filtered[i] for i in val_idx]
                
                # Create training dataset
                self.train_dataset[name] = ContourDataset(
                    sequence_data=seq_tr,
                    variable_data=var_tr,
                    eye_widths=y_tr,
                    case_ids=case_ids_tr,
                    variable_registry=self.registry,
                    train=True,
                    enable_random_subspace=self.enable_random_subspace,
                    min_active_variables=self.min_active_variables,
                    max_active_variables=self.max_active_variables,
                    perturbation_scale=self.perturbation_scale
                )
                
                # Create validation dataset if we have data
                if len(seq_val) > 0:
                    self.val_dataset[name] = ContourDataset(
                        sequence_data=seq_val,
                        variable_data=var_val,
                        eye_widths=y_val,
                        case_ids=case_ids_val,
                        variable_registry=self.registry,
                        train=False,
                        enable_random_subspace=False
                    )
                
                rank_zero_info(f"Successfully created datasets for {name}: "
                              f"train={len(self.train_dataset[name])}, "
                              f"val={len(self.val_dataset.get(name, []))}")
                
            except Exception as e:
                rank_zero_info(f"Failed to process {name}: {e}")
                rank_zero_info(f"Full traceback:\n{traceback.format_exc()}")
                continue
    
    def _convert_to_variable_format(
        self, 
        contour_data: np.ndarray, 
        boundaries: np.ndarray,
        processor: ContourProcessor,
        config_keys: List[str]
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """Convert contour data to variable-token format."""
        
        # Extract sequence structure tokens (simplified - just use segment type info)
        # In practice, this would be more sophisticated based on sequence.csv structure
        n_cases, n_segments, n_features = contour_data.shape
        sequence_data = contour_data[:, :, 2].astype(int)  # Type column as sequence tokens
        
        # Extract variable data from boundaries (these are the parameters we want to vary)
        variable_data = {}
        
        # Use actual config_keys as variable names instead of generic param_i
        if len(config_keys) != boundaries.shape[-1]:
            rank_zero_info(f"Warning: config_keys length ({len(config_keys)}) doesn't match boundaries last dim ({boundaries.shape[-1]})")
            # Fallback to generic names
            config_keys = [f"param_{i}" for i in range(boundaries.shape[-1])]
        
        for i, var_name in enumerate(config_keys):
            if boundaries.ndim == 3:  # [cases, repetitions, params]
                variable_data[var_name] = boundaries[:, :, i]
            else:  # [cases, params]
                variable_data[var_name] = boundaries[:, i]
        
        # Register variables in the registry if not already present
        for var_name in variable_data.keys():
            if var_name not in self.registry:
                values = variable_data[var_name].flatten()
                var_min = float(values.min())
                var_max = float(values.max())
                var_range = var_max - var_min
                
                # Set bounds to None if variable has only one value (no variation)
                # Otherwise, set bounds to 2x the observed range for contour plotting
                if var_range < 1e-10:  # Effectively constant
                    bounds = None
                    rank_zero_info(f"Variable {var_name} has no variation (range={var_range:.2e}), bounds set to None")
                else:
                    # Expand range by 2x for contour plotting (centered on original range)
                    range_center = (var_min + var_max) / 2
                    expanded_range = var_range * 2
                    bounds = (range_center - expanded_range / 2, range_center + expanded_range / 2)
                    rank_zero_info(f"Variable {var_name}: original range=[{var_min:.6e}, {var_max:.6e}], "
                                  f"expanded bounds={bounds}")
                
                self.registry.register_variable(
                    name=var_name,
                    bounds=bounds,
                    role="geometry",  # Default role
                    description=f"Auto-generated variable {var_name}"
                )
        
        return variable_data, sequence_data

    def collate_fn(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Custom collate function for variable batches."""
        
        # Collect all variables from batch
        all_variables = {}
        sequence_tokens = []
        targets = []
        case_ids = []
        active_variables_list = []
        
        for item in batch:
            # Collect variables
            for var_name, var_value in item['variables'].items():
                if var_name not in all_variables:
                    all_variables[var_name] = []
                all_variables[var_name].append(var_value)
            
            sequence_tokens.append(item['sequence_tokens'])
            targets.append(item['targets'])
            case_ids.append(item['case_id'])
            active_variables_list.append(item['active_variables'])
        
        # Stack variables
        batched_variables = {}
        for var_name, var_list in all_variables.items():
            batched_variables[var_name] = torch.stack(var_list, dim=0)
        
        # Stack other tensors
        batched_sequence_tokens = torch.stack(sequence_tokens, dim=0)
        batched_targets = torch.stack(targets, dim=0)
        
        return {
            'variables': batched_variables,
            'sequence_tokens': batched_sequence_tokens, 
            'targets': batched_targets,
            'case_ids': case_ids,
            'active_variables': active_variables_list
        }

    def train_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        # Combine all training datasets
        combined_dataset = self._combine_datasets(self.train_dataset)
        
        return DataLoader(
            combined_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=2,
            pin_memory=True
        )

    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader.""" 
        combined_dataset = self._combine_datasets(self.val_dataset)
        
        return DataLoader(
            combined_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=2,
            pin_memory=True
        )

    def test_dataloader(self) -> DataLoader:
        """Create test dataloader."""
        combined_dataset = self._combine_datasets(self.test_dataset)
        
        return DataLoader(
            combined_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=2,
            pin_memory=True
        )

    def _combine_datasets(self, dataset_dict: Dict[str, ContourDataset]) -> Dataset:
        """Combine multiple datasets into one."""
        from torch.utils.data import ConcatDataset
        
        datasets = list(dataset_dict.values())
        if len(datasets) == 1:
            return datasets[0]
        elif len(datasets) > 1:
            return ConcatDataset(datasets)
        else:
            # Empty dataset
            return ContourDataset(
                sequence_data=np.array([[]]),
                variable_data={},
                eye_widths=np.array([]),
                case_ids=[],
                variable_registry=self.registry,
                train=False
            )
