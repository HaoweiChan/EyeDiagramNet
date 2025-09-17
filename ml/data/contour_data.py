import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
from lightning import LightningDataModule
from lightning.pytorch.utilities.rank_zero import rank_zero_info
from torch.utils.data import Dataset, DataLoader
import random

from ..utils.scaler import MinMaxScaler
from .processors import CSVProcessor
from .variable_registry import VariableRegistry
from common.parameters import convert_configs_to_boundaries
from common.pickle_utils import load_pickle_directory

# Optional sklearn import with fallback
try:
    from sklearn.model_selection import train_test_split
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    
    def train_test_split(X, test_size=0.2, shuffle=True, random_state=None):
        """Simple fallback implementation of train_test_split."""
        if random_state is not None:
            np.random.seed(random_state)
        
        n_samples = len(X)
        if shuffle:
            indices = np.random.permutation(n_samples)
        else:
            indices = np.arange(n_samples)
        
        n_test = int(n_samples * test_size)
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]
        
        return train_indices, test_indices


def get_loader_from_dataset(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = False
):
    """Create DataLoader from dataset with optimized settings."""
    import os
    
    dataset_size = len(dataset)
    
    # Adjust batch_size if it's larger than dataset to avoid issues
    effective_batch_size = min(batch_size, dataset_size)
    
    # Smart drop_last logic: ensure we never get 0 batches when we could get at least 1
    if dataset_size >= effective_batch_size:
        # Can form at least one batch
        if shuffle and dataset_size > effective_batch_size * 2:
            # Only drop last batch for datasets significantly larger than batch size
            drop_last = True
        else:
            # Keep all samples, especially for smaller datasets
            drop_last = False
    else:
        # Dataset smaller than batch size - definitely don't drop
        drop_last = False
    
    # Final safety check: if drop_last would result in 0 batches, force it to False
    potential_batches = dataset_size // effective_batch_size
    if drop_last and potential_batches == 1:
        drop_last = False  # Don't drop the only batch we have

    # Optimize num_workers based on system capabilities
    cpu_count = os.cpu_count()
    num_workers = min(4, cpu_count // 2) if cpu_count else 2

    loader = DataLoader(
        dataset=dataset,
        batch_size=effective_batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False,
        drop_last=drop_last,
    )
    return loader


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
        """Locate sequence.csv and variation.csv files in the given directory."""
        data_path = Path(data_dir)
        
        sequence_file = data_path / "sequence.csv"
        variation_file = data_path / "variation.csv"
        
        if not sequence_file.exists():
            raise FileNotFoundError(f"sequence.csv not found in {data_path}")
        if not variation_file.exists():
            raise FileNotFoundError(f"variation.csv not found in {data_path}")
            
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
        rank_zero_info(f"Found geometric features - Heights: {len(self.height_cols)}, "
                      f"Widths: {len(self.width_cols)}, Lengths: {len(self.length_cols)}")
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
        
        # Log the mappings
        rank_zero_info(f"Feature mappings created:")
        rank_zero_info(f"  Height mappings: {self.var_height_map}")
        rank_zero_info(f"  Width mappings: {self.var_width_map}")
        rank_zero_info(f"  Length mappings: {self.var_length_map}")

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
            case_id = int(var_row['Case'])
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
    """Dataset for contour data combining sequence and variation information with eye width labels."""
    
    def __init__(
        self,
        contour_data: np.ndarray,
        directions: np.ndarray,
        boundaries: np.ndarray,
        eye_widths: np.ndarray,
        case_ids: List[int],
        feature_info: Dict[str, int],
        train: bool = False
    ):
        super().__init__()
        
        self.contour_data = torch.from_numpy(contour_data).float()
        self.directions = torch.from_numpy(directions).int()
        self.boundaries = torch.from_numpy(boundaries).float()
        self.eye_widths = torch.from_numpy(eye_widths).float()
        self.case_ids = case_ids
        self.train = train
        self.feature_info = feature_info  # Store feature counts for slice calculation
        
        # Feature dimensions for reference
        self.n_cases, self.n_segments, self.n_features = self.contour_data.shape
        
        # Repetition represents the number of boundary conditions per case/sequence
        self.repetition = self.boundaries.size(1) if len(self.boundaries.shape) > 1 else 1
        
        rank_zero_info(f"ContourDataset initialized: {self.n_cases} cases, "
                      f"{self.n_segments} segments, {self.n_features} features, "
                      f"{self.repetition} boundary conditions per case")
        rank_zero_info(f"Feature breakdown: {self.feature_info}")

    def __len__(self) -> int:
        return self.n_cases * self.repetition

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        # Calculate seq_index (which case) and bnd_index (which boundary condition within that case)
        seq_index = index // self.repetition
        bnd_index = index % self.repetition
        
        # Get contour sequence for the case
        contour_seq = self.contour_data[seq_index]
        case_id = self.case_ids[seq_index]
        
        # Get direction, boundary, and eye width for the specific boundary condition
        direction = self.directions[seq_index, bnd_index]
        boundary = self.boundaries[seq_index, bnd_index]
        eye_width = self.eye_widths[seq_index, bnd_index]
        
        return contour_seq, direction, boundary, eye_width, case_id


class ContourVariableDataset(Dataset):
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
        if len(self.eye_widths.shape) == 2:
            target = self.eye_widths[seq_index, bnd_index]
        else:
            target = self.eye_widths[seq_index]
        
        # Apply random subspace perturbations if training
        active_variables = list(variables.keys())
        if self.enable_random_subspace:
            variables, active_variables = self._apply_random_subspace_perturbation(variables)
        
        return {
            'variables': variables,
            'sequence_tokens': sequence_tokens,
            'targets': target.unsqueeze(0),  # Ensure shape [1]
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
                # Get variable bounds and apply perturbation
                var_info = self.registry.get_variable(name)
                var_range = var_info.bounds[1] - var_info.bounds[0]
                noise_std = self.perturbation_scale * var_range
                
                # Add Gaussian noise
                noise = torch.randn_like(value) * noise_std
                perturbed_value = value + noise
                
                # Clamp to bounds
                perturbed_value = torch.clamp(
                    perturbed_value,
                    var_info.bounds[0],
                    var_info.bounds[1]
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

    def transform(self, seq_scaler, fix_scaler) -> 'ContourDataset':
        """Apply scaling transformations to the data."""
        # Scale only the continuous features (skip segment, layer, type indices)
        # Features structure: [segment, layer, type, heights..., widths..., length, material_props...]
        scalable_start_idx = 3  # Skip segment, layer, type
        
        scalable_data = self.contour_data[:, :, scalable_start_idx:]
        original_shape = scalable_data.shape
        
        # Flatten for scaling
        scalable_flat = scalable_data.reshape(-1, original_shape[-1])
        scaled_flat = seq_scaler.transform(scalable_flat)
        scaled_data = scaled_flat.reshape(original_shape)
        
        # Update the dataset contour data
        self.contour_data[:, :, scalable_start_idx:] = scaled_data.float()
        
        # Scale boundary features
        num = len(self.contour_data)
        bound_dim = self.boundaries.size(-1)
        scaled_boundary = fix_scaler.transform(self.boundaries).reshape(num, -1, bound_dim)
        self.boundaries = scaled_boundary.float()
        
        return self

    def get_feature_info(self) -> Dict[str, slice]:
        """Get information about feature slices for analysis."""
        start_idx = 3  # Skip segment, layer, type
        
        # Calculate dynamic slices based on actual feature counts
        heights_start = start_idx
        heights_end = heights_start + self.feature_info['n_heights']
        
        widths_start = heights_end
        widths_end = widths_start + self.feature_info['n_widths']
        
        lengths_start = widths_end
        lengths_end = lengths_start + self.feature_info['n_lengths']
        
        material_start = lengths_end
        material_end = material_start + 3  # Always 3 material properties
        
        return {
            'basic_info': slice(0, 3),  # segment, layer, type
            'heights': slice(heights_start, heights_end),
            'widths': slice(widths_start, widths_end),
            'lengths': slice(lengths_start, lengths_end), 
            'material': slice(material_start, material_end)
        }


class ContourVariableDataModule(LightningDataModule):
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
        self.train_dataset: Dict[str, ContourVariableDataset] = {}
        self.val_dataset: Dict[str, ContourVariableDataset] = {}
        self.test_dataset: Dict[str, ContourVariableDataset] = {}

    def setup(self, stage: Optional[str] = None, nan: int = -1):
        """Setup datasets by converting traditional data to variable-token format."""
        # TODO: Implement setup that processes contour/pickle data and converts to variable format
        # This would involve:
        # 1. Processing sequence.csv and variation.csv files
        # 2. Loading eye width labels from pickle files
        # 3. Converting to variable-token format compatible with ContourVariableDataset
        # 4. Creating train/val/test splits
        
        # For now, this is a placeholder - full implementation would depend on
        # the specific data format and processing pipeline
        rank_zero_info("ContourVariableDataModule.setup() - Implementation needed")
        pass

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

    def _combine_datasets(self, dataset_dict: Dict[str, ContourVariableDataset]) -> Dataset:
        """Combine multiple datasets into one."""
        from torch.utils.data import ConcatDataset
        
        datasets = list(dataset_dict.values())
        if len(datasets) == 1:
            return datasets[0]
        elif len(datasets) > 1:
            return ConcatDataset(datasets)
        else:
            # Empty dataset
            return ContourVariableDataset(
                sequence_data=np.array([[]]),
                variable_data={},
                eye_widths=np.array([]),
                case_ids=[],
                variable_registry=self.registry,
                train=False
            )


class ContourDataModule(LightningDataModule):
    """Lightning DataModule for contour data with eye width labels."""
    
    def __init__(
        self,
        data_dirs: Union[Dict[str, str], List[str]],
        label_dir: str,
        batch_size: int = 32,
        test_size: float = 0.2,
        scaler_path: Optional[str] = None,
    ):
        super().__init__()
        self.data_dirs = data_dirs
        self.label_dir = Path(label_dir)
        self.batch_size = batch_size
        self.test_size = test_size
        self.scaler_path = scaler_path
        
        # Initialize datasets
        self.train_dataset: Dict[str, ContourDataset] = {}
        self.val_dataset: Dict[str, ContourDataset] = {}
        self.test_dataset: Dict[str, ContourDataset] = {}

    def setup(self, stage: Optional[str] = None, nan: int = -1):
        """Setup the datasets for training, validation, and testing."""
        # Scalers (similar to eyewidth_data.py)
        fit_scaler = True
        try:
            if self.scaler_path is None:
                raise FileNotFoundError("No scaler path provided")
            # Use weights_only=False for backward compatibility with custom scaler classes
            self.seq_scaler, self.fix_scaler = torch.load(self.scaler_path, weights_only=False)
            rank_zero_info(f"Loaded scalers from {self.scaler_path}")
            fit_scaler = False
        except (FileNotFoundError, AttributeError, EOFError) as e:
            # In test/predict modes, we must have valid scalers - don't create new ones
            if stage in ["test", "predict"] or stage is None:
                error_msg = f"Cannot find or load scaler file for {stage or 'test/predict'} mode"
                if self.scaler_path:
                    error_msg += f" at path: {self.scaler_path}"
                else:
                    error_msg += " (no scaler_path provided)"
                error_msg += f". Original error: {e}"
                rank_zero_info(f"ERROR: {error_msg}")
                raise FileNotFoundError(error_msg)
            
            # Only create new scalers during training
            self.seq_scaler = MinMaxScaler(nan=nan)
            self.fix_scaler = MinMaxScaler(nan=nan)
            rank_zero_info("Could not find scalers on disk, creating new ones for training.")

        # Locate every CSV once via processor
        csv_processor = CSVProcessor()
        csv_paths = csv_processor.locate(self.data_dirs)  # dict[str, Path]
        
        # Process each named dataset
        for name, csv_path in csv_paths.items():
            case_ids, input_arr = csv_processor.parse(csv_path)
            
            # Load labels using unified pickle directory loader
            labels = load_pickle_directory(self.label_dir, name)
            
            # Keep only indices present in labels
            label_keys = set(labels.keys())
            keep_idx = [i for i, cid in enumerate(case_ids) if cid in label_keys]
            input_arr = input_arr[keep_idx]
            sorted_keys = [case_ids[i] for i in keep_idx]
            sorted_vals = [labels[k] for k in sorted_keys]
            
            # Align tensors by selecting entries that match the maximum length
            lengths = [len(v[0]) for v in sorted_vals if v and v[0] is not None]
            if not lengths:
                rank_zero_info(f"No valid label entries for {name}; skipping.")
                continue
            max_len = max(lengths)
            
            keep_indices = [i for i, s in enumerate(sorted_vals) if len(s[0]) == max_len]
            sorted_vals = [sorted_vals[i] for i in keep_indices]
            input_arr = input_arr[keep_indices]
            
            configs_list, directions_list, eye_widths_list, _, metas_list = zip(*sorted_vals)
            config_keys = metas_list[0]['config_keys']
            boundaries = convert_configs_to_boundaries(configs_list, config_keys)
            
            directions, eye_widths = map(np.array, (directions_list, eye_widths_list))
            eye_widths[eye_widths < 0] = 0
            
            rank_zero_info(f"{name}| input_seq {input_arr.shape} | eye_width {eye_widths.shape}")
            
            # Train/val/test split
            indices = np.arange(len(input_arr))
            
            # For test stage, use the entire dataset as test set
            if stage == "test":
                test_idx = indices
                x_seq_test = input_arr[test_idx]
                x_tok_test = directions[test_idx]
                x_fix_test = boundaries[test_idx]
                y_test = eye_widths[test_idx]
                case_ids_test = [case_ids[i] for i in test_idx]
                
                # Create test dataset
                self.test_dataset[name] = ContourDataset(
                    x_seq_test, x_tok_test, x_fix_test, y_test, case_ids_test, 
                    {'n_heights': 0, 'n_widths': 0, 'n_lengths': 0}, train=False
                )
                # Apply scaling if scalers are available
                if hasattr(self, 'seq_scaler') and hasattr(self, 'fix_scaler'):
                    self.test_dataset[name] = self.test_dataset[name].transform(
                        self.seq_scaler, self.fix_scaler
                    )
                continue
            
            # Check if dataset is too small for proper validation
            total_dataset_size = eye_widths.shape[0] * eye_widths.shape[1]
            estimated_batch_size = max(1, self.batch_size // max(1, len(csv_paths)))
            estimated_batches = max(1, total_dataset_size // estimated_batch_size)
            
            if estimated_batches < 10:
                # Dataset too small - use all samples for training
                rank_zero_info(f"Dataset '{name}' too small for validation (~{estimated_batches} batches < 10), using all {total_dataset_size} samples for training")
                x_seq_tr, x_seq_val = input_arr, np.array([])
                x_tok_tr, x_tok_val = directions, np.array([])
                x_fix_tr, x_fix_val = boundaries, np.array([]).reshape(0, boundaries.shape[-1])
                y_tr, y_val = eye_widths, np.array([])
                case_ids_tr = case_ids
                case_ids_val = []
            else:
                # Normal train/val split  
                train_idx, val_idx = train_test_split(
                    indices, test_size=self.test_size, shuffle=True, random_state=42
                )
                
                x_seq_tr, x_seq_val = input_arr[train_idx], input_arr[val_idx]
                x_tok_tr, x_tok_val = directions[train_idx], directions[val_idx]
                x_fix_tr, x_fix_val = boundaries[train_idx], boundaries[val_idx]
                y_tr, y_val = eye_widths[train_idx], eye_widths[val_idx]
                case_ids_tr = [case_ids[i] for i in train_idx]
                case_ids_val = [case_ids[i] for i in val_idx]
            
            # Feature info (placeholder - would need to be computed from contour processing)
            feature_info = {'n_heights': 0, 'n_widths': 0, 'n_lengths': 0}
            
            # Create datasets
            self.train_dataset[name] = ContourDataset(
                x_seq_tr, x_tok_tr, x_fix_tr, y_tr, case_ids_tr, feature_info, train=True
            )
            
            if len(x_seq_val) > 0:
                self.val_dataset[name] = ContourDataset(
                    x_seq_val, x_tok_val, x_fix_val, y_val, case_ids_val, feature_info, train=False
                )
            
            # Fit scalers on training data
            if fit_scaler:
                # Fit seq_scaler on contour sequences (skip first 3 categorical features)
                scalable_start_idx = 3
                scalable_data = x_seq_tr[:, :, scalable_start_idx:]
                scalable_flat = scalable_data.reshape(-1, scalable_data.shape[-1])
                self.seq_scaler.fit(scalable_flat)
                
                # Fit fix_scaler on boundaries
                bound_flat = x_fix_tr.reshape(-1, x_fix_tr.shape[-1])
                self.fix_scaler.fit(bound_flat)
                
                fit_scaler = False  # Only fit once
                
                # Save scalers
                if self.scaler_path:
                    torch.save((self.seq_scaler, self.fix_scaler), self.scaler_path)
                    rank_zero_info(f"Saved scalers to {self.scaler_path}")
            
            # Apply scaling to datasets
            self.train_dataset[name] = self.train_dataset[name].transform(
                self.seq_scaler, self.fix_scaler
            )
            
            if name in self.val_dataset:
                self.val_dataset[name] = self.val_dataset[name].transform(
                    self.seq_scaler, self.fix_scaler
                )

    def _normalize_data_dirs(self) -> Dict[str, str]:
        """Normalize data_dirs to a consistent dict format."""
        if isinstance(self.data_dirs, list):
            return {f"contour_{i}": dir_path for i, dir_path in enumerate(self.data_dirs)}
        return self.data_dirs

    def train_dataloader(self) -> DataLoader:
        """Create training data loader."""
        if not self.train_dataset:
            raise RuntimeError("No training datasets available")
        
        # Calculate batch size per dataset
        num_datasets = len(self.train_dataset)
        per_dataset_batch_size = max(1, self.batch_size // num_datasets)
        
        loaders = {}
        for name, dataset in self.train_dataset.items():
            loader = get_loader_from_dataset(
                dataset, batch_size=per_dataset_batch_size, shuffle=True
            )
            loaders[name] = loader
            rank_zero_info(f"Train loader '{name}': {len(dataset)} samples, "
                          f"batch_size={per_dataset_batch_size}, {len(loader)} batches")
        
        if len(loaders) == 1:
            return list(loaders.values())[0]
        else:
            from lightning.pytorch.utilities import CombinedLoader
            return CombinedLoader(loaders, mode="max_size_cycle")

    def val_dataloader(self) -> Optional[DataLoader]:
        """Create validation data loader."""
        if not self.val_dataset:
            return None
        
        num_datasets = len(self.val_dataset)
        per_dataset_batch_size = max(1, int(self.batch_size * 1.5) // num_datasets)
        
        loaders = {}
        for name, dataset in self.val_dataset.items():
            loader = get_loader_from_dataset(
                dataset, batch_size=per_dataset_batch_size, shuffle=False
            )
            loaders[name] = loader
            rank_zero_info(f"Val loader '{name}': {len(dataset)} samples, "
                          f"batch_size={per_dataset_batch_size}, {len(loader)} batches")
        
        if len(loaders) == 1:
            return list(loaders.values())[0]
        else:
            from lightning.pytorch.utilities import CombinedLoader
            return CombinedLoader(loaders, mode="min_size")

    def test_dataloader(self) -> Optional[DataLoader]:
        """Create test data loader."""
        if not self.test_dataset:
            return None
        
        num_datasets = len(self.test_dataset)
        per_dataset_batch_size = max(1, int(self.batch_size * 1.5) // num_datasets)
        
        loaders = {}
        for name, dataset in self.test_dataset.items():
            loader = get_loader_from_dataset(
                dataset, batch_size=per_dataset_batch_size, shuffle=False
            )
            loaders[name] = loader
            rank_zero_info(f"Test loader '{name}': {len(dataset)} samples, "
                          f"batch_size={per_dataset_batch_size}, {len(loader)} batches")
        
        if len(loaders) == 1:
            return list(loaders.values())[0]
        else:
            from lightning.pytorch.utilities import CombinedLoader
            return CombinedLoader(loaders, mode="min_size")
