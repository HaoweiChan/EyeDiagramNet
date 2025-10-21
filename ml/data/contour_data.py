import random
import traceback
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
from lightning import LightningDataModule
from lightning.pytorch.utilities.rank_zero import rank_zero_info

from ..utils.scaler import MinMaxScaler
from .processors import ContourProcessor
from .variable_registry import VariableRegistry
from common.parameters import convert_configs_to_boundaries
from common.pickle_utils import load_pickle_directory


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
                # Handle both integer and string case IDs (string format: "case_id_suffix")
                label_keys = set(labels.keys())
                keep_idx = []
                for i, cid in enumerate(case_ids):
                    # Check direct match first
                    if cid in label_keys:
                        keep_idx.append(i)
                    # If cid is a string with suffix (e.g., "1_L=1800um_Wgr=2um"), try extracting base case_id
                    elif isinstance(cid, str) and "_" in cid:
                        try:
                            base_case_id = int(cid.split("_")[0])
                            if base_case_id in label_keys:
                                keep_idx.append(i)
                        except (ValueError, IndexError):
                            pass
                
                if not keep_idx:
                    rank_zero_info(f"No matching labels for {name}; skipping.")
                    continue
                
                contour_data_filtered = contour_data[keep_idx]
                case_ids_filtered = [case_ids[i] for i in keep_idx]
                
                # Build sorted_keys and sorted_vals, handling string case IDs with suffixes
                sorted_keys = []
                sorted_vals = []
                has_string_case_ids = any(isinstance(cid, str) for cid in case_ids_filtered)
                
                for cid in case_ids_filtered:
                    # Try direct lookup first
                    if cid in labels:
                        sorted_keys.append(cid)
                        sorted_vals.append(labels[cid])
                    # If cid is string with suffix, extract base case_id for label lookup
                    elif isinstance(cid, str) and "_" in cid:
                        try:
                            base_case_id = int(cid.split("_")[0])
                            if base_case_id in labels:
                                sorted_keys.append(cid)  # Keep full case_id with suffix
                                sorted_vals.append(labels[base_case_id])  # Use base case_id for label lookup
                        except (ValueError, IndexError):
                            pass
                
                if has_string_case_ids:
                    rank_zero_info(f"Processing {len(sorted_keys)} cases with unique case IDs (format: case_id_suffix)")
                    sample_keys = sorted_keys[:min(3, len(sorted_keys))]
                    rank_zero_info(f"Sample unique case IDs: {sample_keys}")
                
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
                
                # Perform data sanity checks and get list of valid variables
                valid_variables = self._validate_data_quality(variable_data, eye_widths, name)
                
                # Filter out constant/zero-variance variables for GP training
                if len(valid_variables) < len(variable_data):
                    rank_zero_info(f"Filtering variables for GP: keeping {len(valid_variables)}/{len(variable_data)} non-constant variables")
                    variable_data_filtered = {k: v for k, v in variable_data.items() if k in valid_variables}
                else:
                    variable_data_filtered = variable_data
                
                rank_zero_info(f"{name}| sequences {sequence_data.shape} | variables {len(variable_data_filtered)} | eye_width {eye_widths.shape}")
                
                # Create train/val/test splits
                indices = np.arange(len(sequence_data))
                
                if stage == "test":
                    test_idx = indices
                    seq_test = sequence_data[test_idx]
                    var_test = {k: v[test_idx] for k, v in variable_data_filtered.items()}
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
                    var_tr = variable_data_filtered
                    var_val = {k: np.array([]) for k in variable_data_filtered.keys()}
                    y_tr, y_val = eye_widths, np.array([])
                    case_ids_tr = case_ids_filtered
                    case_ids_val = []
                else:
                    # Normal train/val split
                    train_idx, val_idx = train_test_split(
                        indices, test_size=self.test_size, shuffle=True, random_state=42
                    )
                    
                    seq_tr, seq_val = sequence_data[train_idx], sequence_data[val_idx]
                    var_tr = {k: v[train_idx] for k, v in variable_data_filtered.items()}
                    var_val = {k: v[val_idx] for k, v in variable_data_filtered.items()}
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
        
        # Register boundary parameters (R, C, L) with clipping
        variable_data = VariableRegistry.register_boundary_parameters(variable_data, self.registry)
        
        # Extract and register GEOMETRIC parameters from contour_data (heights, widths, lengths)
        geometric_features = self._extract_geometric_features(contour_data, processor)
        
        # Aggregate geometric features per case (max across segments)
        # Geometric features are [n_cases, n_segments], but we need [n_cases] or [n_cases, n_repetitions]
        # Use max to represent the maximum extent of each geometric parameter in the structure
        aggregated_geometric_features = {}
        for geom_name, geom_values in geometric_features.items():
            # Take max across segments (axis=1) to get [n_cases] 
            # Then expand to match boundaries shape if needed
            geom_max_per_case = np.max(geom_values, axis=1)  # [n_cases]
            
            # Match the shape of boundary parameters for consistency
            if boundaries.ndim == 3:  # [n_cases, n_repetitions, n_params]
                # Expand to [n_cases, n_repetitions] by broadcasting
                n_repetitions = boundaries.shape[1]
                geom_expanded = np.tile(geom_max_per_case[:, np.newaxis], (1, n_repetitions))
                aggregated_geometric_features[geom_name] = geom_expanded
            else:  # [n_cases, n_params]
                # Keep as [n_cases]
                aggregated_geometric_features[geom_name] = geom_max_per_case
        
        # Register geometric parameters with proper bounds
        aggregated_geometric_features = VariableRegistry.register_geometric_parameters(
            aggregated_geometric_features, self.registry
        )
        
        # Add aggregated geometric features to variable_data
        for geom_name, geom_values in aggregated_geometric_features.items():
            variable_data[geom_name] = geom_values
        
        return variable_data, sequence_data
    
    def _extract_geometric_features(
        self,
        contour_data: np.ndarray,
        processor: ContourProcessor
    ) -> Dict[str, np.ndarray]:
        """
        Extract geometric parameter values from resolved contour data.
        
        Args:
            contour_data: Resolved contour data [cases, segments, features]
            processor: ContourProcessor with geometric feature info
            
        Returns:
            Dict mapping geometric parameter names to their values across all cases
        """
        geometric_features = {}
        feature_info = processor.get_feature_info()
        
        # The contour data structure is:
        # [segment, layer, type, resolved_heights, resolved_widths, resolved_length, material_props]
        # Features start at index 3
        
        n_heights = feature_info['n_heights']
        n_widths = feature_info['n_widths']
        n_lengths = feature_info['n_lengths']
        
        # Extract heights
        height_start = 3
        for i, height_name in enumerate(processor.height_cols):
            height_idx = height_start + i
            if height_idx < contour_data.shape[2]:
                # Get all values across cases and segments
                height_values = contour_data[:, :, height_idx]
                geometric_features[height_name] = height_values
        
        # Extract widths
        width_start = height_start + n_heights
        for i, width_name in enumerate(processor.width_cols):
            width_idx = width_start + i
            if width_idx < contour_data.shape[2]:
                width_values = contour_data[:, :, width_idx]
                geometric_features[width_name] = width_values
        
        # Extract lengths
        length_start = width_start + n_widths
        for i, length_name in enumerate(processor.length_cols):
            length_idx = length_start + i
            if length_idx < contour_data.shape[2]:
                length_values = contour_data[:, :, length_idx]
                geometric_features[length_name] = length_values
        
        return geometric_features

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

    def _validate_data_quality(
        self,
        variable_data: Dict[str, np.ndarray],
        eye_widths: np.ndarray,
        dataset_name: str
    ) -> List[str]:
        """
        Perform comprehensive data quality checks to identify issues that could cause GP training failures.
        
        This function checks for:
        - NaN/Inf values
        - Duplicate samples
        - Zero variance in variables or targets
        - Extremely small ranges that could cause numerical issues
        
        Returns:
            List of valid (non-constant, non-problematic) variable names suitable for GP training
        """
        rank_zero_info(f"\n{'='*60}")
        rank_zero_info(f"DATA QUALITY CHECKS for '{dataset_name}'")
        rank_zero_info(f"{'='*60}")
        
        issues_found = []
        valid_variables = []
        
        # Check eye widths (targets)
        rank_zero_info(f"\n[Target Data] Eye Widths:")
        eye_widths_flat = eye_widths.flatten()
        
        if np.any(np.isnan(eye_widths_flat)):
            n_nan = np.sum(np.isnan(eye_widths_flat))
            issues_found.append(f"Found {n_nan} NaN values in eye_widths")
            rank_zero_info(f"  ⚠️  WARNING: {n_nan} NaN values found!")
        
        if np.any(np.isinf(eye_widths_flat)):
            n_inf = np.sum(np.isinf(eye_widths_flat))
            issues_found.append(f"Found {n_inf} Inf values in eye_widths")
            rank_zero_info(f"  ⚠️  WARNING: {n_inf} Inf values found!")
        
        valid_eye_widths = eye_widths_flat[~np.isnan(eye_widths_flat) & ~np.isinf(eye_widths_flat)]
        if len(valid_eye_widths) > 0:
            ew_std = np.std(valid_eye_widths)
            ew_mean = np.mean(valid_eye_widths)
            ew_min = np.min(valid_eye_widths)
            ew_max = np.max(valid_eye_widths)
            
            rank_zero_info(f"  Mean: {ew_mean:.6f}, Std: {ew_std:.6f}")
            rank_zero_info(f"  Range: [{ew_min:.6f}, {ew_max:.6f}]")
            
            if ew_std < 1e-6:
                issues_found.append(f"Eye widths have near-zero variance (std={ew_std:.2e})")
                rank_zero_info(f"  ⚠️  WARNING: Very low variance (std={ew_std:.2e}) - all targets nearly identical!")
        
        # Check each variable
        rank_zero_info(f"\n[Variable Data] Checking {len(variable_data)} variables:")
        
        for var_name, var_values in variable_data.items():
            var_flat = var_values.flatten()
            var_issues = []
            is_valid = True
            
            # Check for NaN/Inf
            n_nan = np.sum(np.isnan(var_flat))
            n_inf = np.sum(np.isinf(var_flat))
            
            if n_nan > 0:
                var_issues.append(f"{n_nan} NaN")
                issues_found.append(f"Variable '{var_name}' has {n_nan} NaN values")
                is_valid = False
            
            if n_inf > 0:
                var_issues.append(f"{n_inf} Inf")
                issues_found.append(f"Variable '{var_name}' has {n_inf} Inf values")
                is_valid = False
            
            # Get valid values
            valid_values = var_flat[~np.isnan(var_flat) & ~np.isinf(var_flat)]
            
            if len(valid_values) > 0:
                var_std = np.std(valid_values)
                var_mean = np.mean(valid_values)
                var_min = np.min(valid_values)
                var_max = np.max(valid_values)
                var_range = var_max - var_min
                
                # Check for zero/near-zero variance (threshold: 1e-10 for zero, 1e-8 for near-zero)
                if var_std < 1e-10:
                    var_issues.append("ZERO VARIANCE - EXCLUDED")
                    issues_found.append(f"Variable '{var_name}' has zero variance - all values are {var_mean:.6f}")
                    is_valid = False
                elif var_std < 1e-8:
                    var_issues.append(f"Very low variance (std={var_std:.2e}) - EXCLUDED")
                    issues_found.append(f"Variable '{var_name}' has near-zero variance (std={var_std:.2e})")
                    is_valid = False
                elif var_std < 1e-6:
                    var_issues.append(f"Low variance (std={var_std:.2e})")
                
                # Check for very small range
                if var_range < 1e-10:
                    var_issues.append("Zero range")
                    is_valid = False
                
                # Report statistics
                status = "✓ VALID" if is_valid else "✗ EXCLUDED"
                issue_str = f" [{', '.join(var_issues)}]" if var_issues else f" [{status}]"
                rank_zero_info(f"  {var_name}: mean={var_mean:.6f}, std={var_std:.6f}, range=[{var_min:.6f}, {var_max:.6f}]{issue_str}")
                
                # Add to valid variables list if it passes all checks
                if is_valid:
                    valid_variables.append(var_name)
            else:
                rank_zero_info(f"  {var_name}: ⚠️  NO VALID VALUES (all NaN/Inf) - EXCLUDED")
                issues_found.append(f"Variable '{var_name}' has no valid values")
                is_valid = False
        
        # Check for duplicate samples (combining ONLY valid variable values)
        rank_zero_info(f"\n[Duplicate Detection] (using {len(valid_variables)} valid variables):")
        
        # Stack only valid variables into a feature matrix
        feature_list = []
        for var_name in sorted(valid_variables):
            if var_name in variable_data:
                var_values = variable_data[var_name]
                # Flatten to [n_samples]
                if var_values.ndim > 1:
                    # Take first dimension as samples, flatten rest
                    var_flat = var_values.reshape(var_values.shape[0], -1).mean(axis=1)
                else:
                    var_flat = var_values
                feature_list.append(var_flat)
        
        if feature_list:
            features = np.column_stack(feature_list)
            
            # Find duplicates
            unique_features, indices, counts = np.unique(
                features, axis=0, return_index=True, return_counts=True
            )
            
            n_duplicates = np.sum(counts > 1)
            n_total_duplicate_rows = np.sum(counts[counts > 1])
            
            rank_zero_info(f"  Total samples: {len(features)}")
            rank_zero_info(f"  Unique samples: {len(unique_features)}")
            
            if n_duplicates > 0:
                rank_zero_info(f"  ⚠️  WARNING: Found {n_duplicates} duplicate sample groups ({n_total_duplicate_rows} total duplicates)")
                issues_found.append(f"Found {n_duplicates} groups of duplicate samples (after filtering constant variables)")
                
                # Show first few duplicate groups
                dup_groups = counts[counts > 1][:3]
                rank_zero_info(f"  First few duplicate counts: {dup_groups}")
            else:
                rank_zero_info(f"  ✓ No duplicates found")
        
        # Final summary
        rank_zero_info(f"\n{'='*60}")
        rank_zero_info(f"VALIDATION SUMMARY:")
        rank_zero_info(f"  Valid variables for GP: {len(valid_variables)}/{len(variable_data)}")
        rank_zero_info(f"  Excluded variables: {len(variable_data) - len(valid_variables)}")
        
        if len(variable_data) > len(valid_variables):
            excluded = [v for v in variable_data.keys() if v not in valid_variables]
            rank_zero_info(f"  Excluded list: {excluded}")
        
        if issues_found:
            rank_zero_info(f"\n⚠️  FOUND {len(issues_found)} ISSUES (auto-filtered):")
            for i, issue in enumerate(issues_found, 1):
                rank_zero_info(f"  {i}. {issue}")
            rank_zero_info(f"\n💡 Problematic variables have been automatically excluded from GP training.")
        else:
            rank_zero_info(f"\n✓ No critical data quality issues detected")
        
        if len(valid_variables) == 0:
            rank_zero_info(f"\n❌ ERROR: No valid variables remaining after filtering!")
            rank_zero_info(f"   Cannot proceed with GP training.")
        
        rank_zero_info(f"{'='*60}\n")
        
        return valid_variables
    
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
