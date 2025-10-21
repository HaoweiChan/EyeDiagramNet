import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Union
from lightning.pytorch.utilities.rank_zero import rank_zero_info

class CSVProcessor:
    def __init__(self, patterns: List[str] = None, padding_value: int = -1):
        self.patterns = patterns or [
            "input_for_AI*.csv",
            "*AI_input_data*.csv",
            "*AI_input*.csv",
            "*ai*.csv",
        ]
        self.padding = padding_value

    def locate(self, data_dirs: Union[Dict[str, str], List[str]]) -> Union[Dict[str, Path], List[Path]]:
        is_dict = isinstance(data_dirs, dict)
        out = {} if is_dict else []
        items = data_dirs.items() if is_dict else enumerate(data_dirs)

        for key, dir_path in items:
            rank_zero_info(f"Parsing data from {dir_path}")
            matches = [
                p
                for pat in self.patterns
                for p in Path(dir_path).glob(pat)
            ]
            if len(matches) != 1:
                raise FileNotFoundError(
                    f"Expected exactly one match in {dir_path}, got {matches}"
                )
            if is_dict:
                out[key] = matches[0]
            else:
                out.append(matches[0])
        return out

    def parse(self, csv_path: Union[Path, str]) -> Tuple[np.ndarray, np.ndarray]:
        df = pd.read_csv(csv_path, index_col=0, header=0)
        types = ("S", "G", "D")
        type_cols = df.columns[df.columns.str.contains("Type_")]
        df[type_cols] = df[type_cols].replace({t: i for i, t in enumerate(types)})

        mats = [self._spatial_feats(row.dropna()) for _, row in df.iterrows()]
        max_len = max(m.shape[0] for m in mats)
        padded = np.stack(
            [
                np.pad(
                    m,
                    ((0, max_len - m.shape[0]), (0, 0)),
                    constant_values=self.padding
                )
                for m in mats
            ]
        )
        return df.index.values, padded

    def _spatial_feats(self, case: pd.Series) -> np.ndarray:
        idx = case.index
        layer_mask = idx.str.contains("Layer_")
        width_mask = idx.str.contains("W_")
        height_mask = idx.str.contains("H_")

        layers = case[layer_mask].astype(int).values
        widths = case[width_mask].values
        heights = case[height_mask].values

        # per-trace fields
        layer_change = np.r_[True, np.diff(layers) != 0]
        _, layer_count = np.unique(layers, return_counts=True)

        # x coordinate: cumulative widths (shifted by 1)
        cum_x = np.r_[0, widths.cumsum()[:-1]]
        x_dim = cum_x - np.repeat(cum_x[layer_change], layer_count)

        # z coordinate: bottom height of each layer
        cum_h = heights[layer_change].cumsum()
        cum_h = np.roll(cum_h, 1)
        cum_h[0] = 0
        z_dim = np.repeat(cum_h, layer_count)

        # original feature block
        layer_idx = np.flatnonzero(layer_mask)
        feat_dim = layer_idx[1] - layer_idx[0]
        data_col = case.values.reshape(-1, feat_dim)
        
        return np.hstack([data_col, x_dim[:, None], z_dim[:, None]])

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

    def locate_files(self, data_dir: Union[str, Path]) -> Tuple[List[Path], List[Path]]:
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

        # Find matching prefixes - handle cases where variation files have additional parameters
        common_prefixes = self._find_matching_prefixes(seq_prefixes, var_prefixes)

        if not common_prefixes:
            raise FileNotFoundError(
                f"No matching sequence/variation file pairs found in {data_path}. "
                f"Sequence prefixes: {list(seq_prefixes.keys())}, "
                f"Variation prefixes: {list(var_prefixes.keys())}"
            )

        # Get all matching file pairs
        sequence_files_matched = []
        variation_files_matched = []

        for seq_prefix in sorted(common_prefixes):
            sequence_files_matched.append(seq_prefixes[seq_prefix])

            # Find corresponding variation files for this sequence prefix
            seq_variation_files = []
            for var_prefix in var_prefixes.keys():
                if var_prefix == seq_prefix or var_prefix.startswith(seq_prefix + '_'):
                    seq_variation_files.append(var_prefixes[var_prefix])

            if seq_variation_files:
                variation_files_matched.extend(seq_variation_files)
            else:
                # Fallback: use the original logic if no matches found
                if seq_prefix in var_prefixes:
                    variation_files_matched.append(var_prefixes[seq_prefix])

        rank_zero_info(f"Found {len(common_prefixes)} matching file pair(s)")

        return sequence_files_matched, variation_files_matched

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

    def _find_matching_prefixes(self, seq_prefixes: Dict[str, Path], var_prefixes: Dict[str, Path]) -> List[str]:
        """Find matching prefixes between sequence and variation files.

        Handles cases where variation files have longer prefixes that start with sequence prefixes.
        For example: seq='UCle_pattern2_cowos-s_8mil', var='UCle_pattern2_cowos-s_8mil_L=1800um_Wgr=2um'
        """
        common_prefixes = []

        for seq_prefix in seq_prefixes.keys():
            # First try exact match
            if seq_prefix in var_prefixes:
                common_prefixes.append(seq_prefix)
                continue

            # Try to find variation files that start with the sequence prefix
            for var_prefix in var_prefixes.keys():
                if var_prefix.startswith(seq_prefix + '_'):
                    common_prefixes.append(seq_prefix)
                    break

        return common_prefixes

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

    def _extract_parameter_suffix(self, variation_file: Path) -> str:
        """Extract parameter suffix from variation filename for unique case ID creation.
        
        Example:
            Input: "UCle_pattern2_cowos-s_8mil_L=1800um_Wgr=2um_variations_variable.csv"
            Output: "L=1800um_Wgr=2um"
        
        Args:
            variation_file: Path to variation file
            
        Returns:
            Parameter suffix string (e.g., "L=1800um_Wgr=2um")
        """
        filename = variation_file.name
        
        # Remove the "_variations_variable.csv" suffix
        if "_variations_variable.csv" in filename:
            base_part = filename.replace("_variations_variable.csv", "")
            
            # Split by underscore and find where parameters start (after base pattern)
            # Base pattern is typically "prefix_pattern_type_layer" (e.g., "UCle_pattern2_cowos-s_8mil")
            # Parameters come after that (e.g., "L=1800um_Wgr=2um")
            parts = base_part.split("_")
            
            # Look for the first part that contains "=" (indicates parameter)
            param_start_idx = None
            for i, part in enumerate(parts):
                if "=" in part:
                    param_start_idx = i
                    break
            
            if param_start_idx is not None:
                # Join all parts from the parameter start
                suffix = "_".join(parts[param_start_idx:])
            else:
                # No parameters found, use empty suffix
                suffix = ""
        else:
            suffix = ""
        
        return suffix

    def process(self, data_dir: Union[str, Path]) -> Tuple[np.ndarray, List[Union[int, str]]]:
        """Process sequence and variation files to create resolved contour data.
        
        Returns:
            Tuple of (resolved_data, case_ids) where case_ids are strings formatted as 
            "case_id_suffix" when multiple variation files exist, or integers when only one file exists.
        """
        sequence_files, variation_files = self.locate_files(data_dir)

        # Use the first sequence file (typically there's only one)
        sequence_df = self.parse_sequence(sequence_files[0])

        # Extract parameter suffixes from variation filenames for unique case ID creation
        variation_file_suffixes = {}
        for variation_file in variation_files:
            suffix = self._extract_parameter_suffix(variation_file)
            variation_file_suffixes[variation_file] = suffix

        # Parse all variation files and track their source file
        variation_dfs = []
        variation_sources = []  # Track which file each row came from
        
        for variation_file in variation_files:
            variation_df = self.parse_variation(variation_file)
            variation_dfs.append(variation_df)
            # Store the file path for each row in this dataframe
            variation_sources.extend([variation_file] * len(variation_df))

        # Concatenate all variation dataframes
        if variation_dfs:
            combined_variation_df = pd.concat(variation_dfs, ignore_index=True)
        else:
            combined_variation_df = pd.DataFrame()

        # Sync features between sequence and variation files (using first variation file for structure)
        if variation_dfs:
            self._sync_variation_features(variation_dfs[0])

        resolved_cases = []
        case_ids = []
        use_suffixes = len(variation_files) > 1  # Only add suffixes if multiple variation files

        for idx, var_row in enumerate(combined_variation_df.iterrows()):
            _, var_row = var_row  # Unpack tuple from iterrows
            original_case_id = int(var_row['case_id'])
            
            # Create unique case ID if multiple variation files
            if use_suffixes:
                source_file = variation_sources[idx]
                suffix = variation_file_suffixes[source_file]
                case_id = f"{original_case_id}_{suffix}"
            else:
                case_id = original_case_id
            
            resolved_case = self.resolve_case(sequence_df, var_row)
            resolved_cases.append(resolved_case)
            case_ids.append(case_id)

        # Stack all cases
        resolved_data = np.stack(resolved_cases) if resolved_cases else np.array([])

        if use_suffixes:
            rank_zero_info(f"Processed {len(resolved_cases)} cases from {len(variation_files)} variation file(s) with shape: {resolved_data.shape}")
            rank_zero_info(f"Created unique case IDs with suffixes from variation file parameters")
            # Show sample case IDs for verification
            sample_case_ids = case_ids[:min(3, len(case_ids))]
            rank_zero_info(f"Sample case IDs: {sample_case_ids}")
        else:
            rank_zero_info(f"Processed {len(resolved_cases)} cases from {len(variation_files)} variation file(s) with shape: {resolved_data.shape}")
        
        return resolved_data, case_ids
    
    def get_feature_info(self) -> Dict[str, int]:
        """Get information about the number of each feature type."""
        return {
            'n_heights': len(self.height_cols),
            'n_widths': len(self.width_cols),
            'n_lengths': len(self.length_cols)
        }


class TraceSequenceProcessor:
    """Handles semantic parsing of trace sequence data for 3D cross-section analysis.
    
    Data format: [Layer, Type, W, H, Length, feat1, ..., featN, x_dim, z_dim]
    Where:
    - Layer: Integer layer number (categorical)
    - Type: Structure type S/G/D encoded as 0/1/2 (categorical) 
    - W, H, Length: Geometric dimensions (continuous)
    - feat1...featN: Additional local features (continuous)
    - x_dim, z_dim: Spatial coordinates (continuous, uses positional encoding)
    """
    
    # Field indices
    LAYER_IDX = 0
    TYPE_IDX = 1
    GEOM_START = 2  # W, H, Length
    GEOM_END = 5
    SPATIAL_START = -2  # x_dim, z_dim
    
    @classmethod
    def split_features(cls, seq_input: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Split sequence input into semantic components.
        
        Args:
            seq_input: Tensor of shape (B, L, F) where F is total feature dimension
            
        Returns:
            Dictionary with keys: 'layers', 'types', 'geometry', 'features', 'spatial'
        """
        return {
            'layers': seq_input[:, :, cls.LAYER_IDX],
            'types': seq_input[:, :, cls.TYPE_IDX], 
            'geometry': seq_input[:, :, cls.GEOM_START:cls.GEOM_END],  # W, H, Length
            'features': seq_input[:, :, cls.GEOM_END:cls.SPATIAL_START],  # Additional features
            'spatial': seq_input[:, :, cls.SPATIAL_START:]  # x_dim, z_dim
        }
    
    @classmethod
    def get_scalable_features(cls, seq_input: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """Get features that should be scaled (exclude categorical and spatial).
        
        Args:
            seq_input: Input sequence tensor or array
            
        Returns:
            Features that should be scaled: geometry + additional features
        """
        return seq_input[:, :, cls.GEOM_START:cls.SPATIAL_START]
    
    @classmethod
    def get_scalable_slice(cls) -> slice:
        """Get slice for scalable features."""
        return slice(cls.GEOM_START, cls.SPATIAL_START)
    
    @classmethod
    def get_categorical_features(cls, seq_input: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """Get categorical features (layer and type)."""
        return seq_input[:, :, :cls.GEOM_START]
    
    @classmethod
    def get_spatial_features(cls, seq_input: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """Get spatial coordinate features."""
        return seq_input[:, :, cls.SPATIAL_START:]
    
    @classmethod
    def reconstruct_sequence(cls, layers: torch.Tensor, types: torch.Tensor, 
                           geometry: torch.Tensor, features: torch.Tensor, 
                           spatial: torch.Tensor) -> torch.Tensor:
        """Reconstruct full sequence from semantic components."""
        return torch.cat([
            layers.unsqueeze(-1), 
            types.unsqueeze(-1), 
            geometry, 
            features, 
            spatial
        ], dim=-1)
    
    @classmethod
    def split_for_model(cls, seq_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split sequence for model forward pass (matching current trace_model.py format).
        
        Returns:
            layers, types, feats, spatials - matching current model expectations
        """
        layers = seq_input[:, :, cls.LAYER_IDX:cls.LAYER_IDX+1]  # Keep dim
        types = seq_input[:, :, cls.TYPE_IDX:cls.TYPE_IDX+1]     # Keep dim  
        feats = seq_input[:, :, cls.GEOM_START:cls.SPATIAL_START]  # All middle features
        spatials = seq_input[:, :, cls.SPATIAL_START:]            # x_dim, z_dim
        
        return layers, types, feats, spatials
    
    @classmethod 
    def get_feature_dims(cls, total_dim: int) -> Dict[str, int]:
        """Get dimensions for each feature type."""
        return {
            'layer': 1,
            'type': 1, 
            'geometry': cls.GEOM_END - cls.GEOM_START,  # 3 (W, H, Length)
            'features': total_dim - 4 - (cls.GEOM_END - cls.GEOM_START),  # Additional features
            'spatial': 2  # x_dim, z_dim
        } 