"""Contour data generator for synthetic sequence.csv and variation.csv files."""

import numpy as np
import pandas as pd
import random
from pathlib import Path
from typing import List, Dict, Tuple


class ContourDataGenerator:
    """Generate synthetic sequence.csv and variation.csv files for testing."""
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        random.seed(seed)
        
        # Define possible dielectric types
        self.dielectric_types = ['D_1', 'D_2', 'D_3', 'D_4', 'D_5', 'D_6']
        self.metal_types = ['S_1', 'G_1']
        
        # Define height, width, and length feature names
        self.height_features = ['H_h_ubm', 'H_h_cbd', 'H_h_via', 'H_h_sig2', 'H_h_ild', 'H_h_psub']
        self.width_features = ['W_wgr', 'W_wg', 'W_ws', 'W_c']
        self.length_features = ['L_1']
        
        # Corresponding variation file features
        self.var_height_features = ['H_ubm', 'H_cbd', 'H_via', 'H_sig2', 'H_lid', 'H_psut']
        self.var_width_features = ['W_wg1', 'W_wg2', 'W_ws', 'W_c']
        self.var_length_features = ['L_l']

    def generate_layer_structure(self, min_segments: int = 10, max_segments: int = 50) -> List[Dict]:
        """Generate a layer structure following the rule: metals must be wrapped by dielectrics."""
        n_segments = random.randint(min_segments, max_segments)
        
        segments = []
        current_layer = 0
        segment_id = 0
        
        # Start with a dielectric base layer
        segments.append({
            'Segment': segment_id,
            'Layer': current_layer,
            'Type': random.choice(self.dielectric_types)
        })
        segment_id += 1
        current_layer += 1
        
        # Generate middle layers with metal-dielectric structure
        while segment_id < n_segments - 1:  # Reserve last segment for dielectric cap
            # Decide if this will be a metal layer or dielectric layer
            if random.random() < 0.3:  # 30% chance for metal layer
                # Create a metal layer with multiple segments
                metal_segments_in_layer = random.randint(2, 5)
                
                for i in range(metal_segments_in_layer):
                    if segment_id >= n_segments - 1:
                        break
                    
                    # Alternate between signal and ground in metal layers
                    metal_type = 'S_1' if i % 2 == 0 else 'G_1'
                    segments.append({
                        'Segment': segment_id,
                        'Layer': current_layer,
                        'Type': metal_type
                    })
                    segment_id += 1
                
                current_layer += 1
                
                # Add dielectric layer after metal (wrapping rule)
                if segment_id < n_segments - 1:
                    segments.append({
                        'Segment': segment_id,
                        'Layer': current_layer,
                        'Type': random.choice(self.dielectric_types)
                    })
                    segment_id += 1
                    current_layer += 1
            else:
                # Add single dielectric segment
                segments.append({
                    'Segment': segment_id,
                    'Layer': current_layer,
                    'Type': random.choice(self.dielectric_types)
                })
                segment_id += 1
                current_layer += 1
        
        # End with a dielectric cap layer
        if segment_id < n_segments:
            segments.append({
                'Segment': segment_id,
                'Layer': current_layer,
                'Type': random.choice(self.dielectric_types)
            })
        
        return segments

    def add_geometric_features(self, segments: List[Dict]) -> List[Dict]:
        """Add geometric multiplier features to segments."""
        # First, group segments by layer to ensure same-layer segments have same heights
        layers = {}
        for segment in segments:
            layer_id = segment['Layer']
            if layer_id not in layers:
                layers[layer_id] = []
            layers[layer_id].append(segment)
        
        # Generate height values per layer (all segments in same layer get same heights)
        layer_heights = {}
        for layer_id in layers.keys():
            layer_heights[layer_id] = {}
            for h_feat in self.height_features:
                if random.random() < 0.7:  # 70% chance to have non-zero value
                    # Generate multiples of 0.5, ensuring positive final height
                    multiplier = self._generate_multiple_of_half(min_val=0.5, max_val=2.0, allow_negative=False)
                    layer_heights[layer_id][h_feat] = multiplier
                else:
                    layer_heights[layer_id][h_feat] = 0.0
        
        # Apply features to all segments
        for segment in segments:
            layer_id = segment['Layer']
            
            # Add height multipliers (same for all segments in the same layer)
            for h_feat in self.height_features:
                segment[h_feat] = layer_heights[layer_id][h_feat]
            
            # Add width multipliers (can vary per segment)
            segment_width_multipliers = {}
            for w_feat in self.width_features:
                if random.random() < 0.6:  # 60% chance to have non-zero value
                    # Generate multiples of 0.5, can be negative
                    multiplier = self._generate_multiple_of_half(min_val=-1.0, max_val=2.0, allow_negative=True)
                    segment_width_multipliers[w_feat] = multiplier
                else:
                    segment_width_multipliers[w_feat] = 0.0
            
            # Ensure total width will be positive
            segment_width_multipliers = self._ensure_positive_total(segment_width_multipliers, min_positive=0.5)
            for w_feat, multiplier in segment_width_multipliers.items():
                segment[w_feat] = multiplier
            
            # Add length multipliers (usually 1 for most segments)
            for l_feat in self.length_features:
                if random.random() < 0.9:
                    # Generate positive multiples of 0.5 for length
                    segment[l_feat] = self._generate_multiple_of_half(min_val=0.5, max_val=2.0, allow_negative=False)
                else:
                    segment[l_feat] = 0.0
        
        return segments

    def _generate_multiple_of_half(self, min_val: float, max_val: float, allow_negative: bool = True) -> float:
        """Generate a random multiple of 0.5 within the given range."""
        # Convert to multiples of 0.5
        min_multiple = int(min_val / 0.5)
        max_multiple = int(max_val / 0.5)
        
        # Ensure we don't go below 0 if negative values are not allowed
        if not allow_negative:
            min_multiple = max(min_multiple, 1)  # At least 0.5
        
        # Generate random multiple and convert back to float
        multiple = random.randint(min_multiple, max_multiple)
        return multiple * 0.5

    def _ensure_positive_total(self, multipliers: Dict[str, float], min_positive: float = 0.5) -> Dict[str, float]:
        """Ensure that the sum of multipliers is positive by adjusting if needed."""
        total = sum(multipliers.values())
        
        # If total is already positive and meets minimum, return as-is
        if total >= min_positive:
            return multipliers
        
        # If total is negative or too small, add positive value to one random multiplier
        adjustment_needed = min_positive - total
        
        # Choose a random feature to adjust
        features = list(multipliers.keys())
        if features:
            chosen_feature = random.choice(features)
            
            # Add the adjustment in multiples of 0.5
            adjustment_multiples = int(np.ceil(adjustment_needed / 0.5))
            multipliers[chosen_feature] += adjustment_multiples * 0.5
        
        return multipliers

    def generate_sequence_csv(self, output_path: Path, min_segments: int = 10, max_segments: int = 50) -> List[Dict]:
        """Generate sequence.csv file."""
        segments = self.generate_layer_structure(min_segments, max_segments)
        segments = self.add_geometric_features(segments)
        
        # Convert to DataFrame and save
        df = pd.DataFrame(segments)
        
        # Reorder columns to match expected format
        cols = ['Segment', 'Layer', 'Type'] + self.height_features + self.width_features + self.length_features
        df = df.reindex(columns=cols)
        
        df.to_csv(output_path, index=False)
        print(f"Generated sequence.csv with {len(segments)} segments at {output_path}")
        
        return segments

    def get_used_dielectric_types(self, segments: List[Dict]) -> List[str]:
        """Extract unique dielectric types used in the sequence."""
        dielectric_types_used = set()
        for segment in segments:
            if segment['Type'].startswith('D_'):
                dielectric_types_used.add(segment['Type'])
        return sorted(list(dielectric_types_used))

    def generate_variation_csv(self, output_path: Path, segments: List[Dict], min_cases: int = 3, max_cases: int = 15) -> pd.DataFrame:
        """Generate variation.csv file based on the sequence structure."""
        n_cases = random.randint(min_cases, max_cases)
        
        # Get dielectric types used in sequence
        dielectric_types_used = self.get_used_dielectric_types(segments)
        
        variations = []
        
        for case_id in range(n_cases):
            case_data = {'Case': case_id}
            
            # Add metal conductivity
            case_data['M_1_cond'] = round(random.uniform(10e6, 60e6), 0)  # S/m
            
            # Add dielectric properties for all used types
            for d_type in dielectric_types_used:
                d_num = d_type.split('_')[1]  # Extract number from 'D_1' -> '1'
                case_data[f'D_{d_num}_dk'] = round(random.uniform(2.0, 12.0), 2)  # Dielectric constant
                case_data[f'D_{d_num}_df'] = round(random.uniform(0.005, 0.05), 3)  # Loss factor
                case_data[f'D_{d_num}_cond'] = round(random.uniform(1e-6, 1e-3), 6)  # Conductivity
            
            # Add base dimensions - heights
            for var_h_feat in self.var_height_features:
                case_data[var_h_feat] = round(random.uniform(0.1, 50.0), 2)  # μm
            
            # Add base dimensions - widths  
            for var_w_feat in self.var_width_features:
                case_data[var_w_feat] = round(random.uniform(0.5, 20.0), 1)  # μm
            
            # Add base dimensions - lengths
            for var_l_feat in self.var_length_features:
                case_data[var_l_feat] = round(random.uniform(100.0, 5000.0), 0)  # μm
            
            variations.append(case_data)
        
        # Convert to DataFrame and save
        df = pd.DataFrame(variations)
        
        # Reorder columns for better organization
        basic_cols = ['Case', 'M_1_cond']
        dielectric_cols = []
        for d_type in dielectric_types_used:
            d_num = d_type.split('_')[1]
            dielectric_cols.extend([f'D_{d_num}_dk', f'D_{d_num}_df', f'D_{d_num}_cond'])
        
        dimension_cols = self.var_height_features + self.var_width_features + self.var_length_features
        
        all_cols = basic_cols + dielectric_cols + dimension_cols
        df = df.reindex(columns=all_cols)
        
        df.to_csv(output_path, index=False)
        print(f"Generated variation.csv with {n_cases} cases and dielectric types {dielectric_types_used} at {output_path}")
        
        return df

    def generate_dataset(self, output_dir: Path, dataset_name: str = "contour_test", 
                        min_segments: int = 10, max_segments: int = 50,
                        min_cases: int = 3, max_cases: int = 15) -> Tuple[List[Dict], pd.DataFrame]:
        """Generate a complete dataset with sequence.csv and variation.csv."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate sequence.csv
        sequence_path = output_dir / "sequence.csv"
        segments = self.generate_sequence_csv(sequence_path, min_segments, max_segments)
        
        # Generate variation.csv based on sequence
        variation_path = output_dir / "variation.csv"
        variations = self.generate_variation_csv(variation_path, segments, min_cases, max_cases)
        
        print(f"\nGenerated complete dataset '{dataset_name}':")
        print(f"  Segments: {len(segments)}")
        print(f"  Cases: {len(variations)}")
        print(f"  Output directory: {output_dir}")
        
        return segments, variations


def generate_contour_dataset(output_dir: Path, dataset_name: str = "test_contour",
                           min_segments: int = 10, max_segments: int = 50,
                           min_cases: int = 3, max_cases: int = 15, 
                           seed: int = 42) -> None:
    """Public interface function for generating contour datasets."""
    generator = ContourDataGenerator(seed=seed)
    generator.generate_dataset(
        output_dir=output_dir,
        dataset_name=dataset_name,
        min_segments=min_segments,
        max_segments=max_segments,
        min_cases=min_cases,
        max_cases=max_cases
    )
