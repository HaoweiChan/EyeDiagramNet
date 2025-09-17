#!/usr/bin/env python3
"""
Sequence Visualizer for Contour Data

Visualizes 2D cross-section of sequence.csv and variation.csv data showing:
- Layer structure (Y-axis: 0 to N from top to bottom)  
- Segment layout (X-axis: left to right based on widths)
- Material types with color coding:
  - Dielectric (D): Light blue
  - Ground (G): Green  
  - Signal (S): Red
"""

import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from ml.data.contour_data import ContourProcessor
except ImportError:
    print("Error: Could not import ContourProcessor. Make sure the ml.data.contour_data module is available.")
    sys.exit(1)


class SequenceVisualizer:
    """Visualize 2D cross-section of contour sequence data."""
    
    def __init__(self):
        # Color scheme for different material types
        self.colors = {
            'D': '#ADD8E6',  # Light blue for dielectric
            'S': '#FF6B6B',  # Red for signal  
            'G': '#90EE90',  # Green for ground
        }
        
        # Type name mapping
        self.type_names = {
            'D': 'Dielectric',
            'S': 'Signal',
            'G': 'Ground'
        }
    
    def load_data(self, data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load sequence.csv and variation.csv files."""
        sequence_file = data_dir / "sequence.csv"
        variation_file = data_dir / "variation.csv"
        
        if not sequence_file.exists():
            raise FileNotFoundError(f"sequence.csv not found in {data_dir}")
        if not variation_file.exists():
            raise FileNotFoundError(f"variation.csv not found in {data_dir}")
        
        sequence_df = pd.read_csv(sequence_file)
        variation_df = pd.read_csv(variation_file)
        
        return sequence_df, variation_df
    
    def extract_segments_info(self, sequence_df: pd.DataFrame, variation_row: pd.Series) -> List[Dict]:
        """Extract segment information with resolved dimensions."""
        processor = ContourProcessor()
        
        # Parse sequence and sync with variation  
        processor._discover_geometric_features(sequence_df.columns)
        variation_df_temp = pd.DataFrame([variation_row])  # Convert Series to DataFrame
        processor._sync_variation_features(variation_df_temp)
        
        # Extract material and dimension info
        material_props = processor._extract_material_properties(variation_row)
        base_dims = processor._extract_base_dimensions(variation_row)
        
        segments_info = []
        
        for _, seg_row in sequence_df.iterrows():
            # Basic info
            segment_info = {
                'segment_id': seg_row['Segment'],
                'layer': seg_row['Layer'], 
                'type_original': seg_row['Type'],
                'type_category': seg_row['Type'][0] if isinstance(seg_row['Type'], str) else 'D'  # Extract D/S/G
            }
            
            # Calculate resolved widths (sum of all width contributions for this segment)
            total_width = 0
            for w_col in processor.width_cols:
                multiplier = seg_row.get(w_col, 0.0)
                base_key = processor.var_width_map.get(w_col)
                base_value = base_dims.get(base_key, 0.0) if base_key else 0.0
                total_width += multiplier * base_value
            
            # Calculate resolved height (sum of all height contributions for this segment) 
            total_height = 0
            for h_col in processor.height_cols:
                multiplier = seg_row.get(h_col, 0.0)
                base_key = processor.var_height_map.get(h_col)
                base_value = base_dims.get(base_key, 0.0) if base_key else 0.0
                total_height += multiplier * base_value
            
            segment_info['width'] = abs(total_width) if total_width != 0 else 1.0  # Use absolute value, min 1.0
            segment_info['height'] = abs(total_height) if total_height != 0 else 1.0  # Use absolute value, min 1.0
            
            segments_info.append(segment_info)
        
        return segments_info
    
    def organize_layers(self, segments_info: List[Dict]) -> Dict[int, List[Dict]]:
        """Organize segments by layer."""
        layers = {}
        for segment in segments_info:
            layer_id = segment['layer']
            if layer_id not in layers:
                layers[layer_id] = []
            layers[layer_id].append(segment)
        
        # Sort segments within each layer by segment_id
        for layer_id in layers:
            layers[layer_id].sort(key=lambda x: x['segment_id'])
        
        return layers
    
    def calculate_layer_dimensions(self, layers: Dict[int, List[Dict]]) -> Dict[int, Dict]:
        """Calculate dimensions for each layer."""
        layer_info = {}
        
        for layer_id, segments in layers.items():
            # Layer height is the maximum height of any segment in the layer
            max_height = max(seg['height'] for seg in segments)
            
            # Total layer width is sum of all segment widths
            total_width = sum(seg['width'] for seg in segments)
            
            layer_info[layer_id] = {
                'height': max_height,
                'total_width': total_width,
                'segments': segments
            }
        
        return layer_info
    
    def visualize_sequence(self, data_dir: Path, case_id: int = 0, 
                          figsize: Tuple[float, float] = (12, 8),
                          save_path: Optional[Path] = None,
                          show_labels: bool = True) -> plt.Figure:
        """Create 2D visualization of the sequence."""
        
        # Load data
        sequence_df, variation_df = self.load_data(data_dir)
        
        if case_id >= len(variation_df):
            raise ValueError(f"Case {case_id} not found. Available cases: 0-{len(variation_df)-1}")
        
        variation_row = variation_df.iloc[case_id]
        
        # Extract segment information
        segments_info = self.extract_segments_info(sequence_df, variation_row)
        
        # Organize by layers
        layers = self.organize_layers(segments_info)
        layer_info = self.calculate_layer_dimensions(layers)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Calculate total height and max width for layout
        total_height = sum(info['height'] for info in layer_info.values())
        max_width = max(info['total_width'] for info in layer_info.values()) if layer_info else 10
        
        # Draw layers from top to bottom (layer 0 at top)
        current_y = total_height
        
        sorted_layers = sorted(layer_info.items())  # Sort by layer ID
        
        for layer_id, info in sorted_layers:
            layer_height = info['height']
            current_y -= layer_height  # Move to bottom of current layer
            
            # Draw segments in this layer from left to right
            current_x = 0
            
            for segment in info['segments']:
                width = segment['width']
                height = layer_height  # All segments in layer have same height
                
                # Get color based on type
                type_category = segment['type_category']
                color = self.colors.get(type_category, '#CCCCCC')  # Default to gray
                
                # Create rectangle
                rect = patches.Rectangle(
                    (current_x, current_y), width, height,
                    linewidth=1, edgecolor='black', facecolor=color,
                    alpha=0.8
                )
                ax.add_patch(rect)
                
                # Add labels if requested
                if show_labels:
                    # Segment type and ID
                    label = f"{segment['type_original']}\n#{segment['segment_id']}"
                    ax.text(
                        current_x + width/2, current_y + height/2,
                        label, ha='center', va='center',
                        fontsize=8, fontweight='bold'
                    )
                
                current_x += width
        
        # Set axis properties
        ax.set_xlim(0, max_width * 1.1)
        ax.set_ylim(0, total_height * 1.1)
        ax.set_xlabel('Width', fontsize=12)
        ax.set_ylabel('Height (Layer 0 at top)', fontsize=12)
        ax.set_aspect('equal')
        
        # Add layer labels on the right
        current_y = total_height
        for layer_id, info in sorted_layers:
            layer_height = info['height']
            current_y -= layer_height
            ax.text(
                max_width * 1.02, current_y + layer_height/2,
                f'Layer {layer_id}', ha='left', va='center',
                fontsize=10, fontweight='bold'
            )
        
        # Add title
        case_info = f"Case {case_id}"
        title = f"Sequence Cross-Section - {case_info}\nData: {data_dir.name}"
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Create legend
        legend_elements = []
        for type_key, color in self.colors.items():
            legend_elements.append(
                patches.Patch(color=color, label=self.type_names[type_key])
            )
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        return fig
    
    def visualize_all_cases(self, data_dir: Path, save_dir: Optional[Path] = None) -> List[plt.Figure]:
        """Visualize all cases in the dataset."""
        sequence_df, variation_df = self.load_data(data_dir)
        
        figures = []
        
        for case_id in range(len(variation_df)):
            print(f"Generating visualization for case {case_id}...")
            
            save_path = None
            if save_dir:
                save_dir = Path(save_dir)
                save_dir.mkdir(parents=True, exist_ok=True)
                save_path = save_dir / f"{data_dir.name}_case_{case_id}.png"
            
            fig = self.visualize_sequence(
                data_dir, case_id=case_id, 
                save_path=save_path, show_labels=True
            )
            figures.append(fig)
        
        return figures

    def demo_visualizations(self, base_dir: Optional[Path] = None, save_dir: Optional[Path] = None) -> None:
        """Generate demo visualizations from available test contour data."""
        if base_dir is None:
            # Default to project's contour test data
            project_root = Path(__file__).parent.parent.parent
            base_dir = project_root / "tests" / "data_generation" / "contour"
        
        if not base_dir.exists():
            print(f"ERROR: Base directory {base_dir} not found")
            return
        
        print("Sequence Visualizer Demo")
        print("=" * 50)
        
        # Find all contour test directories
        test_dirs = [d for d in base_dir.iterdir() 
                    if d.is_dir() and (d / "sequence.csv").exists() and (d / "variation.csv").exists()]
        
        if not test_dirs:
            print(f"WARNING: No contour test data found in {base_dir}")
            print("TIP: Generate test data with: python tests/data_generation/main.py contour --name demo_data")
            return
        
        total_visualizations = 0
        
        for test_dir in sorted(test_dirs):
            print(f"\nProcessing: {test_dir.name}")
            
            try:
                # Load data to check number of cases
                sequence_df, variation_df = self.load_data(test_dir)
                n_cases = len(variation_df)
                n_segments = len(sequence_df)
                
                print(f"   {n_segments} segments, {n_cases} cases")
                
                # Visualize first case
                case_id = 0
                print(f"   Visualizing case {case_id}...")
                
                # Determine output path
                if save_dir:
                    save_dir_path = Path(save_dir)
                    save_dir_path.mkdir(parents=True, exist_ok=True)
                    output_file = save_dir_path / f"demo_{test_dir.name}_case_{case_id}.png"
                else:
                    output_file = f"demo_{test_dir.name}_case_{case_id}.png"
                
                fig = self.visualize_sequence(
                    test_dir, 
                    case_id=case_id,
                    figsize=(10, 6),
                    save_path=output_file,
                    show_labels=True
                )
                
                plt.close(fig)  # Close to save memory
                print(f"   Saved: {output_file}")
                total_visualizations += 1
                
            except Exception as e:
                print(f"   ERROR: {e}")
        
        if total_visualizations > 0:
            print(f"\nDemo complete! Generated {total_visualizations} visualizations.")
            self._print_usage_examples()
        else:
            print(f"\nNo visualizations generated.")

    def _print_usage_examples(self) -> None:
        """Print usage examples."""
        print("\nUsage examples:")
        print("   # Single case visualization:")
        print("   python tests/data_analyzer/sequence_visualizer.py tests/data_generation/contour/small_contour --case 0 --save output.png")
        print("\n   # All cases:")
        print("   python tests/data_analyzer/sequence_visualizer.py tests/data_generation/contour/small_contour --all-cases --save-dir output/")
        print("\n   # Interactive (no save):")
        print("   python tests/data_analyzer/sequence_visualizer.py tests/data_generation/contour/small_contour --case 0")
        print("\n   # Generate demo visualizations:")
        print("   python tests/data_analyzer/sequence_visualizer.py --demo --save-dir demo_output/")


def main():
    """Command line interface."""
    parser = argparse.ArgumentParser(
        description="Visualize 2D cross-section of contour sequence data"
    )
    parser.add_argument(
        "data_dir", type=str, nargs='?', default=None,
        help="Directory containing sequence.csv and variation.csv"
    )
    parser.add_argument(
        "-c", "--case", type=int, default=0,
        help="Case ID to visualize (default: 0)"
    )
    parser.add_argument(
        "--all-cases", action="store_true",
        help="Generate visualizations for all cases"
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Generate demo visualizations from all available test contour data"
    )
    parser.add_argument(
        "-s", "--save", type=str, default=None,
        help="Save visualization to file"
    )
    parser.add_argument(
        "--save-dir", type=str, default=None,
        help="Directory to save visualizations"
    )
    parser.add_argument(
        "--no-labels", action="store_true",
        help="Hide segment labels"
    )
    parser.add_argument(
        "--figsize", nargs=2, type=float, default=[12, 8],
        help="Figure size in inches (width height)"
    )
    
    args = parser.parse_args()
    
    visualizer = SequenceVisualizer()
    
    try:
        if args.demo:
            # Demo mode - process all available test data
            save_dir = Path(args.save_dir) if args.save_dir else None
            visualizer.demo_visualizations(save_dir=save_dir)
        else:
            # Regular mode - need data_dir
            if not args.data_dir:
                print("Error: data_dir is required when not using --demo mode")
                parser.print_help()
                sys.exit(1)
            
            data_dir = Path(args.data_dir)
            if not data_dir.exists():
                print(f"Error: Directory {data_dir} does not exist")
                sys.exit(1)
            
            if args.all_cases:
                save_dir = Path(args.save_dir) if args.save_dir else None
                figures = visualizer.visualize_all_cases(data_dir, save_dir=save_dir)
                print(f"Generated {len(figures)} visualizations")
                
                if not args.save_dir:
                    plt.show()  # Show all figures
            else:
                save_path = Path(args.save) if args.save else None
                fig = visualizer.visualize_sequence(
                    data_dir, case_id=args.case,
                    figsize=tuple(args.figsize),
                    save_path=save_path,
                    show_labels=not args.no_labels
                )
                
                if not args.save:
                    plt.show()
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
