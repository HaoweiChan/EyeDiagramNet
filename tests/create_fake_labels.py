"""
Create fake label pickle files for testing ContourDataModule.
"""

import pickle
import numpy as np
from pathlib import Path

def create_fake_labels():
    """Create fake label pickle files matching the test data."""
    # Create output directory
    output_dir = Path('tests/data_generation/contour/test')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # We have 8 cases (0-7) from test_variations_variable.csv
    for case_id in range(8):
        # Create fake data for this case
        # Each case has 1 sample with 1 boundary condition
        data = {
            'configs': [[50.0, 0.3]],  # Fake boundary parameters (R, C)
            'line_ews': [[0.5, 0.6, 0.55]],  # Fake eye widths for 3 lines
            'snp_drvs': [f'fake_drv_{case_id}.s4p'],
            'snp_odts': [f'fake_odt_{case_id}.s4p'],
            'directions': [[1, 1, 1]],  # All forward direction
            'meta': {
                'config_keys': ['R_drv', 'C_pkg'],
                'snp_horiz': f'fake_horiz_{case_id}.s16p',
                'n_ports': 16,
                'param_types': ['boundary', 'boundary']
            }
        }
        
        # Save to pickle file named with case ID
        pkl_file = output_dir / f'case_{case_id}.pkl'
        with open(pkl_file, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Created: {pkl_file}")
    
    print(f"\nCreated {8} fake label files in {output_dir}")

if __name__ == '__main__':
    create_fake_labels()

