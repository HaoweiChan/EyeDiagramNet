"""
Test ContourDataModule to verify it loads data correctly.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.data.contour_data import ContourDataModule

def test_contour_datamodule():
    """Test that ContourDataModule can load test data correctly."""
    print("=" * 80)
    print("Testing ContourDataModule with test data")
    print("=" * 80)
    
    # Create datamodule
    dm = ContourDataModule(
        data_dirs={'test': 'tests/data_generation/contour'},
        label_dir='tests/data_generation/contour',
        batch_size=32,
        test_size=0.2
    )
    
    # Setup
    print("\n1. Setting up datamodule...")
    try:
        dm.setup('fit')
        print(f"   ✓ Setup successful")
    except Exception as e:
        print(f"   ✗ Setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Check datasets
    print("\n2. Checking datasets...")
    print(f"   Train dataset: {len(dm.train_dataset)} samples")
    print(f"   Val dataset: {len(dm.val_dataset)} samples")
    
    # Check variable registry
    print("\n3. Checking variable registry...")
    print(f"   Registered variables: {len(dm.registry.variable_names)}")
    print(f"   Variables: {dm.registry.variable_names}")
    
    # Get a sample
    if len(dm.train_dataset) > 0:
        print("\n4. Getting sample data...")
        sample = dm.train_dataset[0]
        print(f"   Variables keys: {sample['variables'].keys()}")
        print(f"   Sequence shape: {sample['sequence'].shape}")
        print(f"   Target shape: {sample['targets'].shape if hasattr(sample['targets'], 'shape') else type(sample['targets'])}")
    
    print("\n" + "=" * 80)
    print(f"Test {'PASSED' if len(dm.train_dataset) > 0 else 'FAILED'}")
    print("=" * 80)
    
    return len(dm.train_dataset) > 0

if __name__ == '__main__':
    success = test_contour_datamodule()
    sys.exit(0 if success else 1)

