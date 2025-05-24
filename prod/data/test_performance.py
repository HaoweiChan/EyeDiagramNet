import os
import time
import numpy as np

# Try to import from both original and refactored
try:
    # For the refactored version, we still use the same function names
    from sparam_to_ew_refactored import snp_eyewidth_simulation as refactored_simulation
    from bound_param import SampleResult
    HAS_REFACTORED = True
except ImportError as e:
    print(f"Import error for refactored version: {e}")
    HAS_REFACTORED = False

try:
    from sparam_to_ew import snp_eyewidth_simulation as original_simulation
    HAS_ORIGINAL = True
except ImportError as e:
    print(f"Import error for original version: {e}")
    HAS_ORIGINAL = False

if not (HAS_REFACTORED or HAS_ORIGINAL):
    print("Error: Could not import either implementation")
    exit(1)

def get_test_files(sandbox=False):
    """Find test S-parameter files
    
    Args:
        sandbox (bool): If True, use s96p files for formal testing. If False, use s8p files.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "../.."))
    
    # Try multiple possible locations for test data
    possible_data_dirs = [
        os.path.join(project_root, "test_data"),
        "test_data",
        "../../test_data",
        "../test_data"
    ]
    
    # Find the first valid data directory
    data_dir = None
    for test_dir in possible_data_dirs:
        if os.path.exists(test_dir) and os.path.isdir(test_dir):
            data_dir = test_dir
            break
            
    if data_dir is None:
        print("Warning: Could not find test_data directory.")
        # Create dummy files for testing
        return ("dummy.s8p", "dummy.s8p", "dummy.s8p")
    
    # Choose file extension based on sandbox flag
    file_extension = '.s96p' if sandbox else '.s8p'
    
    # Look for s-parameter files
    s_files = [f for f in os.listdir(data_dir) if f.endswith(file_extension) or (not sandbox and f.endswith('.snp'))]
    
    if len(s_files) >= 3:
        return tuple(os.path.join(data_dir, s) for s in s_files[:3])
    else:
        print(f"Warning: Not enough {file_extension} files found.")
        # Create dummy files for testing
        return ("dummy.s8p", "dummy.s8p", "dummy.s8p")

def create_test_config(sandbox=False):
    """Create test configuration
    
    Args:
        sandbox (bool): If True, use s96p files for formal testing. If False, use s8p files.
        
    Returns:
        tuple: (config_object, config_dict, snp_files)
    """
    # Get test files
    snp_file = get_test_files(sandbox)
    
    # Configuration parameters based on the working example in sparam_to_ew_refactored.py main()
    config_dict = {
        # Use the exact working parameters from sparam_to_ew_refactored.py main function
        "R_tx": 10,              # Working value from main()
        "R_rx": 1.0e9,           # Working value from main()
        "C_tx": 1e-13,           # Working value from main()
        "C_rx": 1e-13,           # Working value from main()
        "L_tx": 1e-10,           # Working value from main()
        "L_rx": 1e-10,           # Working value from main()
        "pulse_amplitude": 0.4,   # Working value from main()
        "bits_per_sec": 6.4e9,   # Working value from main()
        "vmask": 0.04,           # Working value from main()
        
        # File paths
        "snp_horiz": snp_file[0] if isinstance(snp_file, (tuple, list)) else snp_file,
        "snp_tx": snp_file[1] if isinstance(snp_file, (tuple, list)) and len(snp_file) > 1 else None,
        "snp_rx": snp_file[2] if isinstance(snp_file, (tuple, list)) and len(snp_file) > 2 else None,
        
        # Use the working directions from main() but adjust for sandbox
        "directions": [1] * (48 if sandbox else 1) + [0] * (0 if sandbox else 1) + [1] * (0 if sandbox else 1) + [0] * (0 if sandbox else 1),
        
        # CTLE parameters - disable as in the working main() example  
        "AC_gain": None,         
        "DC_gain": None,         
        "fp1": None,             
        "fp2": None              
    }
    
    # Create config object
    config = SampleResult(**config_dict)
    
    return config, config_dict, snp_file

def run_benchmark(config, config_dict, snp_file, iterations=3):
    """Run performance benchmark
    
    Args:
        config: SampleResult configuration object
        config_dict: Configuration dictionary
        snp_file: S-parameter file paths
        iterations: Number of benchmark iterations
    """
    # Run on CPU for fair comparison (remove device parameter for refactored version)
    device = 'cpu'
    
    # Test original implementation if available
    original_times = []
    if HAS_ORIGINAL:
        print("Testing original implementation...")
        for i in range(iterations):
            start_time = time.time()
            try:
                original_result, _ = original_simulation(config, snp_file, config_dict["directions"], device)
                end_time = time.time()
                original_times.append(end_time - start_time)
                print(f"  Run {i+1}/{iterations}: {original_times[-1]:.4f} seconds")
            except Exception as e:
                print(f"  Run {i+1}/{iterations} failed: {e}")
    else:
        print("Original implementation not available")
    
    # Test refactored implementation if available
    refactored_times = []
    if HAS_REFACTORED:
        print("\nTesting refactored implementation...")
        for i in range(iterations):
            start_time = time.time()
            try:
                # Refactored version uses config with all parameters included
                refactored_result, _ = refactored_simulation(config)
                end_time = time.time()
                refactored_times.append(end_time - start_time)
                print(f"  Run {i+1}/{iterations}: {refactored_times[-1]:.4f} seconds")
            except Exception as e:
                print(f"  Run {i+1}/{iterations} failed: {e}")
    else:
        print("Refactored implementation not available")
    
    # Calculate statistics
    print("\n=== Performance Results ===")
    
    if original_times:
        original_avg = np.mean(original_times)
        print(f"Original implementation: {original_avg:.4f} seconds (avg over {len(original_times)} runs)")
    else:
        print("Original implementation: No successful runs")
        original_avg = None
        
    if refactored_times:
        refactored_avg = np.mean(refactored_times)
        print(f"Refactored implementation: {refactored_avg:.4f} seconds (avg over {len(refactored_times)} runs)")
    else:
        print("Refactored implementation: No successful runs")  
        refactored_avg = None
    
    if original_avg is not None and refactored_avg is not None:
        speedup = original_avg / refactored_avg if refactored_avg > 0 else float('inf')
        print(f"Speedup: {speedup:.2f}x")
        
        if speedup > 1:
            print("\n✅ Refactored implementation is faster!")
        else:
            print("\n❌ Original implementation is still faster.")
    elif refactored_avg is not None:
        print("\n✅ Refactored implementation is working (no original for comparison)")
    else:
        print("\n❌ Neither implementation worked successfully")

def test_integrity(config, config_dict, snp_file):
    """Test that refactored implementation produces correct results
    
    Args:
        config: SampleResult configuration object
        config_dict: Configuration dictionary
        snp_file: S-parameter file paths
        
    Returns:
        bool: True if integrity test passes, False otherwise
    """
    print("\n=== Testing Implementation Integrity ===")
    
    try:
        integrity_passed = True
        
        if HAS_REFACTORED:
            print("Running refactored implementation...")
            result, _ = refactored_simulation(config)
            print(f"Refactored result shape: {result.shape if hasattr(result, 'shape') else type(result)}")
            print(f"Refactored result: {result}")
            
            # Basic sanity checks
            if hasattr(result, 'shape'):
                assert len(result) > 0, "Result should not be empty"
                # Allow small negative values (like -0.1) which indicate no eye opening
                assert all(r >= -1.0 for r in result), "Eye widths should be >= -1.0 (no eye opening)"
                assert all(r <= 100 for r in result), "Eye widths should be <= 100%"
                print("✅ Refactored implementation passes basic sanity checks")
            else:
                print("⚠️  Unexpected result type")
                integrity_passed = False
        else:
            print("⚠️  Refactored implementation not available")
            integrity_passed = False
        
        if HAS_ORIGINAL:
            print("Running original implementation for comparison...")
            original_result, _ = original_simulation(config, snp_file, config_dict["directions"], 'cpu')
            print(f"Original result: {original_result}")
            
            if HAS_REFACTORED and hasattr(result, 'shape') and hasattr(original_result, 'shape'):
                if np.allclose(result, original_result, rtol=1e-3, atol=1e-6):
                    print("✅ Results match between implementations!")
                else:
                    print("⚠️  Results differ between implementations")
                    print(f"Max difference: {np.max(np.abs(result - original_result))}")
                    integrity_passed = False
        
        if integrity_passed:
            print("✅ Integrity test PASSED")
        else:
            print("❌ Integrity test FAILED")
            
        return integrity_passed
        
    except Exception as e:
        print(f"❌ Integrity test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Parse command line arguments for sandbox flag
    import sys
    sandbox = '--sandbox' in sys.argv or '-s' in sys.argv
    
    if sandbox:
        print("🧪 Running in sandbox mode with s96p files")
    else:
        print("🔧 Running in standard mode with s8p files")
    
    # Create shared configuration
    config, config_dict, snp_file = create_test_config(sandbox)
    
    # Test integrity first
    integrity_passed = test_integrity(config, config_dict, snp_file)
    
    # Run benchmark if integrity test passes
    if integrity_passed:
        print("\n" + "="*50)
        print("Integrity test passed - proceeding with benchmark")
        print("="*50)
        run_benchmark(config, config_dict, snp_file, iterations=3)
    else:
        print("\n" + "="*50)
        print("❌ Integrity test failed - skipping benchmark")
        print("Fix implementation issues before running performance tests")
        print("="*50)
        sys.exit(1) 