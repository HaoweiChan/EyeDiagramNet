import os
import numpy as np
import traceback

# Source 1: The "legacy" function from a path that might not exist.
# This is based on the user's request to test against a (potentially old) entry point.
try:
    from simulation.engine.sparam_to_ew import snp_eyewidth_simulation as legacy_snp_eyewidth_simulation
    LEGACY_AVAILABLE = True
    print("Successfully imported legacy `snp_eyewidth_simulation` from `simulation.engine.sparam_to_ew`.")
except ImportError:
    LEGACY_AVAILABLE = False
    print("Info: Could not import `snp_eyewidth_simulation` from `simulation.engine.sparam_to_ew`. This source will be skipped.")

# Source 2: The current EyeWidthSimulator class and its wrapper function.
try:
    from simulation.engine.eye_width_simulator import EyeWidthSimulator, snp_eyewidth_simulation as new_snp_eyewidth_simulation
    from simulation.parameters.bound_param import SampleResult
    NEW_SIMULATOR_AVAILABLE = True
    print("Successfully imported `EyeWidthSimulator` and new `snp_eyewidth_simulation`.")
except ImportError as e:
    NEW_SIMULATOR_AVAILABLE = False
    print(f"Error: Could not import current simulation modules: {e}")
    traceback.print_exc()

def main():
    """
    Main function to run comparison between different simulation sources.
    """
    print("\nStarting simulation comparison test...")
    print("="*50)

    if not NEW_SIMULATOR_AVAILABLE:
        print("Core simulation modules are not available. Aborting test.")
        return

    # Define test configuration, using paths relative to the project root.
    # Assumes the script is run from the project's root directory.
    data_dir = "test_data"
    if not os.path.isdir(data_dir):
        print(f"Error: Test data directory '{data_dir}' not found. Please run from the project root.")
        return

    # Define paths to S-parameter files for the test
    snp_horiz = os.path.join(data_dir, "tlines4_seed0.s8p")
    snp_tx = os.path.join(data_dir, "vertical", "tlines4_seed1.s8p")
    snp_rx = os.path.join(data_dir, "vertical", "tlines4_seed2.s8p")

    # Verify that all required S-parameter files exist before proceeding
    for f_path in [snp_horiz, snp_tx, snp_rx]:
        if not os.path.exists(f_path):
            print(f"Error: Required S-parameter file not found: {f_path}")
            print("Please ensure the test data exists and paths are correct.")
            return

    # Configuration dictionary for the simulation
    config_dict = {
        "R_tx": 4e1,
        "R_rx": 1.0e9,
        "C_tx": 2e-13,
        "C_rx": 2e-13,
        "L_tx": 1.6e-9,
        "L_rx": 0.,
        "pulse_amplitude": 0.9,
        "bits_per_sec": 2.6e10,
        "vmask": 0.03,
        "snp_horiz": snp_horiz,
        "snp_tx": snp_tx,
        "snp_rx": snp_rx,
        "directions": [1, 0, 1, 0]  # Example for 4 lines
    }

    # The SampleResult class is used to hold the configuration
    config = SampleResult.from_dict(config_dict)

    # --- Run Simulations ---
    results = {}

    # Run with legacy function if it was successfully imported
    if LEGACY_AVAILABLE:
        print("\n1. Running legacy `snp_eyewidth_simulation`...")
        try:
            # The legacy function expects snp_files and directions passed as separate arguments.
            snp_files = (config.snp_horiz, config.snp_tx, config.snp_rx)
            ew_legacy = legacy_snp_eyewidth_simulation(config, snp_files, config.directions)
            results['legacy'] = np.array(ew_legacy)
            print(f"  Result: {results['legacy']}")
        except Exception as e:
            print(f"  Error running legacy simulation: {e}")
            traceback.print_exc()
            results['legacy'] = None

    # Run with the current EyeWidthSimulator class
    print("\n2. Running new `EyeWidthSimulator` class instance...")
    try:
        simulator = EyeWidthSimulator(config)
        ew_new_class = simulator.calculate_eyewidth()
        results['new_class'] = np.array(ew_new_class)
        print(f"  Result: {results['new_class']}")
    except Exception as e:
        print(f"  Error running new `EyeWidthSimulator` class: {e}")
        traceback.print_exc()
        results['new_class'] = None
    
    # Run with the wrapper function from the new module
    print("\n3. Running new `snp_eyewidth_simulation` wrapper function...")
    try:
        ew_new_wrapper = new_snp_eyewidth_simulation(config)
        results['new_wrapper'] = np.array(ew_new_wrapper)
        print(f"  Result: {results['new_wrapper']}")
    except Exception as e:
        print(f"  Error running new `snp_eyewidth_simulation` wrapper: {e}")
        traceback.print_exc()
        results['new_wrapper'] = None

    # --- Compare Results ---
    print("\n\n--- COMPARISON SUMMARY ---")
    print("="*50)

    # Compare the two current methods first
    if results.get('new_class') is not None and results.get('new_wrapper') is not None:
        if np.allclose(results['new_class'], results['new_wrapper']):
            print("✅ SUCCESS: `EyeWidthSimulator` class and its wrapper function results match.")
        else:
            print("❌ FAILURE: `EyeWidthSimulator` class and its wrapper function results DO NOT match.")
            print(f"   Class Result:   {results['new_class']}")
            print(f"   Wrapper Result: {results['new_wrapper']}")
    else:
        print("⚠️ SKIPPED: Comparison between new methods due to simulation errors.")

    # Compare legacy with current method if legacy was available
    if LEGACY_AVAILABLE:
        if results.get('legacy') is not None and results.get('new_class') is not None:
            if np.allclose(results['legacy'], results['new_class']):
                print("✅ SUCCESS: Legacy and new simulation results match.")
            else:
                print("❌ FAILURE: Legacy and new simulation results DO NOT match.")
                print(f"   Legacy Result: {results['legacy']}")
                print(f"   New Result:    {results['new_class']}")
        else:
            print("⚠️ SKIPPED: Comparison with legacy method due to simulation errors.")
    else:
        print("ℹ️ INFO: Legacy simulation was not available for comparison.")

    print("\nTest finished.")


if __name__ == "__main__":
    main() 