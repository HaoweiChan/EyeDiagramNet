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

# Source 2: The current EyeWidthSimulator class.
try:
    from simulation.engine.sbr_simulator import EyeWidthSimulator
    from simulation.parameters.bound_param import SampleResult
    NEW_SIMULATOR_AVAILABLE = True
    print("Successfully imported `EyeWidthSimulator`.")
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
    snp_horiz = "/proj/siaiadm/AI_training_data/D2D/UCIe_Trace/pattern2/cowos_9mi/snp/UCIe_pattern2_cowos_9mi-1.s96p"
    snp_tx    = "/proj/siaiadm/ew_predictor/data/add_ind/pattern2_cowos_9mi/auto_thru_96port.s96p"
    snp_rx    = "/proj/siaiadm/ew_predictor/data/add_ind/pattern2_cowos_9mi/auto_thru_96port.s96p"

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
        "directions": [1] * 48
        # "directions": [
        #     0, 0, 0, 0, 0, 0,
        #     1, 1, 1, 1, 1, 1,
        #     1, 1, 1, 1, 1, 1,
        #     1, 1, 1, 1, 1, 1,
        #     0, 0, 0, 0, 0, 0,
        #     1, 1, 1, 1, 1, 1,
        #     0, 0, 0, 0, 0, 0,
        #     0, 0, 0, 0, 0, 0
        # ]
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
            ew_legacy, _ = legacy_snp_eyewidth_simulation(config, snp_files, config.directions, device="cpu")
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

    # --- Compare Results ---
    print("\n\n--- COMPARISON SUMMARY ---")
    print("="*50)

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