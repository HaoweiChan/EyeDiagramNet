"""
High-performance sequential data collector that avoids multiprocessing overhead.

This collector is fully compatible with the same configuration files used by
parallel_collector.py. It uses the same YAML configuration structure:
- dataset: horizontal_dataset and vertical_dataset settings
- data: trace_pattern and output_dir settings  
- boundary: parameter types and sampling settings
- runner: batch_size (other runner settings are ignored for sequential processing)

This collector uses:
1. Single-process execution to eliminate process overhead and resource contention
2. Efficient memory management with in-memory caching
3. Vectorized operations where possible
4. Optimal BLAS thread usage for single-process execution
5. Progress monitoring and interruption handling
"""

import os
import sys
import time
import signal
import pickle
import psutil
import argparse
import threading
import traceback
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Optimize BLAS threads for single-process execution - use all available cores
def optimize_blas_for_sequential(num_threads: int = None):
    """Set optimal BLAS thread count for sequential processing - use all cores"""
    
    if num_threads and num_threads > 0:
        optimal_threads = num_threads
        print(f"Using user-defined BLAS threads: {optimal_threads}")
    else:
        cpu_count = psutil.cpu_count()
        # Use all available cores for maximum performance
        optimal_threads = cpu_count
        print(f"Using all available cores: {optimal_threads} BLAS threads")
    
    # Set all BLAS library thread counts
    os.environ["OMP_NUM_THREADS"] = str(optimal_threads)
    os.environ["MKL_NUM_THREADS"] = str(optimal_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(optimal_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(optimal_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(optimal_threads)
    os.environ["NUMBA_NUM_THREADS"] = str(optimal_threads)
    
    # Additional environment variables for optimal performance
    os.environ["MKL_DYNAMIC"] = "FALSE"  # Disable dynamic adjustment
    os.environ["OMP_DYNAMIC"] = "FALSE"  # Disable dynamic adjustment
    
    return optimal_threads

# BLAS optimization will be called in main() after parsing args
# This allows us to use the centralized argparser

# Now import numerical libraries with optimized settings
from common.signal_utils import read_snp
from common.pickle_utils import DataWriter, SimulationResult
from simulation.parameters.bound_param import PARAM_SETS_MAP
from simulation.io.config_utils import load_config, resolve_trace_pattern, resolve_vertical_dirs, build_argparser
from simulation.io.snp_utils import parse_snps, generate_vertical_snp_pairs
from simulation.io.direction_utils import generate_directions
from simulation.parameters.param_utils import parse_param_types, modify_params_for_inductance

# Import performance monitoring
try:
    from simulation.engine.network_utils import print_performance_summary, reset_performance_data
    NETWORK_PROFILING_AVAILABLE = True
except ImportError:
    NETWORK_PROFILING_AVAILABLE = False
    def print_performance_summary(worker_id=None): pass
    def reset_performance_data(): pass

# Global state for interruption handling
_shutdown_event = threading.Event()

class SNPCache:
    """Memory-efficient SNP cache optimized for sequential processing"""
    
    def __init__(self):
        self.cache: Dict[str, Any] = {}
        self.cache_stats = {"hits": 0, "misses": 0, "memory_mb": 0}
        
    def get_snp(self, snp_path: Path) -> Any:
        """Get SNP from cache or load from disk"""
        snp_key = str(snp_path)
        
        if snp_key in self.cache:
            self.cache_stats["hits"] += 1
            return self.cache[snp_key]
        
        # Load from disk and cache
        self.cache_stats["misses"] += 1
        network = read_snp(snp_path)
        self.cache[snp_key] = network
        
        # Update memory usage estimate
        s_data_mb = network.s.nbytes / (1024 * 1024)
        f_data_mb = network.f.nbytes / (1024 * 1024)
        self.cache_stats["memory_mb"] += s_data_mb + f_data_mb
        
        return network
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_rate = (self.cache_stats["hits"] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "cached_files": len(self.cache),
            "total_requests": total_requests,
            "hit_rate_percent": hit_rate,
            "memory_usage_mb": self.cache_stats["memory_mb"]
        }
    
    def clear(self):
        """Clear the cache"""
        self.cache.clear()
        self.cache_stats = {"hits": 0, "misses": 0, "memory_mb": 0}

class SequentialCollector:
    """High-performance sequential data collector using all cores"""
    
    def __init__(self, config: Dict[str, Any], debug: bool = False, fixed_configs: List[Any] = None):
        self.config = config
        self.debug = debug
        self.snp_cache = SNPCache()
        self.fixed_configs = fixed_configs  # Pre-sampled configs if using --fixed-config
        self.fixed_config_index = 0  # Track which config to use next
        self.stats = {
            "total_simulations": 0,
            "completed_simulations": 0,
            "failed_simulations": 0,
            "duplicate_configs_skipped": 0,
            "files_skipped_exhausted": 0,
            "start_time": None,
            "end_time": None
        }
        
        # Performance optimization settings - use larger batches for all-core processing
        self.batch_size = config.get('runner', {}).get('batch_size', 50)  # Larger batches for all-core processing
        
        # Configuration for duplicate checking
        self.check_duplicates = config.get('boundary', {}).get('check_duplicates', True)
        self.max_duplicate_attempts = config.get('boundary', {}).get('max_duplicate_attempts', 100)
        
        # Register signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """Handle interruption signals"""
        print(f"\n[INTERRUPT] Received signal {signum}, shutting down gracefully...")
        _shutdown_event.set()

    def _load_existing_configs(self, pickle_file: Path) -> set:
        """Load existing configuration tuples from a pickle file for duplication checking."""
        existing_configs = set()
        
        if not pickle_file.exists():
            return existing_configs
        
        try:
            # Use the standardized loader to get SimulationResult objects
            from common.pickle_utils import load_pickle_data
            results = load_pickle_data(pickle_file)
            
            for result in results:
                # Create a tuple of config values for hashing/comparison
                config_tuple = tuple(result.config_values)
                existing_configs.add(config_tuple)
                
        except Exception as e:
            if self.debug:
                print(f"Warning: Could not load existing configs from {pickle_file.name}: {e}")
        
        return existing_configs

    def _is_config_duplicate(self, config, existing_configs: set) -> bool:
        """Check if a configuration is a duplicate of existing ones."""
        if not existing_configs:
            return False
        
        # Convert config to tuple for comparison
        config_values, _ = config.to_list(return_keys=True)
        config_tuple = tuple(config_values)
        
        return config_tuple in existing_configs

    def _get_next_config(self, combined_params):
        """
        Get the next configuration to use for simulation.
        
        If using fixed configs, returns the next one from the pre-sampled list.
        Otherwise, samples randomly from the parameter space.
        
        Returns:
            config: Configuration object
            None: If no more configs available (fixed mode) or parameter space exhausted
        """
        if self.fixed_configs:
            # Use pre-sampled fixed configurations
            if self.fixed_config_index >= len(self.fixed_configs):
                print(f"[FIXED CONFIG] Exhausted all {len(self.fixed_configs)} pre-sampled configs")
                return None
            
            config = self.fixed_configs[self.fixed_config_index]
            self.fixed_config_index += 1
            
            if self.debug and self.fixed_config_index % 10 == 0:
                print(f"[FIXED CONFIG] Using config {self.fixed_config_index}/{len(self.fixed_configs)}")
            
            return config
        else:
            # Random sampling (original behavior)
            return combined_params.sample()
    
    def _sample_unique_config(self, combined_params, existing_configs: set):
        """
        Sample a configuration that doesn't already exist in the data.
        
        Returns:
            config: A unique configuration object
            None: If parameter space is exhausted (should skip collection)
        """
        # If using fixed configs, just get the next one (no duplicate checking needed)
        if self.fixed_configs:
            return self._get_next_config(combined_params)
        
        if not self.check_duplicates:
            return combined_params.sample()
        
        max_attempts = self.max_duplicate_attempts
        
        for attempt in range(max_attempts):
            config = combined_params.sample()
            
            if not self._is_config_duplicate(config, existing_configs):
                return config
            else:
                self.stats["duplicate_configs_skipped"] += 1
                if self.debug and attempt % 20 == 0:  # Log every 20 attempts to avoid spam
                    print(f"[DUPLICATE] Attempt {attempt+1}: Skipping duplicate configuration")
        
        # If we can't find a unique config after max_attempts, signal to skip this file
        print(f"[PARAMETER SPACE EXHAUSTED] Could not find unique configuration after {max_attempts} attempts")
        print(f"[SKIP FILE] Parameter space appears to be exhausted - stopping collection for this file")
        
        return None
    
    def _format_error_metadata(self, trace_snp, snp_drv_path, snp_odt_path, combined_config, 
                              sim_directions, sample_idx, samples_needed, error_msg):
        """
        Format comprehensive error metadata for debugging purposes (SEQUENTIAL version).
        
        Args:
            trace_snp: Path to horizontal trace SNP file
            snp_drv_path: Path to DRV vertical SNP file  
            snp_odt_path: Path to ODT vertical SNP file
            combined_config: Parameter configuration that caused the error
            sim_directions: Directions array used in simulation
            sample_idx: Current sample index (0-based)
            samples_needed: Total number of samples needed
            error_msg: The actual error message
        
        Returns:
            Formatted error metadata string
        """
        try:
            # Get config values and keys for display
            config_values, config_keys = combined_config.to_list(return_keys=True)
            config_dict = dict(zip(config_keys, config_values))
            
            # Format the comprehensive error report
            error_report = f"""
=== SEQUENTIAL SIMULATION ERROR METADATA ===
Error Message: {error_msg}
Trace File: {trace_snp.name} (Full path: {trace_snp})
Vertical DRV: {Path(snp_drv_path).name} (Full path: {snp_drv_path})
Vertical ODT: {Path(snp_odt_path).name} (Full path: {snp_odt_path})
Sample: {sample_idx + 1}/{samples_needed}
Directions: {sim_directions.tolist() if hasattr(sim_directions, 'tolist') else sim_directions}
Number of Lines: {len(sim_directions) if sim_directions is not None else 'Unknown'}

Parameter Configuration:
"""
            
            # Add parameter details in organized groups
            electrical_params = ['R_drv', 'R_odt', 'C_drv', 'C_odt', 'L_drv', 'L_odt']
            signal_params = ['pulse_amplitude', 'bits_per_sec', 'vmask']
            ctle_params = ['DC_gain', 'AC_gain', 'fp1', 'fp2']
            
            for group_name, param_list in [('Electrical', electrical_params), 
                                         ('Signal', signal_params), 
                                         ('CTLE', ctle_params)]:
                error_report += f"  {group_name} Parameters:\n"
                for param in param_list:
                    if param in config_dict:
                        value = config_dict[param]
                        if isinstance(value, float):
                            error_report += f"    {param}: {value:.6e}\n"
                        else:
                            error_report += f"    {param}: {value}\n"
            
            # Add any remaining parameters
            remaining_params = set(config_dict.keys()) - set(electrical_params + signal_params + ctle_params)
            if remaining_params:
                error_report += "  Other Parameters:\n"
                for param in sorted(remaining_params):
                    value = config_dict[param]
                    if isinstance(value, float):
                        error_report += f"    {param}: {value:.6e}\n"
                    else:
                        error_report += f"    {param}: {value}\n"
            
            error_report += "============================================="
            
            return error_report
            
        except Exception as meta_error:
            # Fallback if metadata formatting fails
            return f"""
=== SEQUENTIAL SIMULATION ERROR (METADATA FORMATTING FAILED) ===
Error Message: {error_msg}
Trace File: {trace_snp}
Vertical DRV: {snp_drv_path}
Vertical ODT: {snp_odt_path}
Sample: {sample_idx + 1}/{samples_needed}
Metadata Error: {meta_error}
================================================================
"""
    
    def collect_data(self, trace_pattern_key: str, trace_pattern: str, vertical_dirs: List[str], 
                    output_dir: Path, param_types: List[str], max_samples: int, 
                    enable_direction: bool = False, enable_inductance: bool = False, shuffle: bool = False,
                    simulator_type: str = 'sbr', use_optimized: bool = False, block_size: int = None, 
                    fixed_config: bool = False) -> Dict[str, Any]:
        """
        Collect eye width simulation data using optimized sequential processing
        
        Args:
            trace_pattern_key: Key for trace pattern
            trace_pattern: Pattern for trace SNP files
            vertical_dirs: List of vertical SNP directories
            output_dir: Output directory for results
            param_types: List of parameter types to use
            max_samples: Maximum samples per trace file
            enable_direction: Whether to use random directions
            enable_inductance: Whether to enable inductance modifications
            shuffle: Whether to shuffle the work items before processing
            simulator_type: The type of simulator to use ('sbr' or 'der')
            use_optimized: Whether to use optimized SBR simulation functions.
            block_size: Optional fixed block size for direction generation.
        
        Returns:
            Collection statistics and results
        """
        self.stats["start_time"] = time.time()
        
        print(f"Starting optimized sequential data collection...")
        print(f"Configuration:")
        print(f"  Trace pattern: {trace_pattern_key}")
        print(f"  Parameter types: {param_types}")
        print(f"  Max samples: {max_samples}")
        print(f"  Enable direction: {enable_direction}")
        print(f"  Enable inductance: {enable_inductance}")
        print(f"  Shuffle work items: {shuffle}")
        print(f"  Simulator type: {simulator_type}")
        print(f"  Use optimized: {use_optimized}")
        print(f"  Check duplicates: {self.check_duplicates}")
        if self.check_duplicates:
            print(f"  Max duplicate attempts: {self.max_duplicate_attempts}")
        if block_size is not None:
            print(f"  Fixed block size: {block_size}")
        
        # Validate parameter types against simulator type
        if simulator_type == 'der':
            if not all(ptype == 'DER_PARAMS' for ptype in param_types):
                raise ValueError(f"For 'der' simulator, only 'DER_PARAMS' is allowed as param_type. Got: {param_types}")
        elif simulator_type == 'sbr':
            if 'DER_PARAMS' in param_types:
                raise ValueError(f"For 'sbr' simulator, 'DER_PARAMS' is not allowed as param_type. Got: {param_types}")

        # Create output directory
        trace_specific_output_dir = output_dir / trace_pattern_key
        trace_specific_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load trace SNP files
        trace_snps = parse_snps(trace_pattern)
        if not trace_snps:
            raise ValueError(f"No trace SNP files found in: {trace_pattern}")
        
        print(f"Found {len(trace_snps)} trace SNP files")
        
        # Generate vertical SNP pairs
        vertical_pairs = generate_vertical_snp_pairs(
            vertical_dirs, len(trace_snps), trace_snps, 
            output_dir, trace_pattern_key
        )
        
        print(f"Generated {len(vertical_pairs)} vertical SNP pairs")
        
        # Combine parameter sets
        combined_params = None
        for param_type in param_types:
            param_set = PARAM_SETS_MAP[param_type]
            if combined_params is None:
                combined_params = param_set
            else:
                combined_params = combined_params + param_set
        
        # Apply inductance modification if needed
        combined_params = modify_params_for_inductance(combined_params, enable_inductance)
        
        # Pre-sample fixed configurations if requested (overrides self.fixed_configs from __init__)
        if fixed_config and not self.fixed_configs:
            print(f"\n[FIXED CONFIG] Pre-sampling {max_samples} configurations...")
            self.fixed_configs = []
            for i in range(max_samples):
                config_sample = combined_params.sample()
                self.fixed_configs.append(config_sample)
                if self.debug and (i + 1) % 10 == 0:
                    print(f"[FIXED CONFIG] Sampled {i + 1}/{max_samples} configs")
            print(f"[FIXED CONFIG] Pre-sampled {len(self.fixed_configs)} fixed configurations")
            print(f"[FIXED CONFIG] All trace files will use these same {len(self.fixed_configs)} configs")
            # Reset index for this collection run
            self.fixed_config_index = 0
        
        # Calculate total simulations needed
        total_simulations = 0
        work_items = []  # (trace_snp, vertical_pair, samples_needed, pickle_file)
        
        for trace_snp, vertical_pair in zip(trace_snps, vertical_pairs):
            pickle_file = trace_specific_output_dir / f"{Path(trace_snp).stem}.pkl"
            
            # Check existing samples
            current_samples = 0
            if pickle_file.exists():
                try:
                    with open(pickle_file, 'rb') as f:
                        existing_data = pickle.load(f)
                    current_samples = len(existing_data.get('configs', []))
                except:
                    current_samples = 0
            
            if current_samples < max_samples:
                samples_needed = max_samples - current_samples
                work_items.append((trace_snp, vertical_pair, samples_needed, pickle_file))
                total_simulations += samples_needed
        
        if not work_items:
            print("All files already have sufficient samples")
            return {"status": "complete", "message": "No work needed"}
        
        self.stats["total_simulations"] = total_simulations
        print(f"Total simulations needed: {total_simulations}")
        
        # Shuffle work_items if requested
        if shuffle:
            import random
            random.shuffle(work_items)
        
        # Pre-load all vertical SNPs into cache
        print("Pre-loading vertical SNPs into memory cache...")
        unique_vertical_snps = set()
        for _, vertical_pair, _, _ in work_items:
            snp_drv_path, snp_odt_path = vertical_pair
            unique_vertical_snps.add(snp_drv_path)
            unique_vertical_snps.add(snp_odt_path)
        
        cache_start_time = time.time()
        for snp_path in unique_vertical_snps:
            self.snp_cache.get_snp(snp_path)
        cache_load_time = time.time() - cache_start_time
        
        cache_stats = self.snp_cache.get_stats()
        print(f"Cache loaded in {cache_load_time:.2f}s: {cache_stats['cached_files']} files, "
              f"{cache_stats['memory_usage_mb']:.1f}MB")
        
        # Process each work item with tqdm progress bar
        with tqdm(total=total_simulations, desc=f"Collecting {trace_pattern_key}", unit="sim") as pbar:
            for trace_snp, vertical_pair, samples_needed, pickle_file in work_items:
                if _shutdown_event.is_set():
                    print("Shutdown requested, stopping collection...")
                    break
                    
                self._process_trace_file(
                    trace_snp, vertical_pair, samples_needed, pickle_file,
                    combined_params, enable_direction, pbar, param_types, max_samples,
                    simulator_type, use_optimized, block_size
                )
        
        self.stats["end_time"] = time.time()
        total_time = self.stats["end_time"] - self.stats["start_time"]
        
        # Final statistics
        final_cache_stats = self.snp_cache.get_stats()
        
        results = {
            "status": "complete" if not _shutdown_event.is_set() else "interrupted",
            "total_simulations": total_simulations,
            "completed_simulations": self.stats["completed_simulations"],
            "failed_simulations": self.stats["failed_simulations"],
            "duplicate_configs_skipped": self.stats["duplicate_configs_skipped"],
            "files_skipped_exhausted": self.stats["files_skipped_exhausted"],
            "total_time_seconds": total_time,
            "average_time_per_simulation": total_time / max(1, self.stats["completed_simulations"]),
            "cache_stats": final_cache_stats,
            "output_directory": str(trace_specific_output_dir)
        }
        
        print(f"\n=== Collection Complete ===")
        print(f"Total time: {total_time:.2f}s ({total_time/60:.1f}min)")
        print(f"Completed: {self.stats['completed_simulations']}/{total_simulations} simulations")
        print(f"Failed: {self.stats['failed_simulations']} simulations")
        if self.check_duplicates:
            print(f"Duplicate configs skipped: {self.stats['duplicate_configs_skipped']}")
            if self.stats["files_skipped_exhausted"] > 0:
                print(f"Files skipped (parameter space exhausted): {self.stats['files_skipped_exhausted']}")
        print(f"Average time per simulation: {results['average_time_per_simulation']:.2f}s")
        print(f"Cache efficiency: {final_cache_stats['hit_rate_percent']:.1f}% hit rate")
        
        return results
    
    def _process_trace_file(self, trace_snp: Path, vertical_pair: Tuple[Path, Path], 
                           samples_needed: int, pickle_file: Path, combined_params: Any,
                           enable_direction: bool, pbar: tqdm, param_types: List[str], max_samples: int,
                           simulator_type: str, use_optimized: bool, block_size: int = None):
        """Process a single trace file with optimized sequential processing"""
        
        # INITIAL RACE CONDITION CHECK: Verify quota not already filled by other parallel jobs
        try:
            current_sample_count = 0
            if pickle_file.exists():
                with open(pickle_file, 'rb') as f:
                    check_data = pickle.load(f)
                current_sample_count = len(check_data.get('configs', []))
            
            if current_sample_count >= max_samples:
                print(f"[RACE PROTECTION] {trace_snp.name} already has {current_sample_count} samples (>= {max_samples}), skipping entire file")
                return
            
            # Update samples_needed based on current state
            updated_samples_needed = max_samples - current_sample_count
            if updated_samples_needed != samples_needed:
                print(f"[RACE PROTECTION] {trace_snp.name} samples needed updated from {samples_needed} to {updated_samples_needed}")
                samples_needed = updated_samples_needed
                
        except Exception as e:
            if self.debug:
                print(f"[RACE PROTECTION] Could not perform initial check on {pickle_file}: {e}, proceeding with original plan")
        
        if self.debug:
            print(f"\nProcessing {trace_snp.name}: {samples_needed} samples")
        
        # Dynamically select the simulation function
        if simulator_type == 'der':
            from simulation.engine.der_simulator import snp_der_simulation
            simulation_func = snp_der_simulation
        else: # Default to sbr
            from simulation.engine.sbr_simulator import snp_eyewidth_simulation
            simulation_func = snp_eyewidth_simulation

        # Load trace SNP (will be cached automatically)
        trace_ntwk = read_snp(trace_snp)
        n_ports = trace_ntwk.nports
        n_lines = n_ports // 2
        
        # Load vertical SNPs from cache
        snp_drv_path, snp_odt_path = vertical_pair
        drv_ntwk = self.snp_cache.get_snp(snp_drv_path)
        odt_ntwk = self.snp_cache.get_snp(snp_odt_path)
        
        # Load existing data
        existing_data = {'configs': [], 'line_ews': [], 'snp_drvs': [], 'snp_odts': [], 'directions': [], 'meta': {}}
        if pickle_file.exists():
            try:
                with open(pickle_file, 'rb') as f:
                    existing_data = pickle.load(f)
            except:
                pass
        
        # Load existing configurations for duplication checking
        existing_configs = self._load_existing_configs(pickle_file)
        if self.debug and existing_configs:
            print(f"[DUPLICATE CHECK] {trace_snp.name}: Loaded {len(existing_configs)} existing configurations")
        
        # Collect new samples
        new_results = []
        for sample_idx in range(samples_needed):
            if _shutdown_event.is_set():
                break
            
            # RACE CONDITION PROTECTION: Check if other parallel jobs have already filled the quota
            # This prevents over-collection when multiple bsub jobs run simultaneously
            if sample_idx > 0 and sample_idx % 5 == 0:  # Check every 5 samples to avoid excessive I/O
                try:
                    current_sample_count = 0
                    if pickle_file.exists():
                        with open(pickle_file, 'rb') as f:
                            check_data = pickle.load(f)
                        current_sample_count = len(check_data.get('configs', []))
                    
                    if current_sample_count >= max_samples:
                        print(f"[RACE PROTECTION] {trace_snp.name} already has {current_sample_count} samples (>= {max_samples}), stopping collection")
                        break
                    
                    # Update remaining samples needed
                    remaining_needed = max_samples - current_sample_count
                    if remaining_needed <= 0:
                        print(f"[RACE PROTECTION] {trace_snp.name} quota already filled by other jobs")
                        break
                    
                    if self.debug and remaining_needed < samples_needed - sample_idx:
                        print(f"[RACE PROTECTION] {trace_snp.name} samples reduced from {samples_needed - sample_idx} to {remaining_needed} due to other jobs")
                        
                except Exception as e:
                    # If file check fails, continue with original plan
                    if self.debug:
                        print(f"[RACE PROTECTION] Could not check {pickle_file}: {e}, continuing...")
                    pass
            
            # Retry logic for failed simulations
            max_retries = 3
            retry_count = 0
            simulation_successful = False
            
            while not simulation_successful and retry_count <= max_retries:
                # Initialize variables for error handling
                combined_config = None
                sim_directions = None
                
                try:
                    # Sample parameters (get new unique config for retries)
                    combined_config = self._sample_unique_config(combined_params, existing_configs)
                    
                    # Check if parameter space is exhausted
                    if combined_config is None:
                        print(f"[SKIP FILE] {trace_snp.name}: Parameter space exhausted, skipping remaining samples")
                        self.stats["files_skipped_exhausted"] += 1
                        return  # Exit the entire trace file processing
                    
                    # Generate directions
                    sim_directions = generate_directions(n_lines, enable_direction, block_size=block_size)
                    
                    # Run simulation
                    sim_start_time = time.time()
                    try:
                        # Prepare simulator-specific arguments
                        sim_kwargs = {
                            'config': combined_config,
                            'snp_files': (trace_ntwk, drv_ntwk, odt_ntwk),
                        }
                        if simulator_type == 'sbr':
                            sim_kwargs['directions'] = sim_directions
                            sim_kwargs['use_optimized'] = use_optimized

                        line_ew = simulation_func(**sim_kwargs)
                        simulation_successful = True
                        
                    except Exception as sim_error:
                        # Handle simulation errors with detailed metadata
                        error_msg = str(sim_error)
                        retry_count += 1
                        
                        # Format comprehensive error metadata
                        metadata_report = self._format_error_metadata(
                            trace_snp, snp_drv_path, snp_odt_path, combined_config,
                            sim_directions, sample_idx, samples_needed, error_msg
                        )
                        
                        if retry_count <= max_retries:
                            print(f"[ERROR] SEQUENTIAL simulation failed (attempt {retry_count}/{max_retries + 1}), will retry with new config:")
                            print(metadata_report)
                            print(f"[RETRY] Retrying SEQUENTIAL simulation for {trace_snp.name}, sample {sample_idx + 1}")
                            continue
                        else:
                            print(f"[ERROR] SEQUENTIAL simulation failed after {max_retries + 1} attempts, giving up:")
                            print(metadata_report)
                            print(f"[SKIP] Skipping SEQUENTIAL simulation for {trace_snp.name}, sample {sample_idx + 1}")
                            break
                    
                    if not simulation_successful:
                        continue  # Retry with new config
                    
                except Exception as outer_error:
                    # Handle errors in directions generation or other setup
                    error_msg = str(outer_error)
                    retry_count += 1
                    
                    # Format comprehensive error metadata
                    metadata_report = self._format_error_metadata(
                        trace_snp, snp_drv_path, snp_odt_path, combined_config,
                        sim_directions, sample_idx, samples_needed, error_msg
                    )
                    
                    if retry_count <= max_retries:
                        print(f"[ERROR] SEQUENTIAL setup/preprocessing failed (attempt {retry_count}/{max_retries + 1}), will retry:")
                        print(metadata_report)
                        print(f"[RETRY] Retrying SEQUENTIAL setup for {trace_snp.name}, sample {sample_idx + 1}")
                        continue
                    else:
                        print(f"[ERROR] SEQUENTIAL setup/preprocessing failed after {max_retries + 1} attempts:")
                        print(metadata_report)
                        print(f"[SKIP] Skipping SEQUENTIAL simulation for {trace_snp.name}, sample {sample_idx + 1}")
                        break
            
            # Always count this sample (successful or failed after retries)
            # Update progress regardless of success/failure to maintain accurate counts
            pbar.update(1)
            
            # If simulation failed after all retries, skip result processing but count the sample
            if not simulation_successful:
                self.stats["failed_simulations"] += 1
                if self.debug:
                    print(f"  Sample {sample_idx+1} failed after all retries")
                continue
            
            # Process successful simulation results
            sim_time = time.time() - sim_start_time
            
            # Handle tuple return
            if isinstance(line_ew, tuple):
                line_ew, actual_directions = line_ew
                sim_directions = actual_directions
            
            # Process results
            line_ew = np.array(line_ew)
            line_ew[line_ew >= 99.9] = -0.1
            
            # Create a structured dataclass instance for the result
            config_values, config_keys = combined_config.to_list(return_keys=True)
            result = SimulationResult(
                config_values=config_values,
                config_keys=config_keys,
                line_ews=line_ew.tolist(),
                snp_drv=snp_drv_path.as_posix(),
                snp_odt=snp_odt_path.as_posix(),
                directions=sim_directions.tolist(),
                snp_horiz=str(trace_snp),
                n_ports=n_ports,
                param_types=param_types
            )
            
            # Add this configuration to our existing set to avoid duplicates within this collection run
            config_tuple = tuple(config_values)
            existing_configs.add(config_tuple)
            
            new_results.append(result)
            self.stats["completed_simulations"] += 1
            
            if self.debug:
                print(f"  Sample {sample_idx+1}/{samples_needed}: EW={line_ew} ({sim_time:.2f}s)")
            
            # Batch save every N samples
            if len(new_results) >= self.batch_size:
                self._save_results(pickle_file, new_results)
                new_results = []
        
        # Save remaining results
        if new_results:
            self._save_results(pickle_file, new_results)
    
    def _save_results(self, pickle_file: Path, new_results: List[Dict[str, Any]]):
        """Save results to a pickle file using the standardized DataWriter."""
        try:
            data_writer = DataWriter(pickle_file)
            
            # Add all new results to the writer
            for result in new_results:
                data_writer.add_result(result)
            
            # Save the updated data
            data_writer.save()

            print(f"[SAVE] Saved {len(new_results)} new results to {pickle_file.name} (total: {data_writer.get_sample_count()})")
                
        except Exception as e:
            print(f"Error saving results to {pickle_file}: {e}")

def main():
    """Main function for optimized sequential data collection"""
    print("EyeDiagramNet - Optimized Sequential Data Collector")
    print("=" * 60)
    
    # Parse arguments using centralized argparser
    parser = build_argparser()
    args = parser.parse_args()
    
    # Apply BLAS thread optimization after parsing args
    optimize_blas_for_sequential(num_threads=args.num_threads)
    
    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"Config file not found: {args.config}")
        print("Please create the config file or provide command line arguments")
        return 1
    except Exception as e:
        print(f"Error loading config: {e}")
        return 1
    
    # Resolve dataset paths (same as parallel_collector.py)
    horizontal_dataset = config.get('dataset', {}).get('horizontal_dataset', {})
    vertical_dataset = config.get('dataset', {}).get('vertical_dataset')
    
    # Override config with command line arguments (same pattern as parallel_collector.py)
    trace_pattern_key = args.trace_pattern if args.trace_pattern else config['data']['trace_pattern']
    trace_pattern = resolve_trace_pattern(trace_pattern_key, horizontal_dataset)
    vertical_dirs = resolve_vertical_dirs(vertical_dataset)
    output_dir = Path(args.output_dir) if args.output_dir else Path(config['data']['output_dir'])
    param_type_str = args.param_type if args.param_type else config['boundary']['param_type']
    param_types = parse_param_types(param_type_str)
    max_samples = args.max_samples if args.max_samples else config['boundary']['max_samples']
    
    # Handle enable_direction logic (default to False) - same as parallel_collector.py
    enable_direction = args.enable_direction or config['boundary'].get('enable_direction', False)
    
    # Handle enable_inductance logic (default to False) - same as parallel_collector.py
    enable_inductance = args.enable_inductance or config['boundary'].get('enable_inductance', False)
    
    # Handle block_size from args or config
    block_size = args.block_size if args.block_size is not None else config['boundary'].get('block_size')

    # Handle shuffle logic (default to False)
    shuffle = args.shuffle if hasattr(args, 'shuffle') else config['boundary'].get('shuffle', False)
    
    # Handle simulator type
    simulator_type = args.simulator_type or config.get('runner', {}).get('simulator_type', 'sbr')

    # Handle use_optimized logic
    use_optimized = args.use_optimized or config.get('runner', {}).get('use_optimized', False)
    
    # Handle fixed-config logic
    fixed_config = args.fixed_config or config['boundary'].get('fixed_config', False)

    debug = args.debug if args.debug else config.get('debug', False)
    
    # Get batch size from config, ignoring runner section worker-specific settings
    # Use reasonable defaults for sequential processing
    runner_config = config.get('runner', {})
    batch_size = runner_config.get('batch_size', 50)  # Larger batches for all-core processing
    
    # Display configuration (same style as parallel_collector.py)
    print(f"\nUsing configuration:")
    print(f"  Trace pattern: {trace_pattern_key} -> {trace_pattern}")
    print(f"  Vertical dirs: {vertical_dirs}")
    print(f"  Output dir: {output_dir}")
    print(f"  Parameter types: {param_types}")
    print(f"  Max samples: {max_samples}")
    print(f"  Enable direction: {enable_direction}")
    print(f"  Enable inductance: {enable_inductance}")
    print(f"  Shuffle work items: {shuffle}")
    print(f"  Fixed config: {fixed_config}")
    print(f"  Debug mode: {debug}")
    print(f"  Batch size: {batch_size}")
    print(f"  Processing mode: Sequential (using all cores)")
    print(f"  Simulator Type: {simulator_type}")
    print(f"  Use optimized: {use_optimized}")
    if block_size is not None:
        print(f"  Fixed block size: {block_size}")
    
    # Create collector (fixed configs will be generated inside collect_data if needed)
    collector = SequentialCollector(config, debug=debug)
    
    try:
        # Run collection
        results = collector.collect_data(
            trace_pattern_key=trace_pattern_key,
            trace_pattern=trace_pattern,
            vertical_dirs=vertical_dirs,
            output_dir=output_dir,
            param_types=param_types,
            max_samples=max_samples,
            enable_direction=enable_direction,
            enable_inductance=enable_inductance,
            shuffle=shuffle,
            simulator_type=simulator_type,
            use_optimized=use_optimized,
            block_size=block_size,
            fixed_config=fixed_config
        )
        
        print(f"\nCollection completed successfully!")
        print(f"Results saved to: {results['output_directory']}")
        
        # Print performance summary if available
        if NETWORK_PROFILING_AVAILABLE:
            print("\n=== Performance Summary ===")
            print_performance_summary()
        
        return 0
        
    except KeyboardInterrupt:
        print("\nCollection interrupted by user")
        return 130
    except Exception as e:
        print(f"Collection failed: {e}")
        if debug:
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 