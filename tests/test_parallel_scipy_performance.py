#!/usr/bin/env python3
"""
Parallel SciPy/NumPy Performance Diagnostics

This test diagnoses performance issues when running scipy/numpy operations
in multiprocessing environments, specifically targeting the 20x slowdown
observed in the sequential collector.

The test covers:
1. GIL contention with numpy/scipy operations
2. BLAS threading configuration impact
3. Memory access patterns and cache effects
4. Multiprocessing vs threading performance
5. Specific operations used in sbr_simulator.py

Usage:
    python test_parallel_scipy_performance.py [--workers 4] [--verbose]
"""

import os
import sys
import time
import psutil
import threading
import multiprocessing
import concurrent.futures
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from contextlib import contextmanager
import argparse

# Set up path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import the actual modules we're testing
try:
    import numpy as np
    import scipy
    import scipy.signal
    import scipy.linalg
    import scipy.interpolate
    import skrf as rf
    SCIPY_AVAILABLE = True
except ImportError as e:
    print(f"SciPy/NumPy not available: {e}")
    SCIPY_AVAILABLE = False

try:
    from common.signal_utils import read_snp
    from common.param_types import SampleResult
    from simulation.engine.sbr_simulator import snp_eyewidth_simulation, EyeWidthSimulator
    from simulation.engine.network_utils import s2y, y2s, s2z, z2s, nudge_eig, rsolve
    SIMULATION_AVAILABLE = True
except ImportError as e:
    print(f"Simulation modules not available: {e}")
    SIMULATION_AVAILABLE = False

@dataclass
class PerformanceResult:
    """Container for performance test results"""
    test_name: str
    execution_mode: str
    worker_count: int
    total_time: float
    avg_time_per_op: float
    operations_per_second: float
    blas_threads: str
    cpu_usage_percent: float
    memory_usage_mb: float
    speedup_vs_single: float = 1.0

class BLASConfigManager:
    """Manages BLAS threading configuration"""
    
    def __init__(self):
        self.original_config = self._get_current_config()
    
    def _get_current_config(self) -> Dict[str, str]:
        """Get current BLAS configuration"""
        return {
            'OMP_NUM_THREADS': os.environ.get('OMP_NUM_THREADS', 'unset'),
            'MKL_NUM_THREADS': os.environ.get('MKL_NUM_THREADS', 'unset'),
            'OPENBLAS_NUM_THREADS': os.environ.get('OPENBLAS_NUM_THREADS', 'unset'),
            'NUMEXPR_NUM_THREADS': os.environ.get('NUMEXPR_NUM_THREADS', 'unset'),
        }
    
    @contextmanager
    def set_blas_threads(self, num_threads: int):
        """Context manager to temporarily set BLAS threads"""
        # Save original values
        original = self._get_current_config()
        
        # Set new values
        thread_str = str(num_threads)
        os.environ['OMP_NUM_THREADS'] = thread_str
        os.environ['MKL_NUM_THREADS'] = thread_str
        os.environ['OPENBLAS_NUM_THREADS'] = thread_str
        os.environ['NUMEXPR_NUM_THREADS'] = thread_str
        
        try:
            yield
        finally:
            # Restore original values
            for key, value in original.items():
                if value == 'unset':
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
    
    def get_info_string(self) -> str:
        """Get BLAS configuration info string"""
        config = self._get_current_config()
        return f"OMP:{config['OMP_NUM_THREADS']}, MKL:{config['MKL_NUM_THREADS']}, OPENBLAS:{config['OPENBLAS_NUM_THREADS']}"

class SciPyOperationBenchmark:
    """Benchmarks specific scipy operations used in sbr_simulator.py"""
    
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.blas_manager = BLASConfigManager()
        
    def log(self, message):
        if self.verbose:
            print(f"[BENCH] {message}")
    
    def create_test_matrices(self, size: int = 200) -> Tuple[np.ndarray, np.ndarray]:
        """Create test matrices similar to those in sbr_simulator.py"""
        np.random.seed(42)  # Consistent results
        
        # Create S-parameter-like matrices (complex)
        s_matrix = (np.random.randn(size, size, size) + 
                   1j * np.random.randn(size, size, size)) * 0.3
        
        # Create impedance-like matrices (complex)
        z0 = 50.0 * np.ones((size, size), dtype=complex)
        
        return s_matrix, z0
    
    def create_test_frequency_data(self, nfreq: int = 1000, nports: int = 8) -> Tuple[np.ndarray, np.ndarray]:
        """Create frequency-domain data similar to sbr_simulator.py"""
        np.random.seed(42)
        
        # Frequency array
        freq = np.linspace(1e9, 10e9, nfreq)
        
        # S-parameter data
        s_data = np.random.randn(nfreq, nports, nports) + 1j * np.random.randn(nfreq, nports, nports)
        s_data *= 0.3  # Keep passive
        
        return freq, s_data
    
    def benchmark_linalg_operations(self, iterations: int = 100) -> float:
        """Benchmark linear algebra operations (matrix solve, SVD, etc.)"""
        s_matrix, z0 = self.create_test_matrices(100)
        
        start_time = time.perf_counter()
        
        for i in range(iterations):
            # Operations similar to those in network_utils.py
            try:
                # Matrix solve (used in rsolve)
                A = s_matrix[i % s_matrix.shape[0]]
                B = np.random.randn(A.shape[0], A.shape[1]) + 1j * np.random.randn(A.shape[0], A.shape[1])
                X = np.linalg.solve(A, B)
                
                # SVD (used in nudge_svd)
                U, s, Vh = np.linalg.svd(A)
                
                # Eigenvalue decomposition (used in nudge_eig)
                eigvals, eigvecs = np.linalg.eig(A)
                
            except np.linalg.LinAlgError:
                # Skip singular matrices
                continue
        
        return time.perf_counter() - start_time
    
    def benchmark_signal_processing(self, iterations: int = 50) -> float:
        """Benchmark signal processing operations"""
        freq, s_data = self.create_test_frequency_data(1000, 8)
        
        start_time = time.perf_counter()
        
        for i in range(iterations):
            # Interpolation (used in get_line_sbr)
            new_freq = np.linspace(freq[0], freq[-1], len(freq) // 2)
            
            for port_i in range(s_data.shape[1]):
                for port_j in range(s_data.shape[2]):
                    # Interpolate magnitude and phase separately
                    magnitude = np.abs(s_data[:, port_i, port_j])
                    phase = np.unwrap(np.angle(s_data[:, port_i, port_j]))
                    
                    mag_interp = np.interp(new_freq, freq, magnitude)
                    phase_interp = np.interp(new_freq, freq, phase)
            
            # FFT operations (used in inverse continuous FT)
            for port_i in range(s_data.shape[1]):
                signal_freq = s_data[:, port_i, 0]
                signal_time = np.fft.ifft(signal_freq)
        
        return time.perf_counter() - start_time
    
    def benchmark_frequency_response(self, iterations: int = 20) -> float:
        """Benchmark frequency response calculations (scipy.signal)"""
        start_time = time.perf_counter()
        
        for i in range(iterations):
            # Create transfer function (used in add_ctle)
            zeros = [-1e6]
            poles = [-1e5, -1e7]
            gain = 1.0
            
            system = scipy.signal.ZerosPolesGain(zeros, poles, gain)
            
            # Calculate frequency response
            freq = np.logspace(6, 10, 1000)
            _, response = scipy.signal.freqresp(system, 2 * np.pi * freq)
        
        return time.perf_counter() - start_time
    
    def benchmark_combined_operations(self, iterations: int = 10) -> float:
        """Benchmark combined operations similar to full simulation"""
        s_matrix, z0 = self.create_test_matrices(50)
        
        start_time = time.perf_counter()
        
        for i in range(iterations):
            # Simulate the pattern from sbr_simulator.py
            s_data = s_matrix[i % s_matrix.shape[0]]
            
            # Parameter conversion chain (s2z -> operations -> z2s)
            try:
                # Convert to impedance parameters
                z_data = s_data + np.eye(s_data.shape[0]) * 50.0
                
                # Add some modifications (similar to add_capacitance)
                z_data += np.eye(z_data.shape[0]) * 1j * 1e-12 * 2 * np.pi * 1e9
                
                # Convert back to S-parameters
                s_result = np.linalg.solve(z_data + 50.0 * np.eye(z_data.shape[0]), 
                                          z_data - 50.0 * np.eye(z_data.shape[0]))
                
                # Some signal processing
                time_response = np.fft.ifft(s_result[0, :])
                
            except np.linalg.LinAlgError:
                continue
        
        return time.perf_counter() - start_time

class ParallelPerformanceTester:
    """Tests performance in different parallel execution modes"""
    
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.blas_manager = BLASConfigManager()
        self.benchmark = SciPyOperationBenchmark(verbose)
        
    def log(self, message):
        if self.verbose:
            print(f"[PARALLEL] {message}")
    
    def _run_single_operation(self, operation_name: str, blas_threads: int) -> float:
        """Run a single operation with specified BLAS threads"""
        with self.blas_manager.set_blas_threads(blas_threads):
            if operation_name == "linalg":
                return self.benchmark.benchmark_linalg_operations(50)
            elif operation_name == "signal":
                return self.benchmark.benchmark_signal_processing(20)
            elif operation_name == "frequency":
                return self.benchmark.benchmark_frequency_response(10)
            elif operation_name == "combined":
                return self.benchmark.benchmark_combined_operations(5)
            else:
                raise ValueError(f"Unknown operation: {operation_name}")
    
    def _worker_function(self, args: Tuple[str, int, int]) -> Tuple[float, int]:
        """Worker function for multiprocessing"""
        operation_name, blas_threads, worker_id = args
        
        # Set process affinity if possible (Linux/Mac)
        try:
            import psutil
            process = psutil.Process()
            cpu_count = psutil.cpu_count()
            # Spread workers across CPUs
            cpu_id = worker_id % cpu_count
            process.cpu_affinity([cpu_id])
        except:
            pass
        
        elapsed = self._run_single_operation(operation_name, blas_threads)
        return elapsed, worker_id
    
    def _thread_function(self, args: Tuple[str, int, int]) -> Tuple[float, int]:
        """Thread function for threading"""
        operation_name, blas_threads, thread_id = args
        elapsed = self._run_single_operation(operation_name, blas_threads)
        return elapsed, thread_id
    
    def test_single_threaded(self, operation_name: str, blas_threads: int) -> PerformanceResult:
        """Test single-threaded performance"""
        self.log(f"Testing single-threaded {operation_name} with {blas_threads} BLAS threads")
        
        # Monitor system resources
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        cpu_before = process.cpu_percent()
        
        start_time = time.perf_counter()
        elapsed = self._run_single_operation(operation_name, blas_threads)
        total_time = time.perf_counter() - start_time
        
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        cpu_after = process.cpu_percent()
        
        return PerformanceResult(
            test_name=f"single_{operation_name}",
            execution_mode="single",
            worker_count=1,
            total_time=total_time,
            avg_time_per_op=elapsed,
            operations_per_second=1.0 / elapsed,
            blas_threads=str(blas_threads),
            cpu_usage_percent=(cpu_before + cpu_after) / 2,
            memory_usage_mb=memory_after - memory_before
        )
    
    def test_multiprocessing(self, operation_name: str, num_workers: int, blas_threads: int) -> PerformanceResult:
        """Test multiprocessing performance"""
        self.log(f"Testing multiprocessing {operation_name} with {num_workers} workers, {blas_threads} BLAS threads each")
        
        # Prepare arguments
        args = [(operation_name, blas_threads, i) for i in range(num_workers)]
        
        # Monitor system resources
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        start_time = time.perf_counter()
        
        # Use multiprocessing
        with multiprocessing.Pool(processes=num_workers) as pool:
            results = pool.map(self._worker_function, args)
        
        total_time = time.perf_counter() - start_time
        
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        
        # Analyze results
        worker_times = [r[0] for r in results]
        avg_worker_time = np.mean(worker_times)
        
        return PerformanceResult(
            test_name=f"multiproc_{operation_name}",
            execution_mode="multiprocessing",
            worker_count=num_workers,
            total_time=total_time,
            avg_time_per_op=avg_worker_time,
            operations_per_second=num_workers / avg_worker_time,
            blas_threads=str(blas_threads),
            cpu_usage_percent=psutil.cpu_percent(interval=0.1),
            memory_usage_mb=memory_after - memory_before
        )
    
    def test_threading(self, operation_name: str, num_threads: int, blas_threads: int) -> PerformanceResult:
        """Test threading performance"""
        self.log(f"Testing threading {operation_name} with {num_threads} threads, {blas_threads} BLAS threads")
        
        # Prepare arguments
        args = [(operation_name, blas_threads, i) for i in range(num_threads)]
        
        # Monitor system resources
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        start_time = time.perf_counter()
        
        # Use threading
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(self._thread_function, arg) for arg in args]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        total_time = time.perf_counter() - start_time
        
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        
        # Analyze results
        thread_times = [r[0] for r in results]
        avg_thread_time = np.mean(thread_times)
        
        return PerformanceResult(
            test_name=f"thread_{operation_name}",
            execution_mode="threading",
            worker_count=num_threads,
            total_time=total_time,
            avg_time_per_op=avg_thread_time,
            operations_per_second=num_threads / avg_thread_time,
            blas_threads=str(blas_threads),
            cpu_usage_percent=psutil.cpu_percent(interval=0.1),
            memory_usage_mb=memory_after - memory_before
        )

class SimulationPerformanceTester:
    """Tests actual simulation performance in different modes"""
    
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.blas_manager = BLASConfigManager()
        
    def log(self, message):
        if self.verbose:
            print(f"[SIM] {message}")
    
    def create_test_config(self) -> Dict[str, Any]:
        """Create test configuration for simulation"""
        # Find test data files
        project_root = Path(__file__).parent.parent
        possible_dirs = [
            project_root / "test_data",
            project_root / "test_data_5files",
            Path("test_data"),
            Path("../test_data"),
        ]
        
        test_dir = None
        for d in possible_dirs:
            if d.exists():
                test_dir = d
                break
        
        if test_dir is None:
            raise FileNotFoundError("Could not find test data directory")
        
        # Look for S-parameter files
        snp_files = list(test_dir.glob("*.s*p"))
        if len(snp_files) < 1:
            raise FileNotFoundError(f"No S-parameter files found in {test_dir}")
        
        snp_file = snp_files[0]
        
        return {
            "R_drv": 10.0,
            "R_odt": 1.0e9,
            "C_drv": 1e-13,
            "C_odt": 1e-13,
            "L_drv": 1e-10,
            "L_odt": 1e-10,
            "pulse_amplitude": 0.4,
            "bits_per_sec": 6.4e9,
            "vmask": 0.04,
            "snp_horiz": str(snp_file),
            "directions": [1, 0, 1, 0]  # Assuming 4-line configuration
        }
    
    def run_simulation_worker(self, args: Tuple[Dict[str, Any], int, int]) -> Tuple[float, int]:
        """Worker function for simulation"""
        config, blas_threads, worker_id = args
        
        # Set BLAS threads
        with self.blas_manager.set_blas_threads(blas_threads):
            start_time = time.perf_counter()
            
            try:
                # Create sample result
                sample_result = SampleResult(**config)
                
                # Run simulation
                result = snp_eyewidth_simulation(sample_result)
                
                elapsed = time.perf_counter() - start_time
                return elapsed, worker_id
                
            except Exception as e:
                self.log(f"Worker {worker_id} failed: {e}")
                return float('inf'), worker_id
    
    def test_simulation_performance(self, num_workers: int, blas_threads: int) -> PerformanceResult:
        """Test simulation performance with different worker counts"""
        if not SIMULATION_AVAILABLE:
            self.log("Simulation modules not available, skipping")
            return None
        
        self.log(f"Testing simulation with {num_workers} workers, {blas_threads} BLAS threads each")
        
        try:
            config = self.create_test_config()
        except Exception as e:
            self.log(f"Could not create test config: {e}")
            return None
        
        # Prepare arguments
        args = [(config, blas_threads, i) for i in range(num_workers)]
        
        # Monitor system resources
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        start_time = time.perf_counter()
        
        if num_workers == 1:
            # Single-threaded
            results = [self.run_simulation_worker(args[0])]
        else:
            # Multiprocessing
            with multiprocessing.Pool(processes=num_workers) as pool:
                results = pool.map(self.run_simulation_worker, args)
        
        total_time = time.perf_counter() - start_time
        
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        
        # Analyze results
        worker_times = [r[0] for r in results if r[0] != float('inf')]
        
        if not worker_times:
            self.log("All workers failed")
            return None
        
        avg_worker_time = np.mean(worker_times)
        
        return PerformanceResult(
            test_name=f"simulation_{num_workers}workers",
            execution_mode="simulation",
            worker_count=num_workers,
            total_time=total_time,
            avg_time_per_op=avg_worker_time,
            operations_per_second=num_workers / avg_worker_time,
            blas_threads=str(blas_threads),
            cpu_usage_percent=psutil.cpu_percent(interval=0.1),
            memory_usage_mb=memory_after - memory_before
        )

class ComprehensivePerformanceDiagnostic:
    """Main diagnostic class that runs all tests"""
    
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.parallel_tester = ParallelPerformanceTester(verbose)
        self.sim_tester = SimulationPerformanceTester(verbose)
        self.results = []
        
    def log(self, message):
        if self.verbose:
            print(f"[DIAG] {message}")
    
    def run_blas_threading_analysis(self, max_workers: int = 4):
        """Analyze impact of BLAS threading on performance"""
        self.log("Running BLAS threading analysis...")
        
        cpu_count = psutil.cpu_count()
        blas_thread_configs = [1, 2, 4, cpu_count // 2, cpu_count]
        blas_thread_configs = [t for t in blas_thread_configs if t <= cpu_count]
        
        operations = ["linalg", "signal", "combined"]
        
        for operation in operations:
            self.log(f"Testing {operation} operations...")
            
            for blas_threads in blas_thread_configs:
                # Single-threaded baseline
                result = self.parallel_tester.test_single_threaded(operation, blas_threads)
                result.test_name = f"{operation}_single_blas{blas_threads}"
                self.results.append(result)
                
                # Multiprocessing with varying worker counts
                for workers in [2, max_workers]:
                    if workers <= cpu_count:
                        result = self.parallel_tester.test_multiprocessing(operation, workers, blas_threads)
                        result.test_name = f"{operation}_mp{workers}_blas{blas_threads}"
                        self.results.append(result)
                
                # Threading comparison
                result = self.parallel_tester.test_threading(operation, max_workers, blas_threads)
                result.test_name = f"{operation}_thread{max_workers}_blas{blas_threads}"
                self.results.append(result)
    
    def run_simulation_analysis(self, max_workers: int = 4):
        """Analyze actual simulation performance"""
        self.log("Running simulation analysis...")
        
        blas_configs = [1, 2, psutil.cpu_count() // 2]
        
        for blas_threads in blas_configs:
            # Test different worker counts
            for workers in [1, 2, max_workers]:
                if workers <= psutil.cpu_count():
                    result = self.sim_tester.test_simulation_performance(workers, blas_threads)
                    if result:
                        self.results.append(result)
    
    def calculate_speedups(self):
        """Calculate speedups relative to single-threaded baseline"""
        # Find single-threaded baselines
        baselines = {}
        for result in self.results:
            if result.execution_mode == "single":
                operation = result.test_name.split('_')[0]
                baselines[operation] = result.avg_time_per_op
        
        # Calculate speedups
        for result in self.results:
            operation = result.test_name.split('_')[0]
            if operation in baselines:
                baseline_time = baselines[operation]
                result.speedup_vs_single = baseline_time / result.avg_time_per_op
    
    def print_results(self):
        """Print comprehensive results"""
        print("\n" + "=" * 120)
        print("SCIPY/NUMPY PARALLEL PERFORMANCE DIAGNOSTIC RESULTS")
        print("=" * 120)
        
        # System info
        print(f"\nSystem Information:")
        print(f"  CPU Count: {psutil.cpu_count()}")
        print(f"  Memory: {psutil.virtual_memory().total / 1024 / 1024 / 1024:.1f} GB")
        print(f"  Python Version: {sys.version}")
        print(f"  NumPy Version: {np.__version__}")
        print(f"  SciPy Version: {scipy.__version__}")
        
        # Group results by operation
        by_operation = {}
        for result in self.results:
            operation = result.test_name.split('_')[0]
            if operation not in by_operation:
                by_operation[operation] = []
            by_operation[operation].append(result)
        
        for operation, results in by_operation.items():
            print(f"\n📊 {operation.upper()} OPERATIONS:")
            print("-" * 80)
            
            # Sort by execution mode and worker count
            results.sort(key=lambda x: (x.execution_mode, x.worker_count))
            
            header = f"{'Mode':<15} {'Workers':<8} {'BLAS':<6} {'Avg Time':<10} {'Speedup':<8} {'Ops/sec':<10} {'CPU %':<8} {'Memory MB':<10}"
            print(header)
            print("-" * 80)
            
            for result in results:
                speedup_str = f"{result.speedup_vs_single:.2f}x" if result.speedup_vs_single > 0 else "N/A"
                print(f"{result.execution_mode:<15} {result.worker_count:<8} {result.blas_threads:<6} "
                      f"{result.avg_time_per_op:<10.3f} {speedup_str:<8} {result.operations_per_second:<10.1f} "
                      f"{result.cpu_usage_percent:<8.1f} {result.memory_usage_mb:<10.1f}")
        
        # Performance insights
        print(f"\n🔍 PERFORMANCE INSIGHTS:")
        print("-" * 80)
        
        # Find problematic patterns
        multiproc_results = [r for r in self.results if r.execution_mode == "multiprocessing"]
        single_results = [r for r in self.results if r.execution_mode == "single"]
        
        if multiproc_results and single_results:
            # Look for cases where multiprocessing is slower
            slowdowns = []
            for mp_result in multiproc_results:
                operation = mp_result.test_name.split('_')[0]
                baseline = next((r for r in single_results if r.test_name.startswith(operation)), None)
                if baseline and mp_result.avg_time_per_op > baseline.avg_time_per_op * 1.5:
                    slowdown_factor = mp_result.avg_time_per_op / baseline.avg_time_per_op
                    slowdowns.append((mp_result.test_name, slowdown_factor))
            
            if slowdowns:
                print("⚠️  MULTIPROCESSING SLOWDOWNS DETECTED:")
                for test_name, factor in slowdowns:
                    print(f"    {test_name}: {factor:.1f}x slower than single-threaded")
            
            # Check for GIL-related issues
            thread_results = [r for r in self.results if r.execution_mode == "threading"]
            if thread_results:
                print("\n🔒 GIL CONTENTION ANALYSIS:")
                for thread_result in thread_results:
                    if thread_result.speedup_vs_single < 1.2:  # Poor threading performance
                        print(f"    {thread_result.test_name}: Likely GIL-bound (speedup: {thread_result.speedup_vs_single:.2f}x)")
        
        # BLAS threading recommendations
        print(f"\n💡 OPTIMIZATION RECOMMENDATIONS:")
        print("-" * 80)
        
        # Find best BLAS configuration
        best_configs = {}
        for result in self.results:
            if result.execution_mode == "multiprocessing":
                operation = result.test_name.split('_')[0]
                if operation not in best_configs or result.speedup_vs_single > best_configs[operation][1]:
                    best_configs[operation] = (result.blas_threads, result.speedup_vs_single, result.worker_count)
        
        for operation, (blas_threads, speedup, workers) in best_configs.items():
            print(f"  {operation}: Use {blas_threads} BLAS threads with {workers} workers (speedup: {speedup:.2f}x)")
        
        # Specific recommendations for the sequential collector issue
        print(f"\n🎯 SEQUENTIAL COLLECTOR RECOMMENDATIONS:")
        print("-" * 80)
        
        simulation_results = [r for r in self.results if r.test_name.startswith("simulation")]
        if simulation_results:
            # Find the configuration that shows the issue
            problematic = [r for r in simulation_results if r.speedup_vs_single < 0.5]  # 2x+ slower
            if problematic:
                print("❌ CONFIRMED: Simulation performance degrades in multiprocessing:")
                for result in problematic:
                    print(f"    {result.test_name}: {result.speedup_vs_single:.2f}x (should be >1.0x)")
                
                print("\n🔧 LIKELY CAUSES:")
                print("    1. BLAS threading contention (multiple processes with multi-threaded BLAS)")
                print("    2. Memory bandwidth saturation")
                print("    3. NumPy/SciPy operations not releasing GIL properly")
                print("    4. Cache thrashing from multiple processes")
                
                print("\n✅ SOLUTIONS:")
                print("    1. Set BLAS threads to 1 in multiprocessing workers")
                print("    2. Use process affinity to spread workers across NUMA nodes")
                print("    3. Reduce memory allocation in workers")
                print("    4. Consider using threading instead of multiprocessing for this workload")
        
        else:
            print("⚠️  No simulation results available - check if test data files exist")
    
    def save_results(self, filename: str = "parallel_performance_diagnostic.txt"):
        """Save results to file"""
        with open(filename, 'w') as f:
            f.write("SciPy/NumPy Parallel Performance Diagnostic Results\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"System Information:\n")
            f.write(f"  CPU Count: {psutil.cpu_count()}\n")
            f.write(f"  Memory: {psutil.virtual_memory().total / 1024 / 1024 / 1024:.1f} GB\n")
            f.write(f"  Python Version: {sys.version}\n")
            f.write(f"  NumPy Version: {np.__version__}\n")
            f.write(f"  SciPy Version: {scipy.__version__}\n\n")
            
            for result in self.results:
                f.write(f"Test: {result.test_name}\n")
                f.write(f"  Mode: {result.execution_mode}\n")
                f.write(f"  Workers: {result.worker_count}\n")
                f.write(f"  BLAS Threads: {result.blas_threads}\n")
                f.write(f"  Avg Time: {result.avg_time_per_op:.3f}s\n")
                f.write(f"  Speedup: {result.speedup_vs_single:.2f}x\n")
                f.write(f"  Operations/sec: {result.operations_per_second:.1f}\n")
                f.write(f"  CPU Usage: {result.cpu_usage_percent:.1f}%\n")
                f.write(f"  Memory Usage: {result.memory_usage_mb:.1f} MB\n\n")
        
        print(f"Results saved to: {filename}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Diagnose parallel scipy/numpy performance issues')
    parser.add_argument('--workers', type=int, default=4, help='Maximum number of workers to test (default: 4)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--quick', action='store_true', help='Run quick test (fewer iterations)')
    
    args = parser.parse_args()
    
    if not SCIPY_AVAILABLE:
        print("Error: SciPy/NumPy not available")
        return 1
    
    print("SciPy/NumPy Parallel Performance Diagnostic")
    print("=" * 60)
    print(f"Testing with up to {args.workers} workers")
    print(f"CPU Count: {psutil.cpu_count()}")
    
    # Run comprehensive diagnostic
    diagnostic = ComprehensivePerformanceDiagnostic(verbose=args.verbose)
    
    # Run tests
    try:
        diagnostic.run_blas_threading_analysis(args.workers)
        diagnostic.run_simulation_analysis(args.workers)
        diagnostic.calculate_speedups()
        diagnostic.print_results()
        diagnostic.save_results()
        
        print(f"\n✅ Diagnostic completed successfully!")
        return 0
        
    except Exception as e:
        print(f"❌ Diagnostic failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main()) 