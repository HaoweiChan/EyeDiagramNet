"""Configuration utilities for training data collection."""

import yaml
import argparse
from pathlib import Path

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def resolve_trace_pattern(trace_pattern_key, horizontal_dataset):
    """Resolve trace pattern key to actual directory path"""
    if trace_pattern_key not in horizontal_dataset:
        raise ValueError(f"Trace pattern '{trace_pattern_key}' not found in horizontal_dataset. "
                        f"Available patterns: {list(horizontal_dataset.keys())}")
    return Path(horizontal_dataset[trace_pattern_key])

def resolve_vertical_dirs(vertical_dataset):
    """Convert vertical dataset list to directory paths, handling None/null"""
    if vertical_dataset is None:
        return None
    return [Path(vdir) for vdir in vertical_dataset]

def build_argparser():
    """Build argument parser for the collection script"""
    parser = argparse.ArgumentParser(
        description="Collect eye width simulation data for trace patterns"
    )
    parser.add_argument(
        '--config', type=Path, 
        default='configs/data/default.yaml',
        help="Path to configuration YAML file"
    )
    # Allow command line overrides
    parser.add_argument(
        '--trace_pattern', type=str,
        help="Trace pattern key from horizontal_dataset (overrides config)"
    )
    parser.add_argument(
        '--output_dir', type=Path,
        help="Output directory for pickle files (overrides config)"
    )
    parser.add_argument(
        '--param_type', type=str,
        help="Comma-separated parameter types (overrides config)"
    )
    parser.add_argument(
        '--max_samples', type=int,
        help="Maximum number of samples to collect per trace SNP (overrides config)"
    )
    parser.add_argument(
        '--enable_direction', action='store_true',
        help="Enable random direction generation (default: False, overrides config)"
    )
    parser.add_argument(
        '--enable_inductance', action='store_true',
        help="Enable inductance parameters (default: False, overrides config)"
    )
    parser.add_argument(
        '--debug', action='store_true',
        help="Enable debug mode (overrides config)"
    )
    parser.add_argument(
        '--max_workers', type=int,
        help="Maximum number of worker processes (overrides config)"
    )
    parser.add_argument(
        '--executor_type', type=str, choices=['process', 'thread'], default='process',
        help="Executor type: 'process' for ProcessPoolExecutor, 'thread' for ThreadPoolExecutor (default: process)"
    )
    return parser 