import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List

from common.parameters import SampleResult as SimulationResult
from .cleaning import estimate_block_size, validate_boundary_parameters

# Try to import seaborn, make it optional
try:
    import seaborn as sns
    sns.set_palette("husl")
except ImportError:
    print("Warning: seaborn not available, using default matplotlib styling")
    sns = None

# Try to import direction_utils, handle potential ImportError
try:
    from simulation.io.direction_utils import get_valid_block_sizes
except ImportError:
    print("Warning: Could not import get_valid_block_sizes from simulation.io.direction_utils.")
    def get_valid_block_sizes(n_lines):
        return {1}


def detect_contaminated_configs(results: List[SimulationResult]) -> dict:
    """
    Detect samples where config values are actually parameter names (strings).
    This happens when to_list(return_keys=True) return order was swapped.
    """
    if not results:
        return {
            'total_samples': 0,
            'contaminated_count': 0,
            'contaminated_indices': [],
            'contamination_details': []
        }
    
    contaminated_indices = []
    contamination_details = []
    
    for i, result in enumerate(results):
        is_contaminated = False
        issue_desc = []
        
        # Check if config_values contains strings (should be numeric)
        if result.config_values and isinstance(result.config_values[0], str):
            is_contaminated = True
            issue_desc.append("config_values contains strings instead of numbers")
        
        # Check if config_values match config_keys (indicating they're swapped)
        if result.config_keys and result.config_values:
            if len(result.config_keys) == len(result.config_values):
                # Check if the values are actually the keys
                if all(isinstance(v, str) and v in result.config_keys for v in result.config_values):
                    is_contaminated = True
                    issue_desc.append("config_values match config_keys (likely swapped)")
        
        if is_contaminated:
            contaminated_indices.append(i)
            contamination_details.append({
                'sample_index': i,
                'issues': issue_desc,
                'config_keys': result.config_keys[:3] if result.config_keys else [],
                'config_values': result.config_values[:3] if result.config_values else [],
            })
    
    return {
        'total_samples': len(results),
        'contaminated_count': len(contaminated_indices),
        'contaminated_indices': contaminated_indices,
        'contamination_details': contamination_details[:5]  # Show first 5 examples
    }


def detect_duplicate_configs(results: List[SimulationResult]) -> dict:
    """Detect duplicate configuration values within a list of simulation results."""
    if not results:
        return {'total_samples': 0, 'unique_configs': 0, 'duplicate_count': 0, 'duplicate_groups': []}
    
    # Create config signatures for comparison
    config_signatures = []
    config_to_indices = {}
    
    for i, result in enumerate(results):
        # Create a tuple of config values for hashing/comparison
        config_tuple = tuple(result.config_values)
        config_signatures.append(config_tuple)
        
        if config_tuple not in config_to_indices:
            config_to_indices[config_tuple] = []
        config_to_indices[config_tuple].append(i)
    
    # Find duplicates
    duplicate_groups = []
    duplicate_count = 0
    
    for config_tuple, indices in config_to_indices.items():
        if len(indices) > 1:
            duplicate_groups.append({
                'config_values': list(config_tuple),
                'sample_indices': indices,
                'count': len(indices)
            })
            duplicate_count += len(indices) - 1  # All but one are duplicates
    
    return {
        'total_samples': len(results),
        'unique_configs': len(config_to_indices),
        'duplicate_count': duplicate_count,
        'duplicate_groups': duplicate_groups
    }

def analyze_contamination_across_files(pickle_files_list: list[Path]) -> dict:
    """Analyze contaminated config values across all pickle files."""
    file_contamination_stats = {}
    total_contaminated = 0
    total_samples = 0
    contaminated_files = []
    
    from common.pickle_utils import load_pickle_data
    
    for pfile in pickle_files_list:
        try:
            results = load_pickle_data(pfile)
            if results:
                file_stats = detect_contaminated_configs(results)
                file_contamination_stats[str(pfile)] = file_stats
                total_contaminated += file_stats['contaminated_count']
                total_samples += file_stats['total_samples']
                
                if file_stats['contaminated_count'] > 0:
                    contaminated_files.append({
                        'file': pfile.name,
                        'path': str(pfile),
                        'contaminated_count': file_stats['contaminated_count'],
                        'total_samples': file_stats['total_samples'],
                        'contamination_rate': file_stats['contaminated_count'] / file_stats['total_samples'] * 100
                    })
        except Exception as e:
            file_contamination_stats[str(pfile)] = {
                'error': str(e),
                'total_samples': 0,
                'contaminated_count': 0,
                'contaminated_indices': [],
                'contamination_details': []
            }
    
    return {
        'file_stats': file_contamination_stats,
        'total_samples_across_files': total_samples,
        'total_contaminated_across_files': total_contaminated,
        'contaminated_files': contaminated_files,
        'files_with_contamination_count': len(contaminated_files)
    }

def analyze_out_of_range_across_files(pickle_files_list: list[Path]) -> dict:
    """Analyze samples with out-of-range boundary parameters across all pickle files."""
    file_out_of_range_stats = {}
    total_out_of_range = 0
    total_samples = 0
    files_with_out_of_range = []
    
    from common.pickle_utils import load_pickle_data
    
    for pfile in pickle_files_list:
        try:
            results = load_pickle_data(pfile)
            if results:
                out_of_range_samples = []
                for idx, result in enumerate(results):
                    # Use param_types from the result itself
                    if hasattr(result, 'param_types') and result.param_types:
                        is_valid, out_of_range_params = validate_boundary_parameters(result, result.param_types)
                        if not is_valid:
                            out_of_range_samples.append({
                                'sample_index': idx,
                                'param_types': result.param_types,
                                'out_of_range_params': out_of_range_params[:3]  # Show first 3
                            })
                
                file_out_of_range_stats[str(pfile)] = {
                    'total_samples': len(results),
                    'out_of_range_count': len(out_of_range_samples),
                    'out_of_range_samples': out_of_range_samples[:5]  # Show first 5 examples
                }
                
                total_samples += len(results)
                total_out_of_range += len(out_of_range_samples)
                
                if len(out_of_range_samples) > 0:
                    files_with_out_of_range.append({
                        'file': pfile.name,
                        'path': str(pfile),
                        'out_of_range_count': len(out_of_range_samples),
                        'total_samples': len(results),
                        'out_of_range_rate': len(out_of_range_samples) / len(results) * 100
                    })
        except Exception as e:
            file_out_of_range_stats[str(pfile)] = {
                'error': str(e),
                'total_samples': 0,
                'out_of_range_count': 0,
                'out_of_range_samples': []
            }
    
    total_in_range = total_samples - total_out_of_range
    
    return {
        'file_stats': file_out_of_range_stats,
        'total_samples_across_files': total_samples,
        'total_out_of_range_across_files': total_out_of_range,
        'total_in_range_across_files': total_in_range,
        'files_with_out_of_range': files_with_out_of_range,
        'files_with_out_of_range_count': len(files_with_out_of_range)
    }

def analyze_duplications_across_files(pickle_files_list: list[Path]) -> dict:
    """Analyze duplications in config values across all pickle files."""
    file_duplication_stats = {}
    total_duplicates = 0
    total_samples = 0
    total_unique_configs = 0
    
    from common.pickle_utils import load_pickle_data
    
    for pfile in pickle_files_list:
        try:
            results = load_pickle_data(pfile)
            if results:
                file_stats = detect_duplicate_configs(results)
                file_duplication_stats[str(pfile)] = file_stats
                total_duplicates += file_stats['duplicate_count']
                total_samples += file_stats['total_samples']
                total_unique_configs += file_stats['unique_configs']
        except Exception as e:
            file_duplication_stats[str(pfile)] = {
                'error': str(e),
                'total_samples': 0,
                'unique_configs': 0,
                'duplicate_count': 0,
                'duplicate_groups': []
            }
    
    return {
        'file_stats': file_duplication_stats,
        'total_samples_across_files': total_samples,
        'total_unique_configs_across_files': total_unique_configs,
        'total_duplicates_across_files': total_duplicates,
        'files_with_duplicates': sum(1 for stats in file_duplication_stats.values() 
                                   if isinstance(stats, dict) and stats.get('duplicate_count', 0) > 0)
    }

def detect_legacy_format_files(pickle_files_list: list[Path]) -> dict:
    """Detect which pickle files use legacy naming conventions."""
    legacy_stats = {
        'legacy_files': 0,
        'new_format_files': 0,
        'malformed_files': 0,
        'legacy_file_names': []
    }
    
    for pfile in pickle_files_list:
        try:
            with open(pfile, 'rb') as f:
                data = pickle.load(f)
        except Exception:
            legacy_stats['malformed_files'] += 1
            continue
            
        # Check for legacy vs new format
        if isinstance(data, dict):
            has_legacy = 'snp_txs' in data and 'snp_rxs' in data
            has_new = 'snp_drvs' in data and 'snp_odts' in data
            
            if has_legacy and not has_new:
                legacy_stats['legacy_files'] += 1
                legacy_stats['legacy_file_names'].append(pfile.name)
            elif has_new:
                legacy_stats['new_format_files'] += 1
            else:
                legacy_stats['malformed_files'] += 1
        else:
            legacy_stats['malformed_files'] += 1
    
    return legacy_stats

warnings.filterwarnings('ignore')
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)

def is_numeric_list(data: list) -> bool:
    """Check if a list contains numeric data (non-string)."""
    if not data or not isinstance(data, (list, np.ndarray)):
        return False
    return not isinstance(data[0], str)

def compute_direction_stats_for_files(pickle_files_list: list[Path]) -> pd.DataFrame:
    """Build a per-sample directions stats dataframe across all files."""
    stats: list[dict] = []
    for pfile in pickle_files_list:
        try:
            with open(pfile, 'rb') as f:
                data = pickle.load(f)
        except Exception:
            continue
        directions_list = data.get('directions', []) or []
        for i, dir_arr in enumerate(directions_list):
            arr = np.asarray(dir_arr).astype(int).flatten()
            n_lines = int(arr.size)
            if n_lines == 0:
                continue
            block_est = estimate_block_size(arr)
            valid_sizes = get_valid_block_sizes(n_lines)
            is_valid = block_est in valid_sizes
            stats.append({
                'file': str(pfile),
                'sample_idx': i,
                'n_lines': n_lines,
                'block_size_estimate': block_est,
                'is_valid_block_size': is_valid,
                'zeros': int((arr == 0).sum()),
                'ones': int((arr == 1).sum()),
            })
    if not stats:
        return pd.DataFrame(columns=['file','sample_idx','n_lines','block_size_estimate','is_valid_block_size','zeros','ones'])
    return pd.DataFrame(stats)

def plot_eye_width_distributions(all_results: List[SimulationResult], output_dir: Path):
    """Generates and saves plots for eye-width distributions."""
    if not all_results:
        print("No eye width data to plot.")
        return

    # Extract line_ews from the list of dataclasses
    line_ews_array = np.array([res.line_ews for res in all_results])
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Overall distribution
    axes[0,0].hist(line_ews_array.flatten(), bins=50, alpha=0.7, edgecolor='black')
    axes[0,0].set_title('Overall Eye Width Distribution')
    axes[0,0].set_xlabel('Eye Width')
    axes[0,0].set_ylabel('Frequency')
    axes[0,0].axvline(np.nanmean(line_ews_array), color='red', linestyle='--', label=f'Mean: {np.nanmean(line_ews_array):.2f}')
    axes[0,0].legend()
    
    # Distribution excluding closed eyes
    open_eyes = line_ews_array[line_ews_array >= 0]
    if len(open_eyes) > 0:
        axes[0,1].hist(open_eyes.flatten(), bins=50, alpha=0.7, edgecolor='black', color='green')
        axes[0,1].set_title('Eye Width Distribution (Open Eyes Only)')
        axes[0,1].set_xlabel('Eye Width')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].axvline(np.nanmean(open_eyes), color='red', linestyle='--', label=f'Mean: {np.nanmean(open_eyes):.2f}')
        axes[0,1].legend()
        
    # Box plot by line
    if line_ews_array.ndim > 1 and line_ews_array.shape[1] > 1:
        line_data = [line_ews_array[:, i] for i in range(line_ews_array.shape[1])]
        axes[1,0].boxplot(line_data, labels=[f'Line {i}' for i in range(len(line_data))])
        axes[1,0].set_title('Eye Width Distribution by Line')
        axes[1,0].set_ylabel('Eye Width')
    else:
        axes[1,0].text(0.5, 0.5, 'Single line data or 1D array', ha='center', va='center', transform=axes[1,0].transAxes)
        axes[1,0].set_title('Eye Width by Line (N/A)')
        
    # Cumulative distribution
    sorted_ews = np.sort(line_ews_array.flatten())
    y_vals = np.arange(1, len(sorted_ews) + 1) / len(sorted_ews)
    axes[1,1].plot(sorted_ews, y_vals)
    axes[1,1].set_title('Cumulative Distribution of Eye Widths')
    axes[1,1].set_xlabel('Eye Width')
    axes[1,1].set_ylabel('Cumulative Probability')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_dir / 'eye_width_distributions.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nDistribution plots saved to: {plot_path}")

def generate_summary_report(pickle_dir: Path, pickle_files: list, all_results: List[SimulationResult], analysis_results: dict, output_dir: Path):
    """Generates and saves a final summary report."""
    report = []
    report.append("EYE DIAGRAM TRAINING DATA SUMMARY REPORT")
    report.append("=" * 50)
    report.append(f"Generated: {pd.Timestamp.now()}")
    report.append(f"Data directory: {pickle_dir}")
    report.append("")

    # Detect legacy format files
    legacy_stats = detect_legacy_format_files(pickle_files)
    
    # Dataset overview
    report.append("DATASET OVERVIEW:")
    report.append(f"  Total pickle files: {len(pickle_files)}")
    report.append(f"  Total samples: {len(all_results)}")
    if all_results:
        report.append(f"  Parameters per sample: {len(all_results[0].config_values)}")
        report.append(f"  Lines per sample: {len(all_results[0].line_ews)}")
    report.append("")
    
    # Data format analysis
    report.append("DATA FORMAT ANALYSIS:")
    report.append(f"  New format files (snp_drvs/snp_odts): {legacy_stats['new_format_files']}")
    report.append(f"  Legacy format files (snp_txs/snp_rxs): {legacy_stats['legacy_files']}")
    report.append(f"  Malformed files: {legacy_stats['malformed_files']}")
    if legacy_stats['legacy_files'] > 0:
        report.append(f"  Legacy files found: {', '.join(legacy_stats['legacy_file_names'][:5])}" + 
                      (f" and {len(legacy_stats['legacy_file_names']) - 5} more..." if len(legacy_stats['legacy_file_names']) > 5 else ""))
    report.append("")

    # Eye width statistics
    if all_results:
        line_ews_array = np.array([res.line_ews for res in all_results])
        report.append("EYE WIDTH STATISTICS:")
        report.append(f"  Mean: {np.nanmean(line_ews_array):.3f}")
        report.append(f"  Std: {np.nanstd(line_ews_array):.3f}")
        report.append(f"  Min: {np.nanmin(line_ews_array):.3f}")
        report.append(f"  Max: {np.nanmax(line_ews_array):.3f}")
        report.append(f"  Closed eyes: {(line_ews_array < 0).sum()} ({(line_ews_array < 0).mean()*100:.1f}%)")
        report.append("")

    # File statistics
    if analysis_results.get('file_stats'):
        summary_df = pd.DataFrame(analysis_results['file_stats'])
        report.append("FILE STATISTICS:")
        report.append(f"  Average samples per file: {summary_df['samples'].mean():.1f}")
        report.append(f"  Min samples per file: {summary_df['samples'].min()}")
        report.append(f"  Max samples per file: {summary_df['samples'].max()}")
        report.append("")

    # Contamination analysis (CRITICAL)
    if analysis_results.get('contamination_stats'):
        cont_stats = analysis_results['contamination_stats']
        report.append("⚠️  CONTAMINATION ANALYSIS (CRITICAL):")
        report.append(f"  Files with contaminated configs: {cont_stats['files_with_contamination_count']}/{len(pickle_files)}")
        report.append(f"  Total contaminated samples: {cont_stats['total_contaminated_across_files']}")
        
        if cont_stats['total_samples_across_files'] > 0:
            cont_percentage = (cont_stats['total_contaminated_across_files'] / 
                             cont_stats['total_samples_across_files'] * 100)
            report.append(f"  Contamination rate: {cont_percentage:.2f}%")
        
        # Report contaminated files
        if cont_stats['contaminated_files']:
            report.append(f"\n  ⚠️  CONTAMINATED FILES (config values are parameter names):")
            for file_info in cont_stats['contaminated_files'][:10]:
                report.append(f"    - {file_info['file']}: {file_info['contaminated_count']}/{file_info['total_samples']} "
                            f"samples ({file_info['contamination_rate']:.1f}%)")
            if len(cont_stats['contaminated_files']) > 10:
                report.append(f"    ... and {len(cont_stats['contaminated_files']) - 10} more contaminated files")
            
            report.append(f"\n  ⚠️  RECOMMENDED ACTION: Delete these contaminated files and regenerate data!")
        
        report.append("")
    
    # Duplication analysis
    if analysis_results.get('duplication_stats'):
        dup_stats = analysis_results['duplication_stats']
        report.append("DUPLICATION ANALYSIS:")
        report.append(f"  Files with duplicates: {dup_stats['files_with_duplicates']}/{len(pickle_files)}")
        report.append(f"  Total duplicate samples: {dup_stats['total_duplicates_across_files']}")
        report.append(f"  Total unique configurations: {dup_stats['total_unique_configs_across_files']}")
        
        if dup_stats['total_samples_across_files'] > 0:
            dup_percentage = (dup_stats['total_duplicates_across_files'] / 
                            dup_stats['total_samples_across_files'] * 100)
            report.append(f"  Duplication rate: {dup_percentage:.2f}%")
        
        # Report files with high duplication rates
        high_dup_files = []
        for file_path, file_stats in dup_stats['file_stats'].items():
            if isinstance(file_stats, dict) and file_stats.get('duplicate_count', 0) > 0:
                if file_stats['total_samples'] > 0:
                    dup_rate = file_stats['duplicate_count'] / file_stats['total_samples'] * 100
                    if dup_rate > 10:  # Files with >10% duplication rate
                        file_name = Path(file_path).name
                        high_dup_files.append(f"{file_name} ({dup_rate:.1f}%)")
        
        if high_dup_files:
            report.append(f"  Files with high duplication (>10%): {', '.join(high_dup_files[:5])}")
            if len(high_dup_files) > 5:
                report.append(f"    ... and {len(high_dup_files) - 5} more")
        
        report.append("")
    
    # Out-of-range boundary parameter analysis
    if analysis_results.get('out_of_range_stats'):
        oor_stats = analysis_results['out_of_range_stats']
        report.append("⚠️  OUT-OF-RANGE BOUNDARY PARAMETERS:")
        report.append(f"  Files with out-of-range samples: {oor_stats['files_with_out_of_range_count']}/{len(pickle_files)}")
        report.append(f"  Total samples: {oor_stats['total_samples_across_files']}")
        report.append(f"  Samples within param set ranges: {oor_stats['total_in_range_across_files']}")
        report.append(f"  Samples out-of-range: {oor_stats['total_out_of_range_across_files']}")
        
        if oor_stats['total_samples_across_files'] > 0:
            in_range_percentage = (oor_stats['total_in_range_across_files'] / 
                                  oor_stats['total_samples_across_files'] * 100)
            oor_percentage = (oor_stats['total_out_of_range_across_files'] / 
                            oor_stats['total_samples_across_files'] * 100)
            report.append(f"  In-range rate: {in_range_percentage:.2f}%")
            report.append(f"  Out-of-range rate: {oor_percentage:.2f}%")
        
        # Report files with out-of-range samples
        if oor_stats['files_with_out_of_range']:
            report.append(f"\n  ⚠️  FILES WITH OUT-OF-RANGE SAMPLES (outside param set bounds):")
            for file_info in oor_stats['files_with_out_of_range'][:10]:
                report.append(f"    - {file_info['file']}: {file_info['out_of_range_count']}/{file_info['total_samples']} "
                            f"samples ({file_info['out_of_range_rate']:.1f}%)")
                
                # Show sample details from file_stats
                file_path = file_info['path']
                if file_path in oor_stats['file_stats']:
                    file_details = oor_stats['file_stats'][file_path]
                    for sample_info in file_details.get('out_of_range_samples', [])[:2]:
                        report.append(f"       Sample {sample_info['sample_index']} (param_types={sample_info['param_types']}):")
                        for param_desc in sample_info.get('out_of_range_params', [])[:2]:
                            report.append(f"         • {param_desc}")
            
            if len(oor_stats['files_with_out_of_range']) > 10:
                report.append(f"    ... and {len(oor_stats['files_with_out_of_range']) - 10} more files with out-of-range samples")
            
            report.append(f"\n  ⚠️  RECOMMENDED ACTION: Use 'clean' command to remove these out-of-range samples!")
        
        report.append("")

    # Configuration parameters analysis
    if all_results:
        config_keys = all_results[0].config_keys
        report.append("CONFIGURATION PARAMETERS:")
        report.append(f"  Parameter count: {len(config_keys)}")
        report.append(f"  Parameter names: {', '.join(config_keys)}")
        
        # Show param_types if available
        if hasattr(all_results[0], 'param_types') and all_results[0].param_types:
            report.append(f"  Param sets used: {', '.join(all_results[0].param_types)}")
        
        # Show sample configuration values for first few samples
        report.append("\n  Sample configurations (first 3 samples):")
        for i, result in enumerate(all_results[:3]):
            config_dict = dict(zip(result.config_keys, result.config_values))
            report.append(f"    Sample {i+1}:")
            for key, value in config_dict.items():
                report.append(f"      {key}: {value}")
            if i < 2:  # Add separator between samples
                report.append("")
        
        # Show configuration value ranges
        if len(all_results) > 1:
            report.append("\n  Parameter value ranges:")
            for j, key in enumerate(config_keys):
                values = [result.config_values[j] for result in all_results]
                values_array = np.array(values)
                report.append(f"    {key}: [{np.min(values_array):.6f}, {np.max(values_array):.6f}]")
        
        report.append("")

    # Add more sections as needed...

    report_text = "\n".join(report)
    print("\n" + report_text)
    
    report_file = output_dir / "training_data_summary.txt"
    with open(report_file, 'w') as f:
        f.write(report_text)
    
    print(f"\nReport saved to: {report_file}")
