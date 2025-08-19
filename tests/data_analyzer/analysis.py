import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List
from simulation.io.pickle_utils import SimulationResult

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

from .cleaning import estimate_block_size

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

    # Add more sections as needed...

    report_text = "\n".join(report)
    print("\n" + report_text)
    
    report_file = output_dir / "training_data_summary.txt"
    with open(report_file, 'w') as f:
        f.write(report_text)
    
    print(f"\nReport saved to: {report_file}")
