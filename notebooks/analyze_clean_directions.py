# Eye Diagram Training Data Examination and Cleaning
#
# This script examines, cleans, and analyzes pickle files from the data collector.
# It removes rows with invalid direction block sizes or non-numeric eye-width data.
# Each pickle file contains simulation data for a specific trace SNP file.

# ------------------------------------------------------------
# Imports and setup
# ------------------------------------------------------------
import shutil
import pickle
import argparse
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
warnings.filterwarnings('ignore')

# Direction utilities
from simulation.io.direction_utils import get_valid_block_sizes

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)

# ------------------------------------------------------------
# Helper utilities
# ------------------------------------------------------------
def estimate_block_size(direction_array: np.ndarray) -> int:
    """Estimate the smallest consecutive run length (block size) in a 0/1 array."""
    arr = np.asarray(direction_array).astype(int).flatten()
    if arr.size == 0:
        return 0
    change_idx = np.flatnonzero(np.diff(arr) != 0)
    starts = np.r_[0, change_idx + 1]
    ends = np.r_[change_idx, arr.size - 1]
    run_lengths = ends - starts + 1
    return int(run_lengths.min())

def compute_direction_stats_for_files(pickle_files_list: list[Path]) -> pd.DataFrame:
    """Build a per-sample directions stats dataframe across all files.

    Columns: file, sample_idx, n_lines, block_size_estimate, is_valid_block_size, zeros, ones
    """
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

def clean_pickle_file_inplace(pfile: Path) -> tuple[int, int]:
    """Remove rows with invalid directions or non-numeric eye-widths.

    Returns: (num_samples_before, num_removed)
    Creates a .bak backup next to the original before overwriting.
    """
    with open(pfile, 'rb') as f:
        data = pickle.load(f)

    # Determine number of samples and check for data consistency
    list_keys = [k for k, v in data.items() if isinstance(v, list)]
    if not list_keys:
        return 0, 0
    
    n_samples = len(data[list_keys[0]])
    if n_samples == 0:
        return 0, 0

    for key in list_keys:
        if len(data[key]) != n_samples:
            print(f"  - Warning: inconsistent list lengths in {pfile.name}, skipping.")
            return n_samples, 0

    invalid_indices = set()

    # Check for invalid direction block sizes
    if 'directions' in data:
        for i, dir_arr in enumerate(data['directions']):
            arr = np.asarray(dir_arr).astype(int).flatten()
            if arr.size == 0:
                invalid_indices.add(i)
                continue
            block_est = estimate_block_size(arr)
            valid_sizes = get_valid_block_sizes(arr.size)
            if block_est not in valid_sizes:
                invalid_indices.add(i)

    # Check for non-numeric eye width data
    if 'line_ews' in data:
        for i, ew_val in enumerate(data['line_ews']):
            # If not a list/array or if it contains non-numeric data (like strings)
            if not isinstance(ew_val, (list, np.ndarray)) or \
               (ew_val and isinstance(ew_val[0], str)):
                invalid_indices.add(i)

    num_removed = len(invalid_indices)
    if num_removed == 0:
        return n_samples, 0

    valid_indices = [i for i in range(n_samples) if i not in invalid_indices]

    # Filter all list-based data
    for key in list_keys:
        data[key] = [data[key][i] for i in valid_indices]

    # Backup and overwrite
    backup_path = pfile.with_suffix(pfile.suffix + '.bak')
    try:
        shutil.copy2(pfile, backup_path)
    except Exception:
        pass
    with open(pfile, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    return n_samples, num_removed

def is_numeric_list(data: list) -> bool:
    """Check if a list contains numeric data (non-string)."""
    if not data or not isinstance(data, (list, np.ndarray)):
        return False
    return not isinstance(data[0], str)

# ------------------------------------------------------------
# 1. Load and Examine Pickle Files
# ------------------------------------------------------------
# Configure the path to your pickle files via argv
parser = argparse.ArgumentParser(description="Examine, clean, and analyze training pickle data.")
parser.add_argument("pickle_dir")
parser.add_argument("--clean", action="store_true", help="Perform in-place cleaning of pickle files.")
args = parser.parse_args()
pickle_dir = Path(args.pickle_dir).expanduser()

# Find all pickle files
pickle_files = list(pickle_dir.rglob("*.pkl"))
print(f"Found {len(pickle_files)} pickle files")

# Show first few files
for i, pfile in enumerate(pickle_files[:5]):
    print(f"{i+1}. {pfile.relative_to(pickle_dir)}")

if len(pickle_files) > 5:
    print(f"... and {len(pickle_files) - 5} more files")


# Load and examine the first pickle file
if pickle_files:
    sample_file = pickle_files[0]
    print(f"Examining: {sample_file.name}\n")

    with open(sample_file, 'rb') as f:
        sample_data = pickle.load(f)

    print("Data structure:")
    for key, value in sample_data.items():
        if isinstance(value, list):
            print(f"  {key}: list with {len(value)} items")
            if len(value) > 0:
                print(f"    First item type: {type(value[0])}")
                if key == 'configs' and len(value) > 0:
                    print(f"    Config length: {len(value[0])} parameters")
                elif key == 'line_ews' and len(value) > 0:
                    print(f"    Line EW shape: {np.array(value[0]).shape}")
        else:
            print(f"  {key}: {type(value)}")
else:
    print("No pickle files found. Please check the pickle_dir path.")

# ------------------------------------------------------------
# 2. Data Structure Analysis
# ------------------------------------------------------------
if pickle_files and sample_data:
    print("Detailed data examination:")
    print("=" * 50)

    # Examine configs
    if sample_data['configs']:
        config_array = np.array(sample_data['configs'])
        print(f"\nConfigs:")
        print(f"  Shape: {config_array.shape}")
        print(f"  Min values: {config_array.min(axis=0)[:10]}...")
        print(f"  Max values: {config_array.max(axis=0)[:10]}...")
        print(f"  Mean values: {config_array.mean(axis=0)[:10]}...")

    # Examine line eye widths
    if 'line_ews' in sample_data and sample_data['line_ews']:
        # Filter out non-numeric entries before converting to array
        numeric_line_ews = [item for item in sample_data['line_ews'] if is_numeric_list(item)]
        if numeric_line_ews:
            line_ews_array = np.array(numeric_line_ews)
            print(f"\nLine Eye Widths:")
            print(f"  Shape: {line_ews_array.shape}")
            print(f"  Min: {line_ews_array.min():.3f}")
            print(f"  Max: {line_ews_array.max():.3f}")
            print(f"  Mean: {line_ews_array.mean():.3f}")
            print(f"  Std: {line_ews_array.std():.3f}")
            print(f"  Closed eyes (EW < 0): {(line_ews_array < 0).sum()} / {line_ews_array.size}")
        else:
            print("\nLine Eye Widths: No numeric data found in this sample.")

    # Examine SNP files
    if 'snp_txs' in sample_data and sample_data['snp_txs']:
        unique_tx = set(sample_data['snp_txs'])
        unique_rx = set(sample_data['snp_rxs'])
        print(f"\nSNP Files:")
        print(f"  Unique TX files: {len(unique_tx)}")
        print(f"  Unique RX files: {len(unique_rx)}")
        print(f"  Sample TX: {list(unique_tx)[0] if unique_tx else 'None'}")
        print(f"  Sample RX: {list(unique_rx)[0] if unique_rx else 'None'}")

    # Examine directions
    if sample_data['directions']:
        directions_array = np.array(sample_data['directions'])
        print(f"\nDirections:")
        print(f"  Shape: {directions_array.shape}")
        if directions_array.size > 0:
            print(f"  Unique patterns: {len(set(tuple(row) for row in directions_array))}")
            print(f"  Sample directions: {directions_array[0]}")


# ------------------------------------------------------------
# 3. Summary Statistics Across All Files
# ------------------------------------------------------------
# Aggregate data from all pickle files
all_configs = []
all_line_ews = []
all_directions = []
file_stats = []

print("Loading all pickle files...")
for pfile in pickle_files:
    try:
        with open(pfile, 'rb') as f:
            data = pickle.load(f)

        n_samples = len(data.get('configs', []))
        file_stats.append({
            'file': pfile.name,
            'samples': n_samples,
            'trace_name': pfile.stem,
            'directory': str(pfile.parent.relative_to(pickle_dir)),
        })

        if data.get('configs'):
            all_configs.extend(data['configs'])
        if data.get('line_ews'):
            all_line_ews.extend(data['line_ews'])
        if data.get('directions'):
            all_directions.extend(data['directions'])

    except Exception as e:
        print(f"Error loading {pfile.name}: {e}")

print(f"\nLoaded data from {len(file_stats)} files")
print(f"Total samples: {len(all_configs)}")
print(f"Total eye width measurements: {len(all_line_ews)}")


# Create summary statistics DataFrame
summary_df = pd.DataFrame(file_stats)

if not summary_df.empty:
    print("\n" + "="*50)
    print("--- Per-Directory Report ---")
    per_dir_summary = summary_df.groupby('directory').agg(
        file_count=('file', 'count'),
        total_samples=('samples', 'sum')
    ).reset_index().sort_values('directory')
    print(per_dir_summary.to_string(index=False))
    print("="*50)

    print("\n--- Per-Directory Sample-Count Distribution ---")
    for directory, group in summary_df.groupby('directory'):
        print(f"\nDirectory: {directory}")
        sample_counts = group['samples'].value_counts().sort_index()
        if sample_counts.empty:
            print("  No samples found.")
            continue
        for num_samples, num_files in sample_counts.items():
            print(f"  - {num_samples} samples: {num_files} files")
    print("="*50)


print(f"\nTotal files: {len(summary_df)}")
print(f"Total samples: {summary_df['samples'].sum()}")
print(f"Average samples per file: {summary_df['samples'].mean():.1f}")
print(f"Min samples: {summary_df['samples'].min()}")
print(f"Max samples: {summary_df['samples'].max()}")


# ------------------------------------------------------------
# 4. Directions Distribution and Block Size Analysis (Before Cleaning)
# ------------------------------------------------------------
print("\nDirections Analysis (BEFORE cleaning):")
print("=" * 50)
df_dirs_before = compute_direction_stats_for_files(pickle_files)
if not df_dirs_before.empty:
    print(f"Samples with directions: {len(df_dirs_before)}")
    print(f"Unique n_lines: {sorted(df_dirs_before['n_lines'].unique().tolist())}")
    print("Estimated block size frequency (top 10):")
    print(df_dirs_before['block_size_estimate'].value_counts().head(10))
    print(f"\nInvalid block size estimates (should be 0): {(~df_dirs_before['is_valid_block_size']).sum()}")

    top_groups_before = (
        df_dirs_before.groupby(['n_lines', 'block_size_estimate'])
                      .size()
                      .reset_index(name='count')
                      .sort_values(['n_lines', 'count'], ascending=[True, False])
    )
    print("\nBlock size counts by n_lines (first 20 rows):")
    print(top_groups_before.head(20).to_string(index=False))

    df_dirs_before.to_csv('directions_block_sizes_before.csv', index=False)
    print("Saved: directions_block_sizes_before.csv")
else:
    print("No directions data found in pickle files.")

# ------------------------------------------------------------
# 4b. Clean pickle files in-place (remove invalid-direction rows only)
# ------------------------------------------------------------
if args.clean:
    print("\nCleaning pickle files (removing rows with invalid directions)...")
    total_before = 0
    total_removed = 0
    for pfile in pickle_files:
        try:
            n_before, n_removed = clean_pickle_file_inplace(pfile)
            total_before += n_before
            total_removed += n_removed
            if n_removed > 0:
                print(f"Cleaned {pfile.name}: removed {n_removed}/{n_before}")
        except Exception as e:
            print(f"Error cleaning {pfile.name}: {e}")
    print(f"Total rows before: {total_before}; total removed: {total_removed}")

    # ------------------------------------------------------------
    # 4c. Directions Distribution and Block Size Analysis (After Cleaning)
    # ------------------------------------------------------------
    print("\nDirections Analysis (AFTER cleaning):")
    print("=" * 50)
    pickle_files_after = list(pickle_dir.rglob("*.pkl"))
    df_dirs_after = compute_direction_stats_for_files(pickle_files_after)
    if not df_dirs_after.empty:
        print(f"Samples with directions: {len(df_dirs_after)}")
        print(f"Unique n_lines: {sorted(df_dirs_after['n_lines'].unique().tolist())}")
        print("Estimated block size frequency (top 10):")
        print(df_dirs_after['block_size_estimate'].value_counts().head(10))
        print(f"\nInvalid block size estimates (should be 0): {(~df_dirs_after['is_valid_block_size']).sum()}")

        top_groups_after = (
            df_dirs_after.groupby(['n_lines', 'block_size_estimate'])
                         .size()
                         .reset_index(name='count')
                         .sort_values(['n_lines', 'count'], ascending=[True, False])
        )
        print("\nBlock size counts by n_lines (first 20 rows):")
        print(top_groups_after.head(20).to_string(index=False))

        df_dirs_after.to_csv('directions_block_sizes_after.csv', index=False)
        print("Saved: directions_block_sizes_after.csv")
    else:
        print("No directions data found in pickle files after cleaning.")
else:
    total_before, total_removed = 0, 0 # Ensure these exist for the report


# ------------------------------------------------------------
# 5. Eye Width Distribution Analysis
# ------------------------------------------------------------
if all_line_ews:
    # Convert to numpy array for analysis
    line_ews_array = np.array(all_line_ews)

    print(f"Eye Width Analysis:")
    print(f"  Total measurements: {line_ews_array.size}")
    print(f"  Shape: {line_ews_array.shape}")
    print(f"  Min: {line_ews_array.min():.3f}")
    print(f"  Max: {line_ews_array.max():.3f}")
    print(f"  Mean: {line_ews_array.mean():.3f}")
    print(f"  Median: {np.median(line_ews_array):.3f}")
    print(f"  Std: {line_ews_array.std():.3f}")

    # Analyze closed eyes
    closed_eyes = line_ews_array < 0
    print(f"  Closed eyes (EW < 0): {closed_eyes.sum()} ({closed_eyes.mean()*100:.1f}%)")

    # Analyze very wide eyes (might be artifacts)
    wide_eyes = line_ews_array > 95
    print(f"  Very wide eyes (EW > 95): {wide_eyes.sum()} ({wide_eyes.mean()*100:.1f}%)")

    # Plot distributions
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Overall distribution
    axes[0,0].hist(line_ews_array.flatten(), bins=50, alpha=0.7, edgecolor='black')
    axes[0,0].set_title('Overall Eye Width Distribution')
    axes[0,0].set_xlabel('Eye Width')
    axes[0,0].set_ylabel('Frequency')
    axes[0,0].axvline(line_ews_array.mean(), color='red', linestyle='--', label=f'Mean: {line_ews_array.mean():.2f}')
    axes[0,0].legend()

    # Distribution excluding closed eyes
    open_eyes = line_ews_array[line_ews_array >= 0]
    if len(open_eyes) > 0:
        axes[0,1].hist(open_eyes.flatten(), bins=50, alpha=0.7, edgecolor='black', color='green')
        axes[0,1].set_title('Eye Width Distribution (Open Eyes Only)')
        axes[0,1].set_xlabel('Eye Width')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].axvline(open_eyes.mean(), color='red', linestyle='--', label=f'Mean: {open_eyes.mean():.2f}')
        axes[0,1].legend()

    # Box plot by line (if multiple lines)
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
    plt.show()
else:
    print("No eye width data found in pickle files.")

# ------------------------------------------------------------
# 6. Data Quality Checks and Summary
# ------------------------------------------------------------
print("Data Quality Checks:")
print("=" * 30)

# Check for consistent data lengths
inconsistent_files = []
for pfile in pickle_files:
    try:
        with open(pfile, 'rb') as f:
            data = pickle.load(f)

        lengths = {key: len(val) for key, val in data.items() if isinstance(val, list)}
        if len(set(lengths.values())) > 1:
            inconsistent_files.append((pfile.name, lengths))
    except Exception as e:
        print(f"Error checking {pfile.name}: {e}")

if inconsistent_files:
    print(f"\n⚠️  Found {len(inconsistent_files)} files with inconsistent data lengths:")
    for fname, lengths in inconsistent_files:
        print(f"  {fname}: {lengths}")
else:
    print("✅ All files have consistent data lengths")

# Check for missing or corrupted data
if all_line_ews:
    line_ews_array = np.array(all_line_ews)
    nan_count = np.isnan(line_ews_array).sum()
    inf_count = np.isinf(line_ews_array).sum()

    print(f"\nData integrity:")
    print(f"  NaN values: {nan_count}")
    print(f"  Infinite values: {inf_count}")
    print(f"  Values < -1 (suspicious): {(line_ews_array < -1).sum()}")
    print(f"  Values > 100 (suspicious): {(line_ews_array > 100).sum()}")

# File size analysis
file_sizes = [pfile.stat().st_size / 1024 for pfile in pickle_files]  # KB
if file_sizes:
    print(f"\nFile sizes:")
    print(f"  Average: {np.mean(file_sizes):.1f} KB")
    print(f"  Min: {np.min(file_sizes):.1f} KB")
    print(f"  Max: {np.max(file_sizes):.1f} KB")
    print(f"  Total: {np.sum(file_sizes):.1f} KB ({np.sum(file_sizes)/1024:.1f} MB)")


# Generate and save summary report
report = []
report.append("EYE DIAGRAM TRAINING DATA SUMMARY REPORT")
report.append("=" * 50)
report.append(f"Generated: {pd.Timestamp.now()}")
report.append(f"Data directory: {pickle_dir}")
report.append("")

# Dataset overview
report.append("DATASET OVERVIEW:")
report.append(f"  Total pickle files: {len(pickle_files)}")
report.append(f"  Total samples: {len(all_configs)}")
if len(all_configs) > 0:
    report.append(f"  Parameters per sample: {len(all_configs[0])}")
if len(all_line_ews) > 0:
    line_ews_array = np.array(all_line_ews)
    report.append(f"  Lines per sample: {line_ews_array.shape[1] if line_ews_array.ndim > 1 else 1}")

report.append("")

# Eye width statistics
if len(all_line_ews) > 0:
    line_ews_array = np.array(all_line_ews)
    report.append("EYE WIDTH STATISTICS:")
    report.append(f"  Mean: {line_ews_array.mean():.3f}")
    report.append(f"  Std: {line_ews_array.std():.3f}")
    report.append(f"  Min: {line_ews_array.min():.3f}")
    report.append(f"  Max: {line_ews_array.max():.3f}")
    report.append(f"  Closed eyes: {(line_ews_array < 0).sum()} ({(line_ews_array < 0).mean()*100:.1f}%)")
    report.append("")

# File statistics
if len(file_stats) > 0:
    summary_df = pd.DataFrame(file_stats)
    report.append("FILE STATISTICS:")
    report.append(f"  Average samples per file: {summary_df['samples'].mean():.1f}")
    report.append(f"  Min samples per file: {summary_df['samples'].min()}")
    report.append(f"  Max samples per file: {summary_df['samples'].max()}")
    report.append("")

# Directions summary (before/after)
if 'df_dirs_before' in locals() and isinstance(df_dirs_before, pd.DataFrame) and not df_dirs_before.empty:
    report.append("DIRECTIONS BLOCK SIZE SUMMARY (BEFORE cleaning):")
    vc_b = df_dirs_before['block_size_estimate'].value_counts()
    for bs, cnt in vc_b.head(5).items():
        report.append(f"  Block size {int(bs)}: {int(cnt)} samples")
    invalid_b = int((~df_dirs_before['is_valid_block_size']).sum())
    report.append(f"  Invalid block size estimates: {invalid_b}")
    report.append("")

if args.clean:
    if 'df_dirs_after' in locals() and isinstance(df_dirs_after, pd.DataFrame) and not df_dirs_after.empty:
        report.append("DIRECTIONS BLOCK SIZE SUMMARY (AFTER cleaning):")
        vc_a = df_dirs_after['block_size_estimate'].value_counts()
        for bs, cnt in vc_a.head(5).items():
            report.append(f"  Block size {int(bs)}: {int(cnt)} samples")
        invalid_a = int((~df_dirs_after['is_valid_block_size']).sum())
        report.append(f"  Invalid block size estimates: {invalid_a}")
        report.append("")

    report.append(f"CLEANING SUMMARY: removed {total_removed} invalid rows out of {total_before} before-clean rows")

# Print and save report
report_text = "\n".join(report)
print(report_text)

# Save to file
report_file = Path("training_data_summary.txt")
with open(report_file, 'w') as f:
    f.write(report_text)

print(f"\nReport saved to: {report_file}")