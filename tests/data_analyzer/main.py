#!/usr/bin/env python3
"""
A comprehensive tool to analyze, clean, and validate training pickle data.

Usage:
    python tests/data_analyzer/main.py {analyze,clean,validate} <pickle_dir> [options]
    python -m tests.data_analyzer.main {analyze,clean,validate} <pickle_dir> [options]

Commands:
    analyze     Perform a full analysis of the pickle data.
    clean       Clean the pickle files in-place.
    validate    Validate pickle data against simulation.

For more detailed help on each command, run:
    python -m tests.data_analyzer.main {analyze,clean,validate} --help
"""

import argparse
import pickle
import sys
from pathlib import Path
from datetime import datetime

# Add the project root to the Python path if not already there
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Use absolute imports when running as main script, relative when imported as module
if __name__ == "__main__":
    from tests.data_analyzer import analysis, cleaning, validation
else:
    from . import analysis, cleaning, validation

from common.pickle_utils import load_pickle_data

def main():
    parser = argparse.ArgumentParser(description="A comprehensive tool to analyze, clean, and validate training pickle data.")
    
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    # --- Analyze Command ---
    parser_analyze = subparsers.add_parser("analyze", help="Perform a full analysis of the pickle data.")
    parser_analyze.add_argument("pickle_dir", type=str, help="Path to the directory containing pickle files.")
    
    # --- Clean Command ---
    parser_clean = subparsers.add_parser("clean", help="Clean the pickle files in-place.")
    parser_clean.add_argument("pickle_dir", type=str, help="Path to the directory containing pickle files.")
    parser_clean.add_argument("--block_size", type=int, help="Only keep samples with this block size.")
    parser_clean.add_argument("--remove-block-size-1", action="store_true", help="Remove samples with block size 1 direction patterns.")
    parser_clean.add_argument("--remove-duplicates", action="store_true", help="Remove samples with duplicate configuration values (keeps first occurrence).")

    # --- Validate Command ---
    parser_validate = subparsers.add_parser("validate", help="Validate pickle data against simulation.")
    parser_validate.add_argument("pickle_dir", type=str, help="Path to the directory containing pickle files.")
    parser_validate.add_argument("--max_files", type=int, default=3, help="Max number of files to validate.")
    parser_validate.add_argument("--max_samples", type=int, default=5, help="Max number of samples per file.")

    args = parser.parse_args()
    
    pickle_dir = Path(args.pickle_dir).expanduser()
    if not pickle_dir.is_dir():
        print(f"Error: Provided path '{pickle_dir}' is not a valid directory.")
        return

    date_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path("tests") / f'analyzer_output_{date_str}'
    output_dir.mkdir(exist_ok=True)
    print(f"Outputs will be saved to: {output_dir}")

    pickle_files = list(pickle_dir.rglob("*.pkl"))
    print(f"Found {len(pickle_files)} pickle files in {pickle_dir}")
    if not pickle_files:
        return

    if args.command == "analyze":
        # Load all data for analysis using the new dataclass loader
        all_results = []
        file_stats = []
        for pfile in pickle_files:
            results = load_pickle_data(pfile)
            if results:
                all_results.extend(results)
                file_stats.append({'file': pfile.name, 'samples': len(results)})
        
        # Analyze duplications across all files
        print("Analyzing duplications...")
        duplication_stats = analysis.analyze_duplications_across_files(pickle_files)
        
        analysis_results = {
            'file_stats': file_stats,
            'duplication_stats': duplication_stats
        }
        analysis.plot_eye_width_distributions(all_results, output_dir)
        analysis.generate_summary_report(pickle_dir, pickle_files, all_results, analysis_results, output_dir)

    elif args.command == "clean":
        print("\n" + "="*60)
        print("RUNNING PRE-CLEANING ANALYSIS")
        print("="*60)
        
        # Run analysis before cleaning
        all_results_before = []
        for pfile in pickle_files:
            results = load_pickle_data(pfile)
            if results:
                all_results_before.extend(results)
        
        duplication_stats_before = analysis.analyze_duplications_across_files(pickle_files)
        print(f"BEFORE CLEANING:")
        print(f"  Total samples: {duplication_stats_before['total_samples_across_files']}")
        print(f"  Unique configurations: {duplication_stats_before['total_unique_configs_across_files']}")
        print(f"  Total duplicates: {duplication_stats_before['total_duplicates_across_files']}")
        if duplication_stats_before['total_samples_across_files'] > 0:
            dup_rate_before = (duplication_stats_before['total_duplicates_across_files'] / 
                              duplication_stats_before['total_samples_across_files'] * 100)
            print(f"  Duplication rate: {dup_rate_before:.2f}%")
        print(f"  Files with duplicates: {duplication_stats_before['files_with_duplicates']}/{len(pickle_files)}")
        
        print("\n" + "="*60)
        print("CLEANING PICKLE FILES")
        print("="*60)
        
        total_before, total_removed = 0, 0
        for pfile in pickle_files:
            try:
                n_before, n_removed = cleaning.clean_pickle_file_inplace(
                    pfile, 
                    block_size=args.block_size, 
                    remove_block_size_1=args.remove_block_size_1,
                    remove_duplicates=args.remove_duplicates
                )
                total_before += n_before
                total_removed += n_removed
                if n_removed > 0:
                    print(f"Cleaned {pfile.name}: removed {n_removed}/{n_before} samples.")
            except Exception as e:
                print(f"Error cleaning {pfile.name}: {e}")
        
        print(f"\nCleaning complete. Total rows before: {total_before}, total removed: {total_removed}")
        
        print("\n" + "="*60)
        print("RUNNING POST-CLEANING ANALYSIS")
        print("="*60)
        
        # Run analysis after cleaning
        all_results_after = []
        for pfile in pickle_files:
            results = load_pickle_data(pfile)
            if results:
                all_results_after.extend(results)
        
        duplication_stats_after = analysis.analyze_duplications_across_files(pickle_files)
        print(f"AFTER CLEANING:")
        print(f"  Total samples: {duplication_stats_after['total_samples_across_files']}")
        print(f"  Unique configurations: {duplication_stats_after['total_unique_configs_across_files']}")
        print(f"  Total duplicates: {duplication_stats_after['total_duplicates_across_files']}")
        if duplication_stats_after['total_samples_across_files'] > 0:
            dup_rate_after = (duplication_stats_after['total_duplicates_across_files'] / 
                             duplication_stats_after['total_samples_across_files'] * 100)
            print(f"  Duplication rate: {dup_rate_after:.2f}%")
        print(f"  Files with duplicates: {duplication_stats_after['files_with_duplicates']}/{len(pickle_files)}")
        
        print("\n" + "="*60)
        print("CLEANING SUMMARY")
        print("="*60)
        samples_removed = duplication_stats_before['total_samples_across_files'] - duplication_stats_after['total_samples_across_files']
        duplicates_removed = duplication_stats_before['total_duplicates_across_files'] - duplication_stats_after['total_duplicates_across_files']
        print(f"  Samples removed: {samples_removed}")
        print(f"  Duplicates removed: {duplicates_removed}")
        if duplication_stats_before['total_samples_across_files'] > 0 and duplication_stats_after['total_samples_across_files'] > 0:
            dup_reduction = dup_rate_before - dup_rate_after
            print(f"  Duplication rate reduction: {dup_reduction:.2f} percentage points")
        print("="*60)

    elif args.command == "validate":
        validation.run_validation(pickle_files, args.max_files, args.max_samples, output_dir)

if __name__ == "__main__":
    main()
