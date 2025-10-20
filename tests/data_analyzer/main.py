import argparse
import pickle
from pathlib import Path
from datetime import datetime

from . import analysis, cleaning, validation
from common.pickle_utils import load_pickle_data


def main():
    parser = argparse.ArgumentParser(description="A comprehensive tool to analyze, clean, and validate training pickle data.")
    parser.add_argument("command", type=str, choices=["analyze", "clean", "validate"], help="Available commands")
    parser.add_argument("pickle_dir", type=str, help="Path to the directory containing pickle files.")

    parser.add_argument("--block_size", type=int, help="Only keep samples with this block size (clean command only).")
    parser.add_argument("--remove-block-size-1", action="store_true", help="Remove samples with block size 1 direction patterns (clean command only).")
    parser.add_argument("--remove-duplicates", action="store_true", help="Remove samples with duplicate configuration values (clean command only).")
    parser.add_argument("--remove-legacy", action="store_true", help="Remove samples using legacy parameter naming (snp_tx/snp_rx) (clean command only).")
    parser.add_argument("--keep-contaminated", action="store_true", help="Keep contaminated samples (clean command only, default: remove contaminated).")
    parser.add_argument("--max_files", type=int, help="Max number of files to process (analyze/validate commands only).")
    parser.add_argument("--max_samples", type=int, help="Max number of samples per file (analyze/validate commands only).")

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

    # Set defaults based on command
    max_files = getattr(args, 'max_files', 3) or 3
    max_samples = getattr(args, 'max_samples', 5) or 5

    if args.command == "analyze":
        # Load all data for analysis using the new dataclass loader
        all_results = []
        file_stats = []
        for pfile in pickle_files:
            results = load_pickle_data(pfile)
            if results:
                all_results.extend(results)
                file_stats.append({'file': pfile.name, 'samples': len(results)})
        
        # Analyze contamination across all files (CRITICAL)
        print("Analyzing contaminated configs...")
        contamination_stats = analysis.analyze_contamination_across_files(pickle_files)
        
        # Analyze duplications across all files
        print("Analyzing duplications...")
        duplication_stats = analysis.analyze_duplications_across_files(pickle_files)
        
        # Analyze out-of-range boundary parameters
        print("Analyzing out-of-range boundary parameters...")
        out_of_range_stats = analysis.analyze_out_of_range_across_files(pickle_files)
        
        analysis_results = {
            'file_stats': file_stats,
            'contamination_stats': contamination_stats,
            'duplication_stats': duplication_stats,
            'out_of_range_stats': out_of_range_stats
        }
        analysis.plot_eye_width_distributions(all_results, output_dir)
        analysis.generate_summary_report(pickle_dir, pickle_files, all_results, analysis_results, output_dir)

    elif args.command == "clean":
        print("\nCleaning pickle files...")
        print("By default, contaminated samples (config values are strings) will be removed.")
        print("Boundary parameters will be validated against param sets defined in each file.")
        print("Use --keep-contaminated to preserve them for inspection.\n")
        
        total_before, total_removed, total_out_of_range = 0, 0, 0
        remove_contaminated = not args.keep_contaminated  # Default is to remove contaminated
        all_out_of_range_details = []
        all_param_types = set()  # Track all param_types seen
        
        total_legacy = 0
        
        for pfile in pickle_files:
            try:
                n_before, n_removed, stats = cleaning.clean_pickle_file_inplace(
                    pfile, 
                    block_size=args.block_size, 
                    remove_block_size_1=args.remove_block_size_1,
                    remove_duplicates=args.remove_duplicates,
                    remove_contaminated=remove_contaminated,
                    remove_legacy=args.remove_legacy
                )
                total_before += n_before
                total_removed += n_removed
                total_out_of_range += stats['out_of_range_count']
                total_legacy += stats['legacy_count']
                
                # Collect param_types for reporting
                if 'param_types_seen' in stats:
                    all_param_types.update(stats['param_types_seen'])
                
                if n_removed > 0:
                    print(f"Cleaned {pfile.name}: removed {n_removed}/{n_before} samples.")
                    # Show param_types for this file if available
                    if 'param_types_seen' in stats and stats['param_types_seen']:
                        print(f"  └─ Param sets: {', '.join(sorted(stats['param_types_seen']))}")
                
                # Show out-of-range details if any
                if stats['out_of_range_count'] > 0:
                    print(f"  └─ Out-of-range samples: {stats['out_of_range_count']}")
                    for idx, param_types, out_of_range_params in stats['out_of_range_details'][:3]:
                        print(f"     Sample {idx} (param_types={param_types}):")
                        for param_desc in out_of_range_params[:2]:
                            print(f"       - {param_desc}")
                    if len(stats['out_of_range_details']) > 3:
                        print(f"     ... and {len(stats['out_of_range_details']) - 3} more")
                    all_out_of_range_details.append((pfile.name, stats['out_of_range_details']))
                
                # Show legacy removal count if any
                if stats['legacy_count'] > 0:
                    print(f"  └─ Legacy format samples: {stats['legacy_count']}")
            except Exception as e:
                print(f"Error cleaning {pfile.name}: {e}")
        
        total_in_range = total_before - total_out_of_range - total_legacy
        
        print(f"\n=== Cleaning Summary ===")
        if all_param_types:
            print(f"Param sets found across all files: {', '.join(sorted(all_param_types))}")
        print(f"Total samples before: {total_before}")
        print(f"Samples within param set ranges: {total_in_range}")
        print(f"Total samples removed: {total_removed}")
        print(f"  └─ Out-of-range samples: {total_out_of_range}")
        if args.remove_legacy:
            print(f"  └─ Legacy format samples: {total_legacy}")
        print(f"Total samples remaining: {total_before - total_removed}")
        if total_removed > 0:
            in_range_rate = (total_in_range / total_before * 100) if total_before > 0 else 0
            removal_rate = (total_removed / total_before * 100) if total_before > 0 else 0
            print(f"In-range rate: {in_range_rate:.1f}%")
            print(f"Removal rate: {removal_rate:.1f}%")
            print(f"\n✅ Cleaning complete! Backups saved with .bak extension.")

    elif args.command == "validate":
        validation.run_validation(pickle_files, max_files, max_samples, output_dir)

if __name__ == '__main__':
    main()
