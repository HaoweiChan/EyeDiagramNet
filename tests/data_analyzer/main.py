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
        
        analysis_results = {
            'file_stats': file_stats,
            'contamination_stats': contamination_stats,
            'duplication_stats': duplication_stats
        }
        analysis.plot_eye_width_distributions(all_results, output_dir)
        analysis.generate_summary_report(pickle_dir, pickle_files, all_results, analysis_results, output_dir)

    elif args.command == "clean":
        print("\nCleaning pickle files...")
        print("By default, contaminated samples (config values are strings) will be removed.")
        print("Use --keep-contaminated to preserve them for inspection.\n")
        
        total_before, total_removed = 0, 0
        remove_contaminated = not args.keep_contaminated  # Default is to remove contaminated
        
        for pfile in pickle_files:
            try:
                n_before, n_removed = cleaning.clean_pickle_file_inplace(
                    pfile, 
                    block_size=args.block_size, 
                    remove_block_size_1=args.remove_block_size_1,
                    remove_duplicates=args.remove_duplicates,
                    remove_contaminated=remove_contaminated
                )
                total_before += n_before
                total_removed += n_removed
                if n_removed > 0:
                    print(f"Cleaned {pfile.name}: removed {n_removed}/{n_before} samples.")
            except Exception as e:
                print(f"Error cleaning {pfile.name}: {e}")
        
        print(f"\n=== Cleaning Summary ===")
        print(f"Total samples before: {total_before}")
        print(f"Total samples removed: {total_removed}")
        print(f"Total samples remaining: {total_before - total_removed}")
        if total_removed > 0:
            print(f"Removal rate: {(total_removed / total_before * 100):.1f}%")
            print(f"\n✅ Cleaning complete! Backups saved with .bak extension.")

    elif args.command == "validate":
        validation.run_validation(pickle_files, max_files, max_samples, output_dir)

if __name__ == '__main__':
    main()
