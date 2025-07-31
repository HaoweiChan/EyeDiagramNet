#!/usr/bin/env python3
"""
Clean pickle files by removing samples with block size 1 direction patterns.

This script traverses a directory to find all pickle files and detects samples
that have direction patterns with block size 1 (odd numbers of consecutive 0s or 1s).
These samples are contaminated from the old collection process and should be removed.

Usage:
    python clean_pickle_directions.py <directory> [--clean]
"""

import argparse
import pickle
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any
import sys


def detect_block_size_1_patterns(directions: List[int]) -> bool:
    """
    Detect if a direction pattern contains block size 1 (odd consecutive 0s or 1s).
    
    Args:
        directions: List of direction values (0s and 1s)
        
    Returns:
        True if the pattern contains block size 1, False otherwise
    """
    if not directions:
        return False
    
    # Convert to numpy array for easier processing
    directions = np.array(directions)
    
    # Find transitions (where values change)
    transitions = np.diff(directions)
    change_indices = np.where(transitions != 0)[0]
    
    # Add start and end indices
    all_indices = np.concatenate([[-1], change_indices, [len(directions) - 1]])
    
    # Calculate block lengths
    block_lengths = np.diff(all_indices)
    
    # Check if any block has length 1 (single consecutive value)
    return np.any(block_lengths == 1)


def analyze_pickle_file(file_path: Path, clean: bool = False) -> Dict[str, Any]:
    """
    Analyze a pickle file for block size 1 patterns and optionally clean them.
    
    Args:
        file_path: Path to the pickle file
        clean: If True, actually remove the contaminated samples
        
    Returns:
        Dictionary with analysis results
    """
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        # Check if this is a valid pickle file with the expected structure
        if not isinstance(data, dict):
            return {
                'file': str(file_path),
                'error': 'Not a dictionary',
                'contaminated_samples': [],
                'total_samples': 0,
                'cleaned': False
            }
        
        # Check for required keys
        required_keys = ['directions', 'configs', 'line_ews']
        missing_keys = [key for key in required_keys if key not in data]
        if missing_keys:
            return {
                'file': str(file_path),
                'error': f'Missing required keys: {missing_keys}',
                'contaminated_samples': [],
                'total_samples': 0,
                'cleaned': False
            }
        
        directions_list = data['directions']
        configs_list = data['configs']
        line_ews_list = data['line_ews']
        
        # Verify all lists have the same length
        if not (len(directions_list) == len(configs_list) == len(line_ews_list)):
            return {
                'file': str(file_path),
                'error': f'Inconsistent list lengths: directions={len(directions_list)}, configs={len(configs_list)}, line_ews={len(line_ews_list)}',
                'contaminated_samples': [],
                'total_samples': len(directions_list),
                'cleaned': False
            }
        
        total_samples = len(directions_list)
        contaminated_indices = []
        
        # Check each sample for block size 1 patterns
        for i, directions in enumerate(directions_list):
            if detect_block_size_1_patterns(directions):
                contaminated_indices.append(i)
        
        result = {
            'file': str(file_path),
            'error': None,
            'contaminated_samples': contaminated_indices,
            'total_samples': total_samples,
            'cleaned': False
        }
        
        # Clean the file if requested and contaminated samples found
        if clean and contaminated_indices:
            try:
                # Remove contaminated samples (in reverse order to maintain indices)
                for idx in reversed(contaminated_indices):
                    del directions_list[idx]
                    del configs_list[idx]
                    del line_ews_list[idx]
                
                # Update the data dictionary
                data['directions'] = directions_list
                data['configs'] = configs_list
                data['line_ews'] = line_ews_list
                
                # Write the cleaned data back to file
                with open(file_path, 'wb') as f:
                    pickle.dump(data, f)
                
                result['cleaned'] = True
                result['remaining_samples'] = len(directions_list)
                
            except Exception as e:
                result['error'] = f'Failed to clean file: {e}'
        
        return result
        
    except Exception as e:
        return {
            'file': str(file_path),
            'error': f'Failed to process file: {e}',
            'contaminated_samples': [],
            'total_samples': 0,
            'cleaned': False
        }


def find_pickle_files(directory: Path) -> List[Path]:
    """
    Recursively find all pickle files in the given directory.
    
    Args:
        directory: Root directory to search
        
    Returns:
        List of pickle file paths
    """
    pickle_files = []
    
    if not directory.exists():
        print(f"Error: Directory {directory} does not exist")
        return pickle_files
    
    # Find all .pkl files recursively
    for pkl_file in directory.rglob("*.pkl"):
        pickle_files.append(pkl_file)
    
    return sorted(pickle_files)

def print_analysis_summary(results: List[Dict[str, Any]], clean: bool):
    """
    Print a summary of the analysis results.
    
    Args:
        results: List of analysis results
        clean: Whether cleaning was performed
    """
    print("\n" + "="*80)
    print("PICKLE FILE DIRECTION CLEANING ANALYSIS")
    print("="*80)
    
    total_files = len(results)
    processed_files = 0
    error_files = 0
    contaminated_files = 0
    total_contaminated_samples = 0
    total_samples = 0
    cleaned_files = 0
    
    # Process results
    for result in results:
        processed_files += 1
        total_samples += result['total_samples']
        
        if result['error']:
            error_files += 1
            print(f"❌ {Path(result['file']).name}: {result['error']}")
        else:
            contaminated_count = len(result['contaminated_samples'])
            total_contaminated_samples += contaminated_count
            
            if contaminated_count > 0:
                contaminated_files += 1
                if result['cleaned']:
                    cleaned_files += 1
                    print(f"🧹 {Path(result['file']).name}: {contaminated_count}/{result['total_samples']} samples cleaned")
                else:
                    print(f"⚠️  {Path(result['file']).name}: {contaminated_count}/{result['total_samples']} contaminated samples found")
            else:
                print(f"✅ {Path(result['file']).name}: No contaminated samples")
    
    # Print summary statistics
    print("\n" + "-"*80)
    print("SUMMARY STATISTICS")
    print("-"*80)
    print(f"Total files processed: {processed_files}")
    print(f"Files with errors: {error_files}")
    print(f"Files with contaminated samples: {contaminated_files}")
    print(f"Total samples: {total_samples}")
    print(f"Contaminated samples: {total_contaminated_samples}")
    print(f"Contamination rate: {total_contaminated_samples/total_samples*100:.2f}%" if total_samples > 0 else "N/A")
    
    if clean:
        print(f"Files cleaned: {cleaned_files}")
        print(f"Remaining samples after cleaning: {total_samples - total_contaminated_samples}")
    else:
        print(f"Files that would be cleaned: {contaminated_files}")
        print(f"Samples that would be removed: {total_contaminated_samples}")
    
    # Print recommendations
    print("\n" + "-"*80)
    print("RECOMMENDATIONS")
    print("-"*80)
    
    if total_contaminated_samples > 0:
        if clean:
            print("✅ Cleaning completed successfully!")
            print("📊 The contaminated samples have been removed from the pickle files.")
        else:
            print("⚠️  Contaminated samples detected!")
            print("🔧 Run with --clean flag to remove contaminated samples.")
            print("📝 Contaminated samples have block size 1 patterns (odd consecutive 0s or 1s).")
    else:
        print("✅ No contaminated samples found!")
        print("🎉 All pickle files are clean and ready for use.")

def main():
    """Main function to run the pickle cleaning analysis."""
    parser = argparse.ArgumentParser(
        description='Detect and clean pickle files with contaminated direction patterns',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze files without cleaning
  python clean_pickle_directions.py ./data/training_data/
  
  # Clean contaminated files
  python clean_pickle_directions.py ./data/training_data/ --clean
  
  # Analyze specific directory
  python clean_pickle_directions.py ./test_data/eye_width/local_traces/
        """
    )
    
    parser.add_argument('directory', type=str, help='Directory to search for pickle files')
    parser.add_argument('--clean', action='store_true', 
                       help='Actually remove contaminated samples (default: dry run)')
    
    args = parser.parse_args()
    
    # Convert directory to Path
    directory = Path(args.directory)
    
    print("🔍 Searching for pickle files...")
    pickle_files = find_pickle_files(directory)
    
    if not pickle_files:
        print(f"❌ No pickle files found in {directory}")
        sys.exit(1)
    
    print(f"📁 Found {len(pickle_files)} pickle files")
    
    # Analyze each file
    results = []
    for i, pkl_file in enumerate(pickle_files, 1):
        print(f"🔬 Analyzing {i}/{len(pickle_files)}: {pkl_file.name}")
        result = analyze_pickle_file(pkl_file, clean=args.clean)
        results.append(result)
    
    # Print summary
    print_analysis_summary(results, args.clean)
    
    # Exit with appropriate code
    total_contaminated = sum(len(r['contaminated_samples']) for r in results if r['error'] is None)
    if total_contaminated > 0 and not args.clean:
        print("\n⚠️  Contaminated samples found! Run with --clean to remove them.")
        sys.exit(1)
    elif total_contaminated == 0:
        print("\n✅ No contaminated samples found!")
        sys.exit(0)
    else:
        print("\n✅ Cleaning completed successfully!")
        sys.exit(0)

if __name__ == "__main__":
    main() 