import csv
import re
from pathlib import Path

def create_snp_index_csv(base_dir: Path):
    """
    Scans a directory for .s96p files matching a specific pattern,
    sorts them by seed number, and writes an index CSV file.
    """
    # Pattern to match the snp files
    pattern = re.compile(r'transmission_lines_48lines_seed(\d+)\.s96p$')

    # List and sort the files by seed number
    snp_files = []
    for f in base_dir.iterdir():
        m = pattern.match(f.name)
        if m:
            snp_files.append((int(m.group(1)), f.name))

    snp_files.sort()

    # Write to CSV
    output_path = base_dir / 'snp_index.csv'
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['index', 'snp_file_name'])
        for idx, fname in snp_files:
            writer.writerow([idx, fname])

    print(f"Wrote {len(snp_files)} entries to {output_path}")
