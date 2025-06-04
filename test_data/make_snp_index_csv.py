import os
import csv
from pathlib import Path
import re

# Directory containing this script and the .s96p files
base_dir = Path(__file__).parent

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
with open(base_dir / 'snp_index.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['index', 'snp_file_name'])
    for idx, fname in snp_files:
        writer.writerow([idx, fname])

print(f"Wrote {len(snp_files)} entries to snp_index.csv") 