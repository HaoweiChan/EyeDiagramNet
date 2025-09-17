import argparse
from pathlib import Path

import snp_generator
import indexer
import contour_generator

def main():
    parser = argparse.ArgumentParser(description="A tool to generate synthetic data for EyeDiagramNet testing.")
    
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    # --- Generate SNP Command ---
    parser_generate = subparsers.add_parser("generate", help="Generate one or many 2N-port Touchstone (SNP) files.")
    parser_generate.add_argument("-n", "--n_lines", type=int, default=4,
                        help="number of transmission lines (default: 4)")
    parser_generate.add_argument("-s", "--seed", type=int, default=0,
                        help="base random seed (default: 0)")
    parser_generate.add_argument("-k", "--count", type=int, default=1,
                        help="how many files to generate with consecutive seeds")
    parser_generate.add_argument("-o", "--output", type=str, default=None,
                        help="output file name or stem (optional)")

    # --- Index Command ---
    parser_index = subparsers.add_parser("index", help="Create an index CSV file from generated SNP files.")
    parser_index.add_argument("data_dir", type=str, help="Path to the directory containing .s96p files.")

    # --- Contour Command ---
    parser_contour = subparsers.add_parser("contour", help="Generate synthetic contour data (sequence.csv + variation.csv).")
    parser_contour.add_argument("-n", "--name", type=str, default="test_contour",
                               help="Dataset name prefix (default: test_contour)")
    parser_contour.add_argument("--min-segments", type=int, default=10,
                               help="Minimum number of segments (default: 10)")
    parser_contour.add_argument("--max-segments", type=int, default=50,
                               help="Maximum number of segments (default: 50)")
    parser_contour.add_argument("--min-cases", type=int, default=3,
                               help="Minimum number of variation cases (default: 3)")
    parser_contour.add_argument("--max-cases", type=int, default=15,
                               help="Maximum number of variation cases (default: 15)")
    parser_contour.add_argument("-s", "--seed", type=int, default=42,
                               help="Random seed (default: 42)")
    parser_contour.add_argument("-o", "--output", type=str, default=None,
                               help="Output directory (default: contour/)")

    args = parser.parse_args()
    
    if args.command == "generate":
        if args.count == 1:
            snp_generator.build_snp(n_lines=args.n_lines,
                      seed=args.seed,
                      file_name=args.output)
        else:
            snp_generator.generate_multiple_snps(n_lines=args.n_lines,
                                   base_seed=args.seed,
                                   count=args.count,
                                   file_name=args.output)
    elif args.command == "index":
        data_dir = Path(args.data_dir).expanduser()
        if not data_dir.is_dir():
            print(f"Error: Provided path '{data_dir}' is not a valid directory.")
            return
        indexer.create_snp_index_csv(data_dir)
    elif args.command == "contour":
        output_dir = Path(args.output) if args.output else Path("contour") / args.name
        contour_generator.generate_contour_dataset(
            output_dir=output_dir,
            dataset_name=args.name,
            min_segments=args.min_segments,
            max_segments=args.max_segments,
            min_cases=args.min_cases,
            max_cases=args.max_cases,
            seed=args.seed
        )

if __name__ == '__main__':
    main()
