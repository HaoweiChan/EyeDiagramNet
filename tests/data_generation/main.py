import argparse
from pathlib import Path

from . import generator
from . import indexer

def main():
    parser = argparse.ArgumentParser(description="A tool to generate synthetic Touchstone files and create index files.")
    
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    # --- Generate Command ---
    parser_generate = subparsers.add_parser("generate", help="Generate one or many 2N-port Touchstone files.")
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

    args = parser.parse_args()
    
    if args.command == "generate":
        if args.count == 1:
            generator.build_snp(n_lines=args.n_lines,
                      seed=args.seed,
                      file_name=args.output)
        else:
            generator.generate_multiple_snps(n_lines=args.n_lines,
                                   base_seed=args.seed,
                                   count=args.count,
                                   file_name=args.output)
    elif args.command == "index":
        data_dir = Path(args.data_dir).expanduser()
        if not data_dir.is_dir():
            print(f"Error: Provided path '{data_dir}' is not a valid directory.")
            return
        indexer.create_snp_index_csv(data_dir)

if __name__ == '__main__':
    main()
