#!/usr/bin/env python
"""
Generate one or many 2N-port Touchstone files for N transmission lines.
Port mapping:
    0 … n-1   : Tx ends
    n … 2n-1  : Rx ends
Each line i <-> (i+n) is one physical line.
"""

import numpy as np
import skrf as rf
from pathlib import Path


def build_snp(n_lines: int,
              seed: int,
              f_start_hz: float = 1e6,
              f_stop_hz: float = 10e9,
              n_points: int = 201,
              file_name: str | None = None) -> Path:
    """
    Generate and write a single 2N-port .sNp file with random IL and crosstalk.
    Returns the path of the written file.
    """
    rng = np.random.default_rng(seed)
    n_ports = 2 * n_lines
    freqs = np.logspace(np.log10(f_start_hz), np.log10(f_stop_hz), n_points)
    S = np.zeros((n_points, n_ports, n_ports), dtype=np.complex128)

    # --- tunable parameters ------------------------------------------
    IL_stop_db_nominal = -6.0    # nominal IL at f_stop
    IL_spread_db = 1.5           # ±random spread (dB)
    xtalk_near_db = -40.0
    xtalk_far_db  = -50.0
    xtalk_cross_db= -45.0
    # -----------------------------------------------------------------

    slope = IL_stop_db_nominal / (np.log10(f_stop_hz) - np.log10(f_start_hz))
    il_db_lines = (IL_stop_db_nominal +
                   rng.uniform(-IL_spread_db, IL_spread_db, size=n_lines))

    for k, f in enumerate(freqs):
        log_ratio = np.log10(f / f_start_hz)
        # per-line insertion loss & reflections
        for i in range(n_lines):
            tx, rx = i, i + n_lines
            il_db  = slope * log_ratio + (il_db_lines[i] - IL_stop_db_nominal)
            il_mag = 10 ** (il_db / 20)
            S[k, rx, tx] = S[k, tx, rx] = il_mag
            S[k, tx, tx] = S[k, rx, rx] = 10 ** (-30/20)  # return loss 30 dB

        # crosstalk terms (frequency-dependent)
        xt_near  = 10 ** ((xtalk_near_db  - 10*np.log10(f/f_start_hz)) / 20)
        xt_far   = 10 ** ((xtalk_far_db   - 10*np.log10(f/f_start_hz)) / 20)
        xt_cross = 10 ** ((xtalk_cross_db - 10*np.log10(f/f_start_hz)) / 20)

        for i in range(n_lines):
            for j in range(i+1, n_lines):
                # near-end (same-end) Tx-Tx, Rx-Rx
                S[k, i, j]       = S[k, j, i]       = xt_near
                S[k, i+n_lines, j+n_lines] = S[k, j+n_lines, i+n_lines] = xt_near
                # far-end (opposite-end) Tx-Tx
                S[k, i, j+n_lines] = S[k, j+n_lines, i] = xt_far
                S[k, j, i+n_lines] = S[k, i+n_lines, j] = xt_far
                # cross-end (Tx-Rx) extra coupling
                S[k, i, j]         += xt_cross
                S[k, i+n_lines, j+n_lines] += xt_cross
                S[k, i, j+n_lines] += xt_cross
                S[k, j, i+n_lines] += xt_cross

    ntwk = rf.Network(frequency=rf.Frequency.from_f(freqs, unit='hz'),
                      s=S, z0=50.0)

    if file_name is None:
        file_name = f"tlines{n_lines}_seed{seed}.s{n_ports}p"
    out_path = Path(file_name).with_suffix(f".s{n_ports}p").resolve()
    ntwk.write_touchstone(str(out_path.with_suffix('')))
    print(f"Written: {out_path}")
    return out_path


def generate_multiple_snps(n_lines: int,
                           base_seed: int,
                           count: int,
                           **build_kwargs) -> list[Path]:
    """
    Generate `count` .sNp files, using seeds:
        base_seed, base_seed+1, ..., base_seed+(count-1)
    Returns list of file paths.
    """
    paths = []
    for offset in range(count):
        seed = base_seed + offset
        name = build_kwargs.get("file_name")
        # if a single output name was given, append index:
        if name:
            stem, suffix = Path(name).stem, Path(name).suffix
            name_i = f"{stem}_{offset+1}{suffix}"
            kwargs = {**build_kwargs, "file_name": name_i}
        else:
            kwargs = build_kwargs
        paths.append(build_snp(n_lines=n_lines, seed=seed, **kwargs))
    return paths


# ------------------- CLI --------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate 1 or many 2N-port Touchstone files with random IL/crosstalk")
    parser.add_argument("-n", "--n_lines", type=int, default=4,
                        help="number of transmission lines (default: 4)")
    parser.add_argument("-s", "--seed", type=int, default=0,
                        help="base random seed (default: 0)")
    parser.add_argument("-k", "--count", type=int, default=1,
                        help="how many files to generate with consecutive seeds")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="output file name or stem (optional)")
    args = parser.parse_args()

    if args.count == 1:
        build_snp(n_lines=args.n_lines,
                  seed=args.seed,
                  file_name=args.output)
    else:
        generate_multiple_snps(n_lines=args.n_lines,
                               base_seed=args.seed,
                               count=args.count,
                               file_name=args.output)