#!/usr/bin/env python
r"""Generate synthetic Touchstone files for a bundle of transmission lines.

For a 2\*N port network we model ``N`` parallel traces where port ``i`` and
port ``i+N`` represent the two ends of the same physical trace.  The traces are
assumed to be laid out sequentially so that line ``0`` is adjacent to line ``1``
and so on.  Crosstalk therefore decreases as the spacing (``|i-j|``) between
two lines increases.

Insertion loss and phase delay are determined by the length of each trace.
Trace lengths are randomly chosen based on the supplied seed so different
seeds produce different S\ :sub:`21` magnitudes and phases.  The generated
crosstalk levels follow a simple physical model where near-end coupling is
stronger than far-end coupling and both fall off with frequency and with the
distance between traces.
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
    Generate and write a single 2N-port ``.sNp`` file with randomised trace
    lengths and physically inspired crosstalk.  The returned value is the path
    of the written file.
    """
    rng = np.random.default_rng(seed)
    n_ports = 2 * n_lines
    freqs = np.logspace(np.log10(f_start_hz), np.log10(f_stop_hz), n_points)
    S = np.zeros((n_points, n_ports, n_ports), dtype=np.complex128)

    # --- physical parameters -----------------------------------------
    length_min = 0.05  # m
    length_max = 0.15  # m
    alpha_db_per_m_1ghz = 10.0
    velocity = 0.7 * 3e8  # m/s

    xtalk_near_ref_db = -40.0
    xtalk_far_ref_db = -55.0
    xtalk_cross_ref_db = -45.0
    xtalk_dist_decay_db = 6.0
    # -----------------------------------------------------------------

    lengths = rng.uniform(length_min, length_max, size=n_lines)

    for k, f in enumerate(freqs):
        # per-line insertion loss and delay
        for i, length in enumerate(lengths):
            tx, rx = i, i + n_lines
            alpha_db = alpha_db_per_m_1ghz * np.sqrt(f / 1e9) * length
            mag = 10 ** (-alpha_db / 20)
            phase = -2 * np.pi * f * length / velocity
            s_line = mag * np.exp(1j * phase)
            S[k, rx, tx] = S[k, tx, rx] = s_line
            S[k, tx, tx] = S[k, rx, rx] = 10 ** (-30 / 20)

        # crosstalk terms
        for i in range(n_lines):
            for j in range(i + 1, n_lines):
                dist = j - i
                near_db = (
                    xtalk_near_ref_db
                    - xtalk_dist_decay_db * (dist - 1)
                    - 10 * np.log10(f / f_start_hz)
                )
                far_db = (
                    xtalk_far_ref_db
                    - xtalk_dist_decay_db * (dist - 1)
                    - 10 * np.log10(f / f_start_hz)
                )
                cross_db = (
                    xtalk_cross_ref_db
                    - xtalk_dist_decay_db * (dist - 1)
                    - 10 * np.log10(f / f_start_hz)
                )

                near = 10 ** (near_db / 20) * np.exp(1j * rng.uniform(0, 2 * np.pi))
                far = 10 ** (far_db / 20) * np.exp(1j * rng.uniform(0, 2 * np.pi))
                cross = 10 ** (cross_db / 20) * np.exp(1j * rng.uniform(0, 2 * np.pi))

                # near-end coupling (same ends)
                S[k, i, j] += near
                S[k, j, i] += near
                S[k, i + n_lines, j + n_lines] += near
                S[k, j + n_lines, i + n_lines] += near

                # far-end coupling (opposite ends)
                S[k, i, j + n_lines] += far
                S[k, j, i + n_lines] += far
                S[k, j + n_lines, i] += far
                S[k, i + n_lines, j] += far

                # cross coupling extra terms
                S[k, i, j] += cross
                S[k, i + n_lines, j + n_lines] += cross
                S[k, i, j + n_lines] += cross
                S[k, j, i + n_lines] += cross

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
        description=(
            "Generate one or many 2N-port Touchstone files with randomised "
            "trace lengths and physically meaningful crosstalk"))
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
                               count=args.count,                               file_name=args.output)