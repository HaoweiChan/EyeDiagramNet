import os
import sys
import json
import uuid
import tempfile
import subprocess
import pandas as pd
import shutil
from pathlib import Path
from typing import Any, Dict, List

from simulation.parameters.bound_param import DER_PARAMS

# Absolute path where the `from_enzo` package resides.
# Adjust as necessary if the directory moves.
DEFAULT_MODULE_ROOT = Path("/proj/siaiadm/ddr_peak_distortion_analysis/enzo/20250623_to_willy")

# -------------------------------------------------
# Utility functions
# -------------------------------------------------

def _map_der_params_to_config(der_params: Dict[str, Any]) -> Dict[str, Any]:
    """Map parameter keys from DER_PARAMS to the JSON keys expected by the
    external *from_enzo* simulator.

    The external script follows a lower-case naming convention, while
    DER_PARAMS uses capitalised keys. This helper performs the necessary
    translation and returns a new dictionary ready to be merged into the
    final configuration.
    """
    key_map = {
        "R_drv": "r_drv",
        "R_odt": "r_odt",
        "C_drv": "c_drv",
        "C_odt": "c_odt",
        "L_drv": "l_drv",
        "L_odt": "l_odt",
        "bits_per_sec": "bit_num_per_sec",
        "pulse_amplitude": "vp",
        # The remaining keys in DER_PARAMS already match their JSON names.
    }

    cfg: Dict[str, Any] = {}
    for k, v in der_params.items():
        cfg[key_map.get(k, k.lower())] = v
    return cfg


def _write_json(config: Dict[str, Any], path: Path) -> None:
    """Write *config* to *path* in pretty-printed JSON format."""
    path.write_text(json.dumps(config, indent=2))


def _parse_csv(csv_path: Path) -> pd.DataFrame:
    """Load the CSV produced by the DER simulator and return a DataFrame."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Expected output CSV not found: {csv_path}")
    return pd.read_csv(csv_path)


# -------------------------------------------------
# Public API
# -------------------------------------------------

def run_der_simulation(
    snp_path: str | Path,
    output_dir: str | Path | None = None,
    sim_id: str | None = None,
    der_params: Dict[str, Any] | None = None,
    algorithm: str = "from_enzo.der_spara_pattern_to_wave",
    python_executable: str | None = None,
    module_roots: List[str | Path] | None = None,
) -> pd.DataFrame:
    """Run a double-edge-response (DER) simulation in a separate Python
    subprocess and return the parsed CSV results as a *pandas* DataFrame.

    Parameters
    ----------
    snp_path
        Path to the S-parameter file (.snp/.s96p/npz) to be simulated.
    output_dir
        Directory to store intermediate files and final CSV. A temporary
        directory is created if *None*.
    sim_id
        Optional simulation identifier written into the CSV.
    der_params
        Optional dictionary of DER parameters. If *None*, a sample from
        ``DER_PARAMS`` is used.
    algorithm
        The fully-qualified module path to the external simulation entry
        point. Defaults to ``from_enzo.der_spara_pattern_to_wave``.
    python_executable
        Python interpreter to use. Defaults to ``sys.executable``.

    Returns
    -------
    pandas.DataFrame
        Parsed contents of the output CSV.
    """
    snp_path = Path(snp_path)
    output_dir_provided = output_dir is not None
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="der_sim_"))
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    sim_id = sim_id or uuid.uuid4().hex[:8]
    der_params = der_params or DER_PARAMS.sample().to_dict()

    cfg: Dict[str, Any] = {
        "snp_path": snp_path.as_posix(),
        "if_add_c": True,  # keep default behaviour of adding capacitance
    }
    cfg.update(_map_der_params_to_config(der_params))

    csv_path = output_dir / f"{sim_id}.csv"
    cfg.update({
        "output_path": output_dir.as_posix(),
        "csv_path": csv_path.as_posix(),
        "sim_id": sim_id,
    })

    # Write config JSON
    config_path = output_dir / f"config_{sim_id}.json"
    _write_json(cfg, config_path)

    # Build command
    python_exec = python_executable or sys.executable
    cmd = [python_exec, "-m", algorithm, str(config_path)]

    # Prepare environment with optional extra PYTHONPATH entries so that the
    # external *from_enzo* module can be resolved even if it is not installed
    # in the current environment.
    env = os.environ.copy()
    # Default to the hard-coded module root if caller didn't provide one
    module_roots = module_roots or [DEFAULT_MODULE_ROOT]
    extra_paths = [str(Path(p).expanduser()) for p in module_roots]
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(extra_paths + ([existing] if existing else [])) if existing else os.pathsep.join(extra_paths)

    # Run subprocess and capture output for debugging purposes
    try:
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"DER simulation failed with exit code {exc.returncode}. "
            f"Command: {' '.join(cmd)}"
        ) from exc

    # Parse resulting CSV
    df = _parse_csv(csv_path)

    # Clean up temp directory unless the caller explicitly asked to keep it
    if not output_dir_provided:
        shutil.rmtree(output_dir, ignore_errors=True)

    return df


def main() -> None:
    """CLI wrapper for quick testing.

    Example
    -------
    python -m simulation.engine.def_simulator --snp_path my.s96p --output_dir ./out
    """
    import argparse

    parser = argparse.ArgumentParser(description="Run DER simulation via subprocess")
    parser.add_argument("--snp_path", required=True, help="Path to the SNP file")
    parser.add_argument("--output_dir", help="Directory for outputs; defaults to tmp")
    parser.add_argument("--sim_id", help="Optional simulation id")
    args = parser.parse_args()

    df = run_der_simulation(args.snp_path)
    print(df.head())


if __name__ == "__main__":
    main()
