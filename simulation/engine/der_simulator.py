import os
import sys
import json
import uuid
import tempfile
import argparse
import dataclasses
import subprocess
import skrf as rf
import pandas as pd
from contextlib import contextmanager
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from simulation.parameters.bound_param import DER_PARAMS

# Absolute path where the `from_enzo` package resides.
# Adjust as necessary if the directory moves.
DEFAULT_MODULE_ROOT = Path("/proj/siaiadm/ddr_peak_distorsion_analysis/enzo/20250623_to_willy/from_enzo")


# -------------------------------------------------
# Utility functions
# -------------------------------------------------

@contextmanager
def temp_snp_file(network: rf.Network) -> Iterator[Path]:
    """Context manager to temporarily save an skrf.Network to a file."""
    # Determine the correct suffix based on S-parameter shape
    n_ports = network.nports
    suffix = f".s{n_ports}p"
    
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as fp:
        tmp_path = Path(fp.name)
    try:
        network.write_touchstone(tmp_path)
        yield tmp_path
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

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
        "bits_per_sec": "bit_num_per_sec",
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
    der_params: Dict[str, Any] | None = None,
    algorithm: str = "der_spara_pattern_to_wave",
    python_executable: str | None = None,
    module_roots: List[str | Path] | None = None,
) -> pd.DataFrame:
    """Run a double-edge-response (DER) simulation in a separate Python
    subprocess and return the parsed CSV results as a *pandas* DataFrame.

    This function automatically manages intermediate files in a temporary directory.

    Parameters
    ----------
    snp_path
        Path to the S-parameter file (.snp/.s96p/npz) to be simulated.
    der_params
        Optional dictionary of DER parameters. If *None*, a sample from
        ``DER_PARAMS`` is used.
    algorithm
        The fully-qualified module path to the external simulation entry
        point. Defaults to ``der_spara_pattern_to_wave``.
    python_executable
        Python interpreter to use. Defaults to ``sys.executable``.
    module_roots
        Optional list of directories to add to PYTHONPATH for the subprocess.

    Returns
    -------
    pandas.DataFrame
        Parsed contents of the output CSV.
    """
    snp_path = Path(snp_path)

    with tempfile.TemporaryDirectory(prefix="der_sim_") as tmpdir:
        output_dir = Path(tmpdir)
        sim_id = uuid.uuid4().hex[:8]
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
        # external module can be resolved even if it is not installed
        # in the current environment.
        env = os.environ.copy()
        # Default to the hard-coded module root if caller didn't provide one
        module_roots = module_roots or [DEFAULT_MODULE_ROOT]
        
        # Validate that the DEFAULT_MODULE_ROOT exists
        for root in module_roots:
            if Path(root) == DEFAULT_MODULE_ROOT and not DEFAULT_MODULE_ROOT.exists():
                raise FileNotFoundError(
                    f"DEFAULT_MODULE_ROOT ({DEFAULT_MODULE_ROOT}) does not exist. "
                    f"Please ensure the external module path is correct and accessible."
                )

        extra_paths = [str(Path(p).expanduser()) for p in module_roots]
        # Explicitly set PYTHONPATH to only include our desired paths
        # This prevents issues with inherited inconsistent PYTHONPATH values
        env["PYTHONPATH"] = os.pathsep.join(extra_paths)

        # Run subprocess and capture output for debugging purposes
        try:
            subprocess.run(cmd, check=True, env=env, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            print("DEBUG: Subprocess stdout (on error):", exc.stdout, file=sys.stderr)
            print("DEBUG: Subprocess stderr (on error):", exc.stderr, file=sys.stderr)
            raise RuntimeError(
                f"DER simulation failed with exit code {exc.returncode}. "
                f"Command: {' '.join(cmd)}"
            ) from exc

        # Parse resulting CSV
        df = _parse_csv(csv_path)
        return df


@dataclass
class DERTransientParams:
    """Configuration parameters for DER transient analysis."""
    R_drv: float
    R_odt: float
    C_drv: float
    C_odt: float
    bits_per_sec: float
    snp_horiz: Any
    snp_drv: Optional[Any] = None
    snp_odt: Optional[Any] = None

    # Parameters from DER_PARAMS in bound_param.py
    vmask: Optional[float] = None
    vh: Optional[float] = None
    vl: Optional[float] = None
    tvl: Optional[float] = None
    tvh: Optional[float] = None
    tr_rising: Optional[float] = None
    vp: Optional[float] = None
    tvp: Optional[float] = None
    tf_rising: Optional[float] = None
    tf_falling: Optional[float] = None

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'DERTransientParams':
        """Create a DERTransientParams instance from a configuration dictionary."""
        valid_fields = {f.name for f in fields(cls)}
        filtered_config = {key: value for key, value in config.items() if key in valid_fields}
        return cls(**filtered_config)

class DERCollectorSimulator:
    """
    Simulator class for DER analysis, compatible with the collection framework.
    """

    def __init__(self, config, snp_files=None):
        """Initialize the DERCollectorSimulator."""
        if isinstance(config, dict):
            config_dict = config.copy()
        else:
            if hasattr(config, 'to_dict'):
                config_dict = config.to_dict()
            else:
                config_dict = {attr: getattr(config, attr) for attr in dir(config) if not attr.startswith('_')}

        if snp_files:
            if isinstance(snp_files, (tuple, list)) and len(snp_files) >= 1:
                config_dict['snp_horiz'] = snp_files[0]
            if len(snp_files) >= 3:
                config_dict['snp_drv'] = snp_files[1]
                config_dict['snp_odt'] = snp_files[2]

        self.params = DERTransientParams.from_config(config_dict)
        self.ntwk = self._load_and_cascade_networks()

    def _load_and_cascade_networks(self) -> rf.Network:
        """Load and cascade S-parameter networks."""
        ntwk_horiz = self.params.snp_horiz
        if isinstance(ntwk_horiz, (str, Path)):
            ntwk_horiz = rf.Network(ntwk_horiz)

        if self.params.snp_drv and self.params.snp_odt:
            ntwk_drv = self.params.snp_drv
            ntwk_odt = self.params.snp_odt
            if isinstance(ntwk_drv, (str, Path)):
                ntwk_drv = rf.Network(ntwk_drv)
            if isinstance(ntwk_odt, (str, Path)):
                ntwk_odt = rf.Network(ntwk_odt)
                ntwk_odt.flip()

            # Cascade the networks
            ntwk = ntwk_drv ** ntwk_horiz ** ntwk_odt
        else:
            ntwk = ntwk_horiz

        return ntwk

    def run_simulation(self) -> pd.DataFrame:
        """Run the DER simulation and return the results."""
        der_params = dataclasses.asdict(self.params)

        # Clean up None values and network objects before passing
        params_to_pass = {
            k: v for k, v in der_params.items() 
            if v is not None and k not in ['snp_horiz', 'snp_drv', 'snp_odt']
        }
        
        # Use the original SNP path instead of creating a temporary file
        if isinstance(self.params.snp_horiz, (str, Path)):
            snp_path = self.params.snp_horiz
        else:
            # If snp_horiz is already a Network object, we need to save it temporarily
            # This is a fallback for when the original path is not available
            with temp_snp_file(self.ntwk) as tmp_path:
                snp_path = tmp_path
        
        df = run_der_simulation(
            snp_path=snp_path,
            der_params=params_to_pass
        )
        return df


def snp_der_simulation(config, snp_files=None) -> List[float]:
    """
    Main function for DER simulation compatible with data collectors.
    """
    try:
        simulator = DERCollectorSimulator(config, snp_files)
        df = simulator.run_simulation()

        # Extract a single metric from the result DataFrame.
        # Assuming the CSV has a column 'der_metric' and we need one value per line
        # For now, just return the first value of the first column.
        # This part might need adjustment based on the actual CSV format.
        if not df.empty and 'der_metric' in df.columns:
             return df['der_metric'].tolist()

        # Fallback: if no 'der_metric' column, return first value of first column
        if not df.empty:
            return [df.iloc[0, 0]]
            
        return [-1.0] # Return a default value for failure

    except Exception as e:
        print(f"DER simulation failed: {str(e)}")
        raise RuntimeError(f"DER simulation failed: {str(e)}") from e

def main() -> None:
    """CLI wrapper for quick testing.

    Example
    -------
    python -m simulation.engine.der_simulator --snp_path my.s96p
    """
    parser = argparse.ArgumentParser(description="Run DER simulation via subprocess")
    parser.add_argument("--snp_path", required=True, help="Path to the SNP file")
    args = parser.parse_args()

    # Randomly sample parameters from DER_PARAMS
    der_params = DER_PARAMS.sample().to_dict()
    print(f"Using sampled parameters: {der_params}")
    
    df = run_der_simulation(args.snp_path, der_params=der_params)
    print(df.head())

if __name__ == "__main__":
    main()
