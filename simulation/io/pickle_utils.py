import pickle
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass

@dataclass
class SimulationResult:
    """
    A structured dataclass to hold the results of a single eye-width simulation.
    This ensures a consistent data schema across all data collectors.
    """
    config_values: List[float]
    config_keys: List[str]
    line_ews: List[float]
    snp_drv: str
    snp_odt: str
    directions: List[int]
    snp_horiz: str
    n_ports: int
    param_types: List[str]

class DataWriter:
    """
    A standardized writer for simulation data to ensure consistent pickle format.

    This class handles loading existing data, appending new results in a
    structured format, and writing the final data back to a pickle file.
    """
    def __init__(self, pickle_file: Path):
        self.pickle_file = Path(pickle_file)
        self.data = self._load_existing_data()

    def _load_existing_data(self) -> Dict[str, Any]:
        """Load existing data from the pickle file if it exists."""
        if self.pickle_file.exists():
            try:
                with open(self.pickle_file, 'rb') as f:
                    return pickle.load(f)
            except (pickle.UnpicklingError, EOFError, FileNotFoundError):
                # Handle corrupted or empty files
                pass
        
        # Return the default, empty structure if no valid data is found
        return {
            'configs': [],
            'line_ews': [],
            'snp_drvs': [],
            'snp_odts': [],
            'directions': [],
            'meta': {}
        }

    def add_result(self, result: SimulationResult):
        """
        Add a single simulation result to the internal data structure.

        Args:
            result (SimulationResult): A dataclass instance containing the
                                     simulation output for a single run.
        """
        # Append core results from the dataclass instance
        self.data['configs'].append(result.config_values)
        self.data['line_ews'].append(result.line_ews)
        self.data['snp_drvs'].append(result.snp_drv)
        self.data['snp_odts'].append(result.snp_odt)
        self.data['directions'].append(result.directions)

        # Update metadata only if it's not already set
        if not self.data['meta']:
            self.data['meta'] = {
                'config_keys': result.config_keys,
                'snp_horiz': result.snp_horiz,
                'n_ports': result.n_ports,
                'param_types': result.param_types
            }

    def save(self):
        """
        Save the collected data to the pickle file.
        This will overwrite the existing file with the updated data.
        """
        self.pickle_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.pickle_file, 'wb') as f:
            pickle.dump(self.data, f)

    def get_sample_count(self) -> int:
        """Returns the number of samples currently in the data."""
        return len(self.data.get('configs', []))


def load_pickle_data(pfile: Path) -> List[SimulationResult]:
    """
    Loads a pickle file and converts its contents into a list of SimulationResult dataclasses.
    This function handles the logic to reconstruct the dataclass from the stored dictionary format.
    
    Args:
        pfile: Path to the pickle file to load
        
    Returns:
        List of SimulationResult dataclasses, empty list if file is malformed or missing
    """
    try:
        with open(pfile, 'rb') as f:
            data = pickle.load(f)
    except (pickle.UnpicklingError, EOFError, FileNotFoundError):
        return []

    results = []
    
    # Check for the expected dictionary structure
    if not isinstance(data, dict) or not all(k in data for k in ['configs', 'line_ews', 'meta']):
        print(f"Warning: Pickle file {pfile.name} has an unexpected format. Skipping.")
        return []

    n_samples = len(data.get('configs', []))
    
    # Get metadata, which should be consistent for all samples in the file
    meta = data.get('meta', {})
    config_keys = meta.get('config_keys', [])
    snp_horiz = meta.get('snp_horiz', '')
    n_ports = meta.get('n_ports', 0)
    param_types = meta.get('param_types', [])

    for i in range(n_samples):
        try:
            # Handle both new and legacy naming conventions for SNP files
            if 'snp_drvs' in data and 'snp_odts' in data:
                snp_drv = data['snp_drvs'][i]
                snp_odt = data['snp_odts'][i]
            elif 'snp_txs' in data and 'snp_rxs' in data:
                # Legacy format - map to new naming
                snp_drv = data['snp_txs'][i]
                snp_odt = data['snp_rxs'][i]
            else:
                raise KeyError("No SNP file paths found in data (neither new nor legacy format)")
            
            result = SimulationResult(
                config_values=data['configs'][i],
                config_keys=config_keys,
                line_ews=data['line_ews'][i],
                snp_drv=snp_drv,
                snp_odt=snp_odt,
                directions=data['directions'][i],
                snp_horiz=snp_horiz,
                n_ports=n_ports,
                param_types=param_types
            )
            results.append(result)
        except (IndexError, KeyError) as e:
            print(f"Warning: Skipping malformed sample {i} in {pfile.name} due to missing data: {e}")
            continue
            
    return results


def load_pickle_directory(label_dir: Path, dataset_name: str, config_keys: list = None) -> dict:
    """
    Load and process all pickle files from a directory for a specific dataset.
    
    Args:
        label_dir: Path to the directory containing pickle files
        dataset_name: Name of the dataset subdirectory 
        config_keys: Optional list of config keys for structured array conversion
        
    Returns:
        Dictionary mapping case_ids to processed data tuples containing:
        (configs, directions_list, line_ews_list, snp_vert, meta)
    """
    from simulation.parameters.bound_param import SampleResult, to_new_param_name
    
    labels = {}
    
    for pkl_file in Path(label_dir, dataset_name).glob("*.pkl"):
        # Load data as list of SimulationResult dataclasses
        results = load_pickle_data(pkl_file)
        
        if not results:
            print(f"Warning: Skipping malformed or empty pickle: {pkl_file.name}")
            continue

        # Extract data from the first result to get metadata
        first_result = results[0]
        snp_horiz_path = first_result.snp_horiz

        if not snp_horiz_path:
            print(f"Warning: Skipping malformed pickle: {pkl_file.name} ('snp_horiz' not found).")
            continue

        # The key must match the case_id from the CSV file
        try:
            key = int(Path(snp_horiz_path).stem.replace("-", "_").split("_")[-1].split(".")[0])
        except (ValueError, IndexError):
            print(f"Warning: Could not parse case ID from snp_horiz: '{snp_horiz_path}'. "
                   f"Skipping pickle file: {pkl_file.name}")
            continue

        # Convert dataclass results to the format expected by downstream code
        configs = []
        directions_list = []
        line_ews_list = []
        snp_drvs = []
        snp_odts = []

        for result in results:
            # Convert config from keys+values back to dict format for backward compatibility
            config_dict = dict(zip(result.config_keys, result.config_values))
            if isinstance(config_dict, dict):
                config_dict = to_new_param_name(config_dict)
            configs.append(config_dict)
            
            directions_list.append(result.directions)
            line_ews_list.append(result.line_ews)
            snp_drvs.append(result.snp_drv)
            snp_odts.append(result.snp_odt)

        # Create SNP vertical data tuple
        snp_vert = tuple(zip(snp_drvs, snp_odts))

        # Create metadata dict from the first result
        meta = {
            'config_keys': first_result.config_keys,
            'snp_horiz': first_result.snp_horiz,
            'n_ports': first_result.n_ports,
            'param_types': first_result.param_types
        }

        labels[key] = (
            configs,
            directions_list,
            line_ews_list,
            snp_vert,
            meta
        )
    
    return labels


def convert_configs_to_boundaries(configs_list: list, config_keys: list):
    """
    Convert a list of config dictionaries to structured boundary arrays.
    
    Args:
        configs_list: List of lists of config dictionaries
        config_keys: List of parameter keys for structured array conversion
        
    Returns:
        List of lists of structured numpy arrays
    """
    from simulation.parameters.bound_param import SampleResult
    
    boundaries_list = []
    for configs in configs_list:
        # Convert each list of config dicts to structured arrays
        sample_boundaries = []
        for config_dict in configs:
            # Convert dict to SampleResult and then to structured array
            sample_result = SampleResult.from_dict(config_dict)
            sample_boundaries.append(sample_result.to_structured_array(config_keys))
        boundaries_list.append(sample_boundaries)
    
    return boundaries_list
