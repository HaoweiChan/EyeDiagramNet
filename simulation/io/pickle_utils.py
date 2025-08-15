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
