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
    from lightning.pytorch.utilities.rank_zero import rank_zero_info
    
    try:
        with open(pfile, 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        rank_zero_info(f"File not found: {pfile}")
        return []
    except (pickle.UnpicklingError, EOFError) as e:
        rank_zero_info(f"Error loading pickle file {pfile}: {e}")
        return []

    results = []
    
    # Check for the expected dictionary structure
    if not isinstance(data, dict) or not all(k in data for k in ['configs', 'line_ews', 'meta']):
        rank_zero_info(f"Warning: Pickle file {pfile.name} has an unexpected format. Skipping.")
        return []

    n_samples = len(data.get('configs', []))
    
    # Get metadata, which should be consistent for all samples in the file
    meta = data.get('meta', {})
    config_keys = meta.get('config_keys', [])
    snp_horiz = meta.get('snp_horiz', '')
    n_ports = meta.get('n_ports', 0)
    param_types = meta.get('param_types', [])
    
    # Convert legacy parameter names to new format using centralized mapping
    from common.parameters import convert_legacy_param_names
    config_keys = convert_legacy_param_names(config_keys, target_format='new')

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
            rank_zero_info(f"Warning: Skipping malformed sample {i} in {pfile.name} due to missing data: {e}")
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
    from common.parameters import SampleResult, to_new_param_name
    from lightning.pytorch.utilities.rank_zero import rank_zero_info
    
    labels = {}
    processed_files = 0
    skipped_files = 0
    
    dataset_path = Path(label_dir, dataset_name)
    if not dataset_path.exists():
        rank_zero_info(f"Dataset path does not exist: {dataset_path}")
        return {}
        
    pkl_files = list(dataset_path.glob("*.pkl"))
    rank_zero_info(f"Found {len(pkl_files)} pickle files in {dataset_path}")
    
    for pkl_file in pkl_files:
        try:
            # Load data as list of SimulationResult dataclasses
            results = load_pickle_data(pkl_file)
            
            if not results:
                rank_zero_info(f"Warning: Skipping malformed or empty pickle: {pkl_file.name}")
                skipped_files += 1
                continue

            # Extract data from the first result to get metadata
            first_result = results[0]
            snp_horiz_path = first_result.snp_horiz

            if not snp_horiz_path:
                rank_zero_info(f"Warning: Skipping malformed pickle: {pkl_file.name} ('snp_horiz' not found).")
                skipped_files += 1
                continue

            # Try multiple methods to extract case ID
            key = None
            
            # Method 1: Try to extract from snp_horiz path 
            try:
                key = int(Path(snp_horiz_path).stem.replace("-", "_").split("_")[-1].split(".")[0])
            except (ValueError, IndexError):
                pass
            
            # Method 2: If method 1 fails, try to extract from pickle filename
            if key is None:
                try:
                    # Extract number from pickle filename
                    filename_parts = pkl_file.stem.replace("-", "_").split("_")
                    for part in reversed(filename_parts):
                        try:
                            key = int(part)
                            break
                        except ValueError:
                            continue
                except (ValueError, IndexError):
                    pass
            
            # Method 3: If both methods fail, use a hash of the filename as fallback
            if key is None:
                import hashlib
                key = int(hashlib.md5(pkl_file.name.encode()).hexdigest()[:8], 16)
                rank_zero_info(f"Warning: Could not parse case ID from '{snp_horiz_path}' or '{pkl_file.name}'. "
                      f"Using hash-based key: {key}")
            
            # Check for duplicate keys
            if key in labels:
                rank_zero_info(f"Warning: Duplicate key {key} found for file {pkl_file.name}. "
                      f"Previous file will be overwritten.")
                
        except Exception as e:
            rank_zero_info(f"Error processing file {pkl_file.name}: {str(e)}")
            skipped_files += 1
            continue

        # Convert dataclass results to the format expected by downstream code
        try:
            configs = []
            directions_list = []
            line_ews_list = []
            snp_drvs = []
            snp_odts = []

            for result in results:
                try:
                    # Convert config from keys+values back to dict format for backward compatibility
                    config_dict = dict(zip(result.config_keys, result.config_values))
                    if isinstance(config_dict, dict):
                        config_dict = to_new_param_name(config_dict)
                    configs.append(config_dict)
                    
                    directions_list.append(result.directions)
                    line_ews_list.append(result.line_ews)
                    snp_drvs.append(result.snp_drv)
                    snp_odts.append(result.snp_odt)
                except Exception as e:
                    rank_zero_info(f"Warning: Error processing result in {pkl_file.name}: {str(e)}")
                    continue

            # Skip file if no valid results were processed
            if not configs:
                rank_zero_info(f"Warning: No valid results found in {pkl_file.name}")
                skipped_files += 1
                continue

            # Create SNP vertical data tuple
            snp_vert = tuple(zip(snp_drvs, snp_odts))

            # Create metadata dict from the first result  
            meta = {
                'config_keys': config_keys,
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
            processed_files += 1
            
        except Exception as e:
            rank_zero_info(f"Error processing data from {pkl_file.name}: {str(e)}")
            skipped_files += 1
            continue
    
    # Print summary
    rank_zero_info(f"Successfully processed {processed_files} pickle files, skipped {skipped_files} files")
    
    return labels
