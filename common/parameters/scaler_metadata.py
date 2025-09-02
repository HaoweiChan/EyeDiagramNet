"""
Scaler metadata utilities for EyeDiagramNet.

This module handles saving and loading scalers with associated
metadata like config_keys for proper inference compatibility.
"""

import torch
from pathlib import Path
from typing import Tuple, List

from .validation import validate_scaler_metadata


def save_scaler_with_config_keys(scalers: tuple, config_keys: list, save_path: Path) -> None:
    """
    Save scalers along with training config_keys metadata.
    
    Args:
        scalers: Tuple of (seq_scaler, fix_scaler)
        config_keys: List of parameter keys used during training
        save_path: Path to save the enhanced scaler file
        
    Raises:
        ValueError: If inputs are invalid
    """
    if not isinstance(scalers, tuple) or len(scalers) != 2:
        raise ValueError("scalers must be a tuple of length 2 (seq_scaler, fix_scaler)")
    
    if not isinstance(config_keys, list) or len(config_keys) == 0:
        raise ValueError("config_keys must be a non-empty list")
    
    scaler_data = {
        'scalers': scalers,
        'config_keys': config_keys,
        'version': '1.0'  # For future compatibility
    }
    
    # Validate the data structure before saving
    is_valid, error_messages = validate_scaler_metadata(scaler_data)
    if not is_valid:
        raise ValueError(f"Invalid scaler metadata: {error_messages}")
    
    # Ensure parent directory exists
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(scaler_data, save_path)

def load_scaler_with_config_keys(load_path: Path) -> Tuple[tuple, list]:
    """
    Load scalers and extract training config_keys metadata.
    
    Args:
        load_path: Path to the scaler file
        
    Returns:
        Tuple of (scalers, config_keys)
        
    Raises:
        ValueError: If config_keys metadata is not found or invalid
        FileNotFoundError: If scaler file doesn't exist
    """
    if not load_path.exists():
        raise FileNotFoundError(f"Scaler file not found: {load_path}")
    
    try:
        # Try loading as enhanced scaler with metadata
        scaler_data = torch.load(load_path, weights_only=False)
        
        if isinstance(scaler_data, dict) and 'scalers' in scaler_data and 'config_keys' in scaler_data:
            # Validate the loaded data
            is_valid, error_messages = validate_scaler_metadata(scaler_data)
            if not is_valid:
                raise ValueError(f"Invalid scaler metadata in {load_path}: {error_messages}")
            
            return scaler_data['scalers'], scaler_data['config_keys']
    except Exception as e:
        if "Invalid scaler metadata" in str(e):
            raise e
    
    # Fallback: Try loading as legacy scaler (tuple only)
    try:
        scalers = torch.load(load_path, weights_only=False)
        if isinstance(scalers, tuple) and len(scalers) == 2:
            raise ValueError(
                f"Legacy scaler format detected at {load_path}. "
                "This scaler does not contain config_keys metadata. "
                "Please retrain the model with the updated scaler saving logic, "
                "or manually provide config_keys for inference."
            )
    except Exception as e:
        if "config_keys metadata" in str(e):
            raise e
        else:
            raise ValueError(f"Cannot load scaler from {load_path}: {e}")
    
    raise ValueError(f"Unrecognized scaler format at {load_path}")

def extract_config_keys_from_scaler(load_path: Path) -> List[str]:
    """
    Extract just the config_keys from a scaler file without loading the full scalers.
    
    Args:
        load_path: Path to the scaler file
        
    Returns:
        List of config_keys
        
    Raises:
        ValueError: If config_keys cannot be extracted
    """
    _, config_keys = load_scaler_with_config_keys(load_path)
    return config_keys

def is_enhanced_scaler(load_path: Path) -> bool:
    """
    Check if a scaler file contains enhanced metadata.
    
    Args:
        load_path: Path to the scaler file
        
    Returns:
        True if scaler contains metadata, False otherwise
    """
    try:
        if not load_path.exists():
            return False
        
        scaler_data = torch.load(load_path, weights_only=False)
        return (isinstance(scaler_data, dict) and 
                'scalers' in scaler_data and 
                'config_keys' in scaler_data)
    except:
        return False

def upgrade_legacy_scaler(
    legacy_scaler_path: Path, 
    config_keys: List[str], 
    output_path: Path = None
) -> Path:
    """
    Upgrade a legacy scaler file to include config_keys metadata.
    
    Args:
        legacy_scaler_path: Path to legacy scaler file
        config_keys: Config keys to associate with the scaler
        output_path: Optional output path (defaults to overwriting input)
        
    Returns:
        Path to the upgraded scaler file
        
    Raises:
        ValueError: If legacy scaler is invalid or upgrade fails
    """
    if output_path is None:
        output_path = legacy_scaler_path
    
    try:
        # Load legacy scaler
        scalers = torch.load(legacy_scaler_path, weights_only=False)
        
        if not isinstance(scalers, tuple) or len(scalers) != 2:
            raise ValueError(f"Invalid legacy scaler format at {legacy_scaler_path}")
        
        # Save as enhanced scaler
        save_scaler_with_config_keys(scalers, config_keys, output_path)
        
        return output_path
        
    except Exception as e:
        raise ValueError(f"Failed to upgrade legacy scaler {legacy_scaler_path}: {e}")

def get_scaler_info(load_path: Path) -> dict:
    """
    Get information about a scaler file without fully loading it.
    
    Args:
        load_path: Path to the scaler file
        
    Returns:
        Dictionary with scaler information
    """
    info = {
        'path': str(load_path),
        'exists': load_path.exists(),
        'is_enhanced': False,
        'config_keys': None,
        'version': None,
        'error': None
    }
    
    if not info['exists']:
        info['error'] = 'File does not exist'
        return info
    
    try:
        info['is_enhanced'] = is_enhanced_scaler(load_path)
        
        if info['is_enhanced']:
            scaler_data = torch.load(load_path, weights_only=False)
            info['config_keys'] = scaler_data.get('config_keys')
            info['version'] = scaler_data.get('version')
        else:
            info['error'] = 'Legacy scaler format (no metadata)'
            
    except Exception as e:
        info['error'] = str(e)
    
    return info