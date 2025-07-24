import os
import torch
import torch.nn as nn
from typing import Dict, Optional, Union

from ..models.snp_model import SNPEmbedding, OptimizedSNPEmbedding

def extract_snp_encoder_weights(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    """
    Extract SNP encoder weights from a pretrained checkpoint
    
    Args:
        checkpoint_path: Path to the checkpoint file
    
    Returns:
        Dictionary of encoder state dict
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Check if it's a Lightning checkpoint with encoder_state_dict saved
    if 'encoder_state_dict' in checkpoint:
        return checkpoint['encoder_state_dict']
    
    # Otherwise, extract from the full state dict
    state_dict = checkpoint.get('state_dict', checkpoint)
    encoder_weights = {}
    
    # Extract encoder weights (they should have 'encoder.' prefix in Lightning module)
    for key, value in state_dict.items():
        if key.startswith('encoder.'):
            # Remove 'encoder.' prefix
            new_key = key[8:]
            encoder_weights[new_key] = value
    
    if not encoder_weights:
        raise ValueError("No encoder weights found in checkpoint")
    
    return encoder_weights

def load_pretrained_snp_encoder(
    model: nn.Module,
    checkpoint_path: str,
    encoder_attr: str = 'snp_encoder',
    strict: bool = True,
    freeze: bool = False
) -> nn.Module:
    """
    Load pretrained SNP encoder weights into a model
    
    Args:
        model: Model containing SNP encoder (e.g., EyeWidthRegressor)
        checkpoint_path: Path to pretrained checkpoint
        encoder_attr: Attribute name of the encoder in the model
        strict: Whether to strictly enforce that the keys match
        freeze: Whether to freeze the encoder weights after loading
    
    Returns:
        Model with loaded pretrained weights
    """
    # Get encoder from model
    if not hasattr(model, encoder_attr):
        raise AttributeError(f"Model does not have attribute '{encoder_attr}'")
    
    encoder = getattr(model, encoder_attr)
    
    # Load pretrained weights
    pretrained_weights = extract_snp_encoder_weights(checkpoint_path)
    
    # Load weights into encoder
    missing_keys, unexpected_keys = encoder.load_state_dict(
        pretrained_weights, strict=strict
    )
    
    if missing_keys:
        print(f"Warning: Missing keys when loading pretrained encoder: {missing_keys}")
    if unexpected_keys:
        print(f"Warning: Unexpected keys when loading pretrained encoder: {unexpected_keys}")
    
    # Freeze encoder if requested
    if freeze:
        freeze_snp_encoder(model, encoder_attr)
    
    print(f"Successfully loaded pretrained SNP encoder from {checkpoint_path}")
    if freeze:
        print(f"SNP encoder weights frozen")
    
    return model

def freeze_snp_encoder(model: nn.Module, encoder_attr: str = 'snp_encoder'):
    """
    Freeze SNP encoder weights to prevent training
    
    Args:
        model: Model containing SNP encoder
        encoder_attr: Attribute name of the encoder
    """
    encoder = getattr(model, encoder_attr)
    
    for param in encoder.parameters():
        param.requires_grad = False
    
    # Set encoder to eval mode
    encoder.eval()
    
    # Count frozen parameters
    num_frozen = sum(p.numel() for p in encoder.parameters())
    print(f"Frozen {num_frozen:,} parameters in SNP encoder")

def unfreeze_snp_encoder(model: nn.Module, encoder_attr: str = 'snp_encoder'):
    """
    Unfreeze SNP encoder weights to allow training
    
    Args:
        model: Model containing SNP encoder
        encoder_attr: Attribute name of the encoder
    """
    encoder = getattr(model, encoder_attr)
    
    for param in encoder.parameters():
        param.requires_grad = True
    
    # Set encoder back to train mode
    encoder.train()
    
    # Count unfrozen parameters
    num_unfrozen = sum(p.numel() for p in encoder.parameters())
    print(f"Unfrozen {num_unfrozen:,} parameters in SNP encoder")

def create_pretrained_snp_encoder(
    checkpoint_path: str,
    encoder_type: Optional[str] = None,
    model_dim: Optional[int] = None,
    freq_length: Optional[int] = None,
    device: Union[str, torch.device] = 'cpu'
) -> Union[SNPEmbedding, OptimizedSNPEmbedding]:
    """
    Create a standalone SNP encoder from a pretrained checkpoint
    
    Args:
        checkpoint_path: Path to pretrained checkpoint
        encoder_type: Type of encoder ('SNPEmbedding' or 'OptimizedSNPEmbedding')
                     If None, will try to infer from checkpoint
        model_dim: Model dimension. If None, will try to infer from checkpoint
        freq_length: Frequency length. If None, will try to infer from checkpoint
        device: Device to load the encoder on
    
    Returns:
        Pretrained SNP encoder
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Try to get encoder config from checkpoint
    if 'encoder_config' in checkpoint:
        config = checkpoint['encoder_config']
        encoder_type = encoder_type or config.get('encoder_type', 'OptimizedSNPEmbedding')
        model_dim = model_dim or config.get('model_dim', 768)
        freq_length = freq_length or config.get('freq_length', 1601)
    else:
        # Use defaults if not provided
        encoder_type = encoder_type or 'OptimizedSNPEmbedding'
        model_dim = model_dim or 768
        freq_length = freq_length or 1601
    
    # Create encoder
    if encoder_type == 'SNPEmbedding':
        encoder = SNPEmbedding(model_dim=model_dim, freq_length=freq_length)
    elif encoder_type == 'OptimizedSNPEmbedding':
        encoder = OptimizedSNPEmbedding(model_dim=model_dim, freq_length=freq_length)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
    
    # Load weights
    encoder_weights = extract_snp_encoder_weights(checkpoint_path)
    encoder.load_state_dict(encoder_weights)
    
    # Move to device and set to eval mode
    encoder = encoder.to(device)
    encoder.eval()
    
    return encoder

def compare_encoder_weights(
    checkpoint1: str,
    checkpoint2: str,
    tolerance: float = 1e-6
) -> Dict[str, float]:
    """
    Compare encoder weights between two checkpoints
    
    Args:
        checkpoint1: Path to first checkpoint
        checkpoint2: Path to second checkpoint
        tolerance: Tolerance for considering weights equal
    
    Returns:
        Dictionary with comparison metrics
    """
    weights1 = extract_snp_encoder_weights(checkpoint1)
    weights2 = extract_snp_encoder_weights(checkpoint2)
    
    if set(weights1.keys()) != set(weights2.keys()):
        raise ValueError("Checkpoints have different encoder architectures")
    
    metrics = {}
    total_diff = 0.0
    max_diff = 0.0
    
    for key in weights1:
        diff = torch.abs(weights1[key] - weights2[key])
        total_diff += diff.sum().item()
        max_diff = max(max_diff, diff.max().item())
    
    total_params = sum(w.numel() for w in weights1.values())
    
    metrics['mean_absolute_diff'] = total_diff / total_params
    metrics['max_absolute_diff'] = max_diff
    metrics['num_equal_params'] = sum(
        (torch.abs(weights1[k] - weights2[k]) < tolerance).sum().item()
        for k in weights1
    )
    metrics['total_params'] = total_params
    metrics['percent_equal'] = 100 * metrics['num_equal_params'] / total_params
    
    return metrics 