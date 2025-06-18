import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple

def snp_reconstruction_error(
    pred: torch.Tensor, 
    target: torch.Tensor, 
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Calculate reconstruction error for SNP matrices
    
    Args:
        pred: Predicted SNP tensor (complex)
        target: Target SNP tensor (complex)
        reduction: 'mean', 'sum', or 'none'
    
    Returns:
        Reconstruction error
    """
    error = torch.abs(pred - target)
    
    if reduction == 'mean':
        return torch.mean(error)
    elif reduction == 'sum':
        return torch.sum(error)
    else:
        return error

def embedding_similarity(
    embed1: torch.Tensor, 
    embed2: torch.Tensor, 
    method: str = 'cosine'
) -> torch.Tensor:
    """
    Compare similarity between embeddings
    
    Args:
        embed1: First embedding tensor
        embed2: Second embedding tensor
        method: 'cosine', 'euclidean', or 'correlation'
    
    Returns:
        Similarity score
    """
    if method == 'cosine':
        # Cosine similarity
        embed1_norm = F.normalize(embed1, p=2, dim=-1)
        embed2_norm = F.normalize(embed2, p=2, dim=-1)
        similarity = torch.sum(embed1_norm * embed2_norm, dim=-1)
        return torch.mean(similarity)
    elif method == 'euclidean':
        # Negative euclidean distance (so higher is more similar)
        distance = torch.norm(embed1 - embed2, p=2, dim=-1)
        return -torch.mean(distance)
    elif method == 'correlation':
        # Pearson correlation
        embed1_centered = embed1 - embed1.mean(dim=-1, keepdim=True)
        embed2_centered = embed2 - embed2.mean(dim=-1, keepdim=True)
        
        numerator = torch.sum(embed1_centered * embed2_centered, dim=-1)
        denominator = torch.sqrt(
            torch.sum(embed1_centered ** 2, dim=-1) * 
            torch.sum(embed2_centered ** 2, dim=-1)
        )
        correlation = numerator / (denominator + 1e-8)
        return torch.mean(correlation)
    else:
        raise ValueError(f"Unknown similarity method: {method}")

def frequency_response_correlation(
    pred: torch.Tensor, 
    target: torch.Tensor,
    freq_dim: int = 2
) -> Dict[str, torch.Tensor]:
    """
    Validate frequency domain properties through correlation
    
    Args:
        pred: Predicted SNP tensor
        target: Target SNP tensor
        freq_dim: Dimension index for frequency
    
    Returns:
        Dictionary of correlation metrics
    """
    metrics = {}
    
    # Magnitude correlation
    pred_mag = torch.abs(pred)
    target_mag = torch.abs(target)
    
    # Flatten batch and spatial dimensions, keep frequency
    pred_mag_flat = pred_mag.flatten(0, freq_dim-1).flatten(freq_dim+1)
    target_mag_flat = target_mag.flatten(0, freq_dim-1).flatten(freq_dim+1)
    
    # Correlation per frequency point
    mag_corr = []
    for i in range(pred_mag_flat.size(freq_dim)):
        if freq_dim == 0:
            p = pred_mag_flat[i].flatten()
            t = target_mag_flat[i].flatten()
        else:
            p = pred_mag_flat.select(freq_dim, i).flatten()
            t = target_mag_flat.select(freq_dim, i).flatten()
        
        if p.numel() > 1:
            corr = torch.corrcoef(torch.stack([p, t]))[0, 1]
            mag_corr.append(corr)
    
    metrics['magnitude_correlation'] = torch.stack(mag_corr).mean()
    
    # Phase correlation
    pred_phase = torch.angle(pred)
    target_phase = torch.angle(target)
    
    # Unwrap phase difference
    phase_diff = pred_phase - target_phase
    phase_diff = torch.atan2(torch.sin(phase_diff), torch.cos(phase_diff))
    
    metrics['phase_rmse'] = torch.sqrt(torch.mean(phase_diff ** 2))
    metrics['phase_mae'] = torch.mean(torch.abs(phase_diff))
    
    return metrics

def snp_matrix_properties(snp: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Calculate properties of SNP matrices for validation
    
    Args:
        snp: SNP tensor of shape (B, D, F, P1, P2)
    
    Returns:
        Dictionary of matrix properties
    """
    properties = {}
    
    # Reciprocity check: S_ij should equal S_ji
    reciprocity_error = torch.mean(torch.abs(snp - snp.transpose(-1, -2)))
    properties['reciprocity_error'] = reciprocity_error
    
    # Passivity check: all singular values should be <= 1
    # Check a few frequency points to avoid memory issues
    b, d, f, p1, p2 = snp.shape
    sample_freqs = torch.linspace(0, f-1, min(10, f), dtype=torch.long)
    
    max_singular_values = []
    for freq_idx in sample_freqs:
        snp_at_freq = snp[:, :, freq_idx]  # (B, D, P1, P2)
        for i in range(b):
            for j in range(d):
                U, S, V = torch.linalg.svd(snp_at_freq[i, j])
                max_singular_values.append(S.max())
    
    properties['max_singular_value'] = torch.stack(max_singular_values).mean()
    properties['passivity_violation'] = (properties['max_singular_value'] > 1.0).float()
    
    # Energy conservation check
    # For passive networks, ||S||_F <= sqrt(P) where P is number of ports
    frobenius_norm = torch.norm(snp, p='fro', dim=(-2, -1))
    theoretical_max = np.sqrt(p1)
    properties['energy_ratio'] = torch.mean(frobenius_norm) / theoretical_max
    
    return properties

def reconstruction_quality_report(
    pred: torch.Tensor,
    target: torch.Tensor,
    embeddings_pred: Optional[torch.Tensor] = None,
    embeddings_target: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    """
    Comprehensive quality report for SNP reconstruction
    
    Args:
        pred: Predicted SNP tensor
        target: Target SNP tensor
        embeddings_pred: Optional predicted embeddings
        embeddings_target: Optional target embeddings
    
    Returns:
        Dictionary with all metrics
    """
    report = {}
    
    # Basic reconstruction errors
    report['mae'] = snp_reconstruction_error(pred, target, 'mean').item()
    report['mse'] = F.mse_loss(
        torch.view_as_real(pred), 
        torch.view_as_real(target)
    ).item()
    
    # Relative error
    rel_error = torch.abs(pred - target) / (torch.abs(target) + 1e-8)
    report['relative_error_mean'] = torch.mean(rel_error).item()
    report['relative_error_max'] = torch.max(rel_error).item()
    
    # Frequency response metrics
    freq_metrics = frequency_response_correlation(pred, target)
    for key, value in freq_metrics.items():
        report[f'freq_{key}'] = value.item()
    
    # Matrix properties
    pred_props = snp_matrix_properties(pred)
    target_props = snp_matrix_properties(target)
    
    for key in pred_props:
        report[f'pred_{key}'] = pred_props[key].item()
        report[f'target_{key}'] = target_props[key].item()
        report[f'diff_{key}'] = abs(pred_props[key] - target_props[key]).item()
    
    # Embedding similarity if provided
    if embeddings_pred is not None and embeddings_target is not None:
        report['embedding_cosine_sim'] = embedding_similarity(
            embeddings_pred.flatten(0, -2), 
            embeddings_target.flatten(0, -2),
            'cosine'
        ).item()
        report['embedding_euclidean_sim'] = embedding_similarity(
            embeddings_pred.flatten(0, -2), 
            embeddings_target.flatten(0, -2),
            'euclidean'
        ).item()
    
    return report 