import torch
import torch.nn as nn
import torch.nn.functional as F

def log_magnitude_loss(pred_mag, target_mag, reduction='mean'):
    """MSE loss on the log-magnitude (decibel) scale."""
    return F.mse_loss(
        20 * torch.log10(pred_mag + 1e-8),
        20 * torch.log10(target_mag + 1e-8),
        reduction=reduction
    )

def phase_loss(pred_phase, target_phase, reduction='mean'):
    """Mean squared error on the phase, handling angle wrapping."""
    phase_diff = pred_phase - target_phase
    # Wrap phase difference to the [-pi, pi] interval
    phase_diff = torch.atan2(torch.sin(phase_diff), torch.cos(phase_diff))
    if reduction == 'mean':
        return torch.mean(phase_diff ** 2)
    elif reduction == 'sum':
        return torch.sum(phase_diff ** 2)
    return phase_diff ** 2

def frequency_weighted_loss(pred, target, freq_weights=None, reduction='mean'):
    """
    Weight reconstruction loss by frequency importance
    
    Args:
        pred: Predicted complex tensor of shape (B, D, F, P1, P2)
        target: Target complex tensor of shape (B, D, F, P1, P2)
        freq_weights: Optional frequency weights of shape (F,)
        reduction: 'mean', 'sum', or 'none'
    
    Returns:
        Weighted loss value
    """
    # If no weights provided, use uniform weights
    if freq_weights is None:
        return complex_mse_loss(pred, target, reduction)
    
    # Calculate per-frequency MSE
    pred_real = torch.view_as_real(pred)
    target_real = torch.view_as_real(target)
    
    # MSE per frequency: (B, D, F, P1, P2, 2) -> (B, D, F)
    per_freq_loss = torch.mean((pred_real - target_real) ** 2, dim=(-3, -2, -1))
    
    # Apply frequency weights
    freq_weights = freq_weights.to(pred.device)
    weighted_loss = per_freq_loss * freq_weights.unsqueeze(0).unsqueeze(0)
    
    if reduction == 'mean':
        return torch.mean(weighted_loss)
    elif reduction == 'sum':
        return torch.sum(weighted_loss)
    else:
        return weighted_loss

def latent_regularization(embeddings, reg_type='l2', beta=1.0):
    """
    Regularization term for latent embeddings
    
    Args:
        embeddings: Latent embeddings tensor
        reg_type: 'l2' for L2 norm, 'kl' for KL divergence from standard normal
        beta: Regularization weight
    
    Returns:
        Regularization loss
    """
    if reg_type == 'l2':
        # L2 norm regularization
        reg_loss = torch.mean(torch.sum(embeddings ** 2, dim=-1))
    elif reg_type == 'kl':
        # KL divergence assuming embeddings are mean of Gaussian
        # KL(q||p) where p is N(0,1)
        # Assuming unit variance for q
        reg_loss = -0.5 * torch.mean(1 - embeddings ** 2)
    else:
        raise ValueError(f"Unknown regularization type: {reg_type}")
    
    return beta * reg_loss

class SNPReconstructionLoss(nn.Module):
    """
    Combined loss for SNP reconstruction with configurable components.
    Prioritizes log-magnitude and phase for better perceptual results.
    """
    
    def __init__(
        self,
        magnitude_weight: float = 1.0,
        phase_weight: float = 0.5,
        latent_reg_type: str = 'l2',
        latent_reg_weight: float = 0.01
    ):
        super().__init__()
        self.magnitude_weight = magnitude_weight
        self.phase_weight = phase_weight
        self.latent_reg_type = latent_reg_type
        self.latent_reg_weight = latent_reg_weight
    
    def forward(self, pred, target, embeddings=None):
        """
        Calculates total loss based on log-magnitude and phase.
        """
        loss_dict = {}
        
        # --- Main Reconstruction Loss ---
        pred_mag = torch.abs(pred)
        target_mag = torch.abs(target)
        mag_loss = log_magnitude_loss(pred_mag, target_mag)
        
        pred_phase = torch.angle(pred)
        target_phase = torch.angle(target)
        ph_loss = phase_loss(pred_phase, target_phase)
        
        recon_loss = (self.magnitude_weight * mag_loss) + (self.phase_weight * ph_loss)
        
        loss_dict['reconstruction'] = recon_loss
        loss_dict['log_magnitude_loss'] = mag_loss
        loss_dict['phase_loss'] = ph_loss
        total_loss = recon_loss
        
        # --- Latent Regularization ---
        if embeddings is not None and self.latent_reg_weight > 0:
            reg_loss = latent_regularization(
                embeddings, 
                self.latent_reg_type, 
                self.latent_reg_weight
            )
            loss_dict['regularization'] = reg_loss
            total_loss = total_loss + reg_loss
        
        loss_dict['total'] = total_loss
        return total_loss, loss_dict 