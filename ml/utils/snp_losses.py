import torch
import torch.nn as nn
import torch.nn.functional as F

def complex_mse_loss(pred, target, reduction='mean'):
    """
    MSE loss for complex-valued tensors
    
    Args:
        pred: Predicted complex tensor
        target: Target complex tensor
        reduction: 'mean', 'sum', or 'none'
    
    Returns:
        Loss value
    """
    # Convert to real representation for loss calculation
    pred_real = torch.view_as_real(pred)
    target_real = torch.view_as_real(target)
    
    # Calculate MSE on both real and imaginary parts
    loss = F.mse_loss(pred_real, target_real, reduction=reduction)
    
    return loss

def magnitude_phase_loss(pred, target, magnitude_weight=0.7, phase_weight=0.3, reduction='mean'):
    """
    Separate losses for magnitude and phase components
    
    Args:
        pred: Predicted complex tensor
        target: Target complex tensor
        magnitude_weight: Weight for magnitude loss
        phase_weight: Weight for phase loss
        reduction: 'mean', 'sum', or 'none'
    
    Returns:
        Total weighted loss
    """
    # Calculate magnitudes
    pred_mag = torch.abs(pred)
    target_mag = torch.abs(target)
    
    # Calculate phases (angle in radians)
    pred_phase = torch.angle(pred)
    target_phase = torch.angle(target)
    
    # Magnitude loss (MSE)
    mag_loss = F.mse_loss(pred_mag, target_mag, reduction=reduction)
    
    # Phase loss (angular distance)
    # Handle phase wrapping: difference should be in [-pi, pi]
    phase_diff = pred_phase - target_phase
    phase_diff = torch.atan2(torch.sin(phase_diff), torch.cos(phase_diff))
    
    if reduction == 'mean':
        phase_loss = torch.mean(phase_diff ** 2)
    elif reduction == 'sum':
        phase_loss = torch.sum(phase_diff ** 2)
    else:
        phase_loss = phase_diff ** 2
    
    # Weighted combination
    total_loss = magnitude_weight * mag_loss + phase_weight * phase_loss
    
    return total_loss

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
    """Combined loss for SNP reconstruction with configurable components"""
    
    def __init__(
        self,
        loss_type='complex_mse',
        magnitude_weight=0.7,
        phase_weight=0.3,
        freq_weights=None,
        latent_reg_type='l2',
        latent_reg_weight=0.01
    ):
        super().__init__()
        self.loss_type = loss_type
        self.magnitude_weight = magnitude_weight
        self.phase_weight = phase_weight
        self.freq_weights = freq_weights
        self.latent_reg_type = latent_reg_type
        self.latent_reg_weight = latent_reg_weight
    
    def forward(self, pred, target, embeddings=None):
        """
        Calculate total loss
        
        Args:
            pred: Predicted SNP tensor
            target: Target SNP tensor
            embeddings: Optional latent embeddings for regularization
        
        Returns:
            total_loss: Combined reconstruction and regularization loss
            loss_dict: Dictionary with individual loss components
        """
        loss_dict = {}
        
        # Main reconstruction loss
        if self.loss_type == 'complex_mse':
            recon_loss = complex_mse_loss(pred, target)
        elif self.loss_type == 'magnitude_phase':
            recon_loss = magnitude_phase_loss(
                pred, target, 
                self.magnitude_weight, 
                self.phase_weight
            )
        elif self.loss_type == 'frequency_weighted':
            recon_loss = frequency_weighted_loss(pred, target, self.freq_weights)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        loss_dict['reconstruction'] = recon_loss
        total_loss = recon_loss
        
        # Add regularization if embeddings provided
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