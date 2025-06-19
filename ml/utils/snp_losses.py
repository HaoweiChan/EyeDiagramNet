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

def unwrapped_phase_loss(pred_phase, target_phase, reduction='mean'):
    """Phase loss using cosine similarity to avoid wrapping issues."""
    cos_diff = torch.cos(pred_phase) * torch.cos(target_phase) + torch.sin(pred_phase) * torch.sin(target_phase)
    # Convert cosine similarity to loss (1 - cos_sim)
    loss = 1 - cos_diff
    if reduction == 'mean':
        return torch.mean(loss)
    elif reduction == 'sum':
        return torch.sum(loss)
    return loss

def complex_cosine_loss(pred_complex, target_complex, reduction='mean'):
    """
    Loss based on the cosine of the angle between complex numbers.
    This naturally handles phase wrapping and maintains magnitude information.
    """
    # Normalize to unit vectors
    pred_norm = pred_complex / (torch.abs(pred_complex) + 1e-8)
    target_norm = target_complex / (torch.abs(target_complex) + 1e-8)
    
    # Complex dot product (real part gives cosine similarity)
    cos_sim = torch.real(pred_norm * torch.conj(target_norm))
    
    # Convert to loss
    loss = 1 - cos_sim
    
    if reduction == 'mean':
        return torch.mean(loss)
    elif reduction == 'sum':
        return torch.sum(loss)
    return loss

def spectral_loss(pred_complex, target_complex, reduction='mean'):
    """
    Loss in both magnitude and phase using complex representation directly.
    More stable than separate magnitude/phase losses.
    """
    # Direct complex MSE
    diff = pred_complex - target_complex
    loss = torch.real(diff * torch.conj(diff))
    
    if reduction == 'mean':
        return torch.mean(loss)
    elif reduction == 'sum':
        return torch.sum(loss)
    return loss

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

class ImprovedSNPReconstructionLoss(nn.Module):
    """
    Improved loss for SNP reconstruction that addresses phase learning issues.
    Uses multiple complementary loss terms for stability.
    """
    
    def __init__(
        self,
        magnitude_weight: float = 1.0,
        phase_weight: float = 1.0,  # Increased default
        complex_weight: float = 0.5,
        spectral_weight: float = 0.2,
        use_unwrapped_phase: bool = True,
        latent_reg_type: str = 'l2',
        latent_reg_weight: float = 0.001,  # Reduced default
        gradient_penalty_weight: float = 0.0
    ):
        super().__init__()
        self.magnitude_weight = magnitude_weight
        self.phase_weight = phase_weight
        self.complex_weight = complex_weight
        self.spectral_weight = spectral_weight
        self.use_unwrapped_phase = use_unwrapped_phase
        self.latent_reg_type = latent_reg_type
        self.latent_reg_weight = latent_reg_weight
        self.gradient_penalty_weight = gradient_penalty_weight
    
    def forward(self, pred, target, embeddings=None):
        """
        Calculates total loss using multiple complementary terms.
        """
        loss_dict = {}
        total_loss = 0.0
        
        # 1. Magnitude loss (in dB scale)
        pred_mag = torch.abs(pred)
        target_mag = torch.abs(target)
        mag_loss = log_magnitude_loss(pred_mag, target_mag)
        loss_dict['log_magnitude_loss'] = mag_loss
        total_loss += self.magnitude_weight * mag_loss
        
        # 2. Phase loss (with optional unwrapping)
        pred_phase = torch.angle(pred)
        target_phase = torch.angle(target)
        if self.use_unwrapped_phase:
            ph_loss = unwrapped_phase_loss(pred_phase, target_phase)
        else:
            ph_loss = phase_loss(pred_phase, target_phase)
        loss_dict['phase_loss'] = ph_loss
        total_loss += self.phase_weight * ph_loss
        
        # 3. Complex cosine loss (helps with phase alignment)
        if self.complex_weight > 0:
            complex_loss = complex_cosine_loss(pred, target)
            loss_dict['complex_cosine_loss'] = complex_loss
            total_loss += self.complex_weight * complex_loss
        
        # 4. Direct spectral loss (complex MSE)
        if self.spectral_weight > 0:
            spec_loss = spectral_loss(pred, target)
            loss_dict['spectral_loss'] = spec_loss
            total_loss += self.spectral_weight * spec_loss
        
        # 5. Gradient penalty on phase (helps smooth phase predictions)
        if self.gradient_penalty_weight > 0 and pred.dim() >= 3:
            # Compute phase gradient along frequency dimension
            phase_grad = torch.diff(pred_phase, dim=2)
            target_phase_grad = torch.diff(target_phase, dim=2)
            
            # Penalize large deviations in phase gradient
            grad_penalty = F.mse_loss(phase_grad, target_phase_grad)
            loss_dict['gradient_penalty'] = grad_penalty
            total_loss += self.gradient_penalty_weight * grad_penalty
        
        # 6. Latent regularization
        if embeddings is not None and self.latent_reg_weight > 0:
            reg_loss = latent_regularization(
                embeddings, 
                self.latent_reg_type, 
                self.latent_reg_weight
            )
            loss_dict['regularization'] = reg_loss
            total_loss += reg_loss
        
        loss_dict['total'] = total_loss
        return total_loss, loss_dict 